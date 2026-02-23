#!/usr/bin/python
# -*- encoding: utf-8 -*-

# ============================================================
# STDC-Seg Training Script
# Paper: "Rethinking BiSeNet for Real-Time Semantic Segmentation"
#         Fan et al., CVPR 2021
#
# What this script does in plain English:
#   1. Loads Cityscapes images (street scenes) with their pixel-level labels.
#   2. Builds the STDC-Seg network (backbone + Context Path + FFM).
#   3. Trains with two kinds of loss:
#        a) Segmentation loss  -- "did we label each pixel correctly?"
#        b) Boundary loss      -- "did we correctly find the edges between objects?"
#   4. Saves the best model based on mIOU (a standard accuracy metric).
# ============================================================

from logger import setup_logger
from models.model_stages import BiSeNet       # The full STDC-Seg network (Paper Fig. 4)
from cityscapes import CityScapes             # Cityscapes dataset loader
from loss.loss import OhemCELoss              # Segmentation loss (OHEM cross-entropy)
from loss.detail_loss import DetailAggregateLoss  # Boundary loss (Paper §3.3)
from evaluation import MscEvalV0              # mIOU evaluator
from optimizer_loss import Optimizer          # SGD with warmup + poly LR decay (Paper §4)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse

logger = logging.getLogger()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    parse.add_argument(
            '--n_workers_train',
            dest = 'n_workers_train',
            type = int,
            default = 8,
            )
    parse.add_argument(
            '--n_workers_val',
            dest = 'n_workers_val',
            type = int,
            default = 0,
            )
    parse.add_argument(
            '--n_img_per_gpu',
            dest = 'n_img_per_gpu',
            type = int,
            default = 16,
            )
    parse.add_argument(
            '--max_iter',
            dest = 'max_iter',
            type = int,
            default = 40000,
            )
    parse.add_argument(
            '--save_iter_sep',
            dest = 'save_iter_sep',
            type = int,
            default = 1000,
            )
    parse.add_argument(
            '--warmup_steps',
            dest = 'warmup_steps',
            type = int,
            default = 1000,
            )      
    parse.add_argument(
            '--mode',
            dest = 'mode',
            type = str,
            default = 'train',
            )
    parse.add_argument(
            '--ckpt',
            dest = 'ckpt',
            type = str,
            default = None,
            )
    parse.add_argument(
            '--respath',
            dest = 'respath',
            type = str,
            default = None,
            )
    parse.add_argument(
            '--backbone',
            dest = 'backbone',
            type = str,
            default = 'CatNetSmall',
            )
    parse.add_argument(
            '--pretrain_path',
            dest = 'pretrain_path',
            type = str,
            default = '',
            )
    parse.add_argument(
            '--use_conv_last',
            dest = 'use_conv_last',
            type = str2bool,
            default = False,
            )
    # -----------------------------------------------------------------
    # Boundary supervision flags (Paper §3.3 — Detail Aggregation Learning)
    #
    # The paper's big idea: instead of keeping a whole separate "Detail Branch"
    # running at inference time (which is slow), we teach the network about
    # object edges ONLY during training using these boundary loss flags.
    #
    # Each flag turns on a boundary prediction head at a specific image scale:
    #   use_boundary_2  → supervise edges at 1/2  of the original image size
    #   use_boundary_4  → supervise edges at 1/4  of the original image size
    #   use_boundary_8  → supervise edges at 1/8  of the original image size  ← most common
    #   use_boundary_16 → supervise edges at 1/16 of the original image size
    #
    # At inference time these heads are simply not called — the network
    # still benefits from the boundary-aware training without any extra cost.
    # -----------------------------------------------------------------
    parse.add_argument(
            '--use_boundary_2',
            dest = 'use_boundary_2',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_4',
            dest = 'use_boundary_4',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_8',
            dest = 'use_boundary_8',
            type = str2bool,
            default = False,
            )
    parse.add_argument(
            '--use_boundary_16',
            dest = 'use_boundary_16',
            type = str2bool,
            default = False,
            )
    return parse.parse_args()


def train():
    args = parse_args()
    
    save_pth_path = os.path.join(args.respath, 'pths')
    dspth = './data'
    
    # print(save_pth_path)
    # print(osp.exists(save_pth_path))
    # if not osp.exists(save_pth_path) and dist.get_rank()==0: 
    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)
    
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:33274',
                world_size = torch.cuda.device_count(),
                rank=args.local_rank
                )
    
    setup_logger(args.respath)
    ## dataset
    n_classes = 19
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2
    
    mode = args.mode

    # Input images are cropped to 1024×512 during training (Paper §4 / Table 4 training settings)
    cropsize = [1024, 512]

    # Multi-scale random resizing for data augmentation (Paper §4).
    # Think of it like showing the network the same scene from different distances —
    # from very zoomed-out (12.5% of original size) to slightly zoomed-in (150%).
    # This helps the model recognise objects at any size in the real world.
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    if dist.get_rank()==0: 
        logger.info('n_workers_train: {}'.format(n_workers_train))
        logger.info('n_workers_val: {}'.format(n_workers_val))
        logger.info('use_boundary_2: {}'.format(use_boundary_2))
        logger.info('use_boundary_4: {}'.format(use_boundary_4))
        logger.info('use_boundary_8: {}'.format(use_boundary_8))
        logger.info('use_boundary_16: {}'.format(use_boundary_16))
        logger.info('mode: {}'.format(args.mode))
    
    
    ds = CityScapes(dspth, cropsize=cropsize, mode=mode, randomscale=randomscale)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = n_workers_train,
                    pin_memory = False,
                    drop_last = True)
    # exit(0)
    dsval = CityScapes(dspth, mode='val', randomscale=randomscale)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = DataLoader(dsval,
                    batch_size = 2,
                    shuffle = False,
                    sampler = sampler_val,
                    num_workers = n_workers_val,
                    drop_last = False)

    ## model
    # ---------------------------------------------------------------
    # Build the full STDC-Seg network (Paper Fig. 4).
    #
    # backbone        → which STDC encoder to use: STDCNet813 (fast) or STDCNet1446 (accurate)
    # n_classes       → 19 Cityscapes categories (road, car, person, sky, ...)
    # pretrain_model  → start from ImageNet-pretrained STDC weights to speed up training
    # use_boundary_*  → turn on the boundary prediction heads described in Paper §3.3
    # ---------------------------------------------------------------
    ignore_idx = 255  # Cityscapes uses label 255 for "don't care" pixels — we ignore them in the loss
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8, 
    use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)

    if not args.ckpt is None:
        # Resume training from a previously saved checkpoint
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    net.cuda()
    net.train()
    # Spread training across multiple GPUs (one process per GPU)
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank,
            find_unused_parameters=True
            )

    # ---------------------------------------------------------------
    # Segmentation loss: OHEM Cross-Entropy (Paper §4 training details)
    #
    # Standard cross-entropy asks "how wrong were we on every pixel?"
    # OHEM (Online Hard Example Mining) goes further: it focuses training
    # on the pixels the model is MOST confused about (confidence < 0.7),
    # ignoring the easy pixels.  This forces the model to improve on
    # tricky areas like thin objects or ambiguous boundaries.
    #
    # Three separate instances supervise the three segmentation outputs:
    #   criteria_p  → main head (fused features at 1/8 scale, Paper §3.2)
    #   criteria_16 → auxiliary head (Context Path output at 1/8, deep supervision)
    #   criteria_32 → auxiliary head (Context Path output at 1/16, deep supervision)
    # The two auxiliary heads are only used during training and help the
    # network learn better intermediate representations (Paper §3.2).
    # ---------------------------------------------------------------
    score_thres = 0.7   # pixels where model confidence < 70% are considered "hard"
    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//16  # minimum number of hard pixels per batch
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    
    # ---------------------------------------------------------------
    # Boundary loss: Detail Aggregation Loss (Paper §3.3)
    #
    # This loss teaches the network to correctly predict WHERE the edges
    # between objects are.  It uses a Laplacian filter on the ground-truth
    # masks to automatically figure out which pixels are on a boundary,
    # then penalises the network if it misses those edges.
    # It combines two loss terms:
    #   BCE  → per-pixel binary classification (is this pixel a boundary or not?)
    #   Dice → overlap-based loss that handles the imbalance between the tiny
    #          number of edge pixels vs. the large number of non-edge pixels.
    # ---------------------------------------------------------------
    boundary_loss_func = DetailAggregateLoss()
    # ---------------------------------------------------------------
    # Optimiser: SGD with Warmup + Polynomial LR Decay (Paper §4)
    #
    # Learning rate (LR) controls how big each update step is.
    # Training uses two phases:
    #   1. Warmup (first 1000 steps): LR gradually rises from 1e-5 to 1e-2.
    #      Starting with a tiny LR prevents the randomly-initialised heads
    #      from destabilising the pretrained backbone early in training.
    #   2. Polynomial decay (rest of training): LR slowly decreases following
    #      LR = lr_start * (1 - iter/max_iter)^0.9  so the model makes finer
    #      and finer adjustments as it converges.
    #
    # The boundary_loss_func is also passed in because its fuse_kernel
    # (a learnable parameter) needs to be optimised alongside the network.
    # ---------------------------------------------------------------
    maxmIOU50 = 0.   # track the best model by mIOU at IoU threshold 0.50
    maxmIOU75 = 0.   # track the best model by mIOU at IoU threshold 0.75
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2      # peak learning rate after warmup
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9          # controls how steeply the LR decays (polynomial exponent)
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5   # tiny LR at the very start of warmup

    if dist.get_rank()==0: 
        print('max_iter: ', max_iter)
        print('save_iter_sep: ', save_iter_sep)
        print('warmup_steps: ', warmup_steps)
    optim = Optimizer(
            model = net.module,
            loss = boundary_loss_func,   # include the learnable fuse_kernel in optimisation
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    
    # ---------------------------------------------------------------
    # Main training loop (Paper §4)
    # Each iteration:
    #   1. Run a batch of images through the network (forward pass).
    #   2. Compute how wrong the predictions were (loss).
    #   3. Work out how to nudge each weight to reduce that error (backward pass).
    #   4. Update the weights (optimiser step).
    # ---------------------------------------------------------------
    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()  # clear gradients from the previous iteration

        # ---------------------------------------------------------------
        # Forward pass (Paper Fig. 4 — full STDC-Seg network)
        #
        # The network always returns three segmentation outputs:
        #   out   → main prediction from the Feature Fusion Module (1/8 scale)
        #   out16 → auxiliary prediction from Context Path (1/8 scale)
        #   out32 → auxiliary prediction from Context Path (1/16 scale)
        #
        # If boundary supervision is enabled, it also returns raw boundary
        # score maps (detail2/4/8) from the shallow STDC backbone stages.
        # These boundary maps are ONLY used to compute the boundary loss below;
        # they do not feed into the final segmentation output (Paper §3.3).
        # ---------------------------------------------------------------
        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(im)
        
        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
            out, out16, out32, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
            out, out16, out32 = net(im)

        # ---------------------------------------------------------------
        # Segmentation loss — "how wrong were we at labelling each pixel?"
        # (OHEM cross-entropy, Paper §4 training details)
        #
        # lossp  → error on the main output (full fusion, most important)
        # loss2  → error on the 1/8 auxiliary output  (deep supervision)
        # loss3  → error on the 1/16 auxiliary output (deep supervision)
        #
        # Deep supervision means we penalise intermediate outputs too, not
        # just the final one.  This pushes gradients deeper into the network
        # and helps the backbone learn better features faster.
        # ---------------------------------------------------------------
        lossp = criteria_p(out, lb)      # main segmentation head loss
        loss2 = criteria_16(out16, lb)   # auxiliary head at Context Path 1/8
        loss3 = criteria_32(out32, lb)   # auxiliary head at Context Path 1/16
        
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.
        
        # ---------------------------------------------------------------
        # Boundary loss — "did we correctly find the edges between objects?"
        # (Detail Aggregation Learning, Paper §3.3 / Eq. 3)
        #
        # For each enabled scale we compare the network's boundary predictions
        # (detail2/4/8) against automatically-generated GT edge maps.
        # The GT edges are computed by running a Laplacian filter on the
        # ground-truth segmentation mask (see DetailAggregateLoss in detail_loss.py).
        #
        # Accumulate BCE and Dice terms across all active scales.
        # ---------------------------------------------------------------
        if use_boundary_2: 
            boundery_bce_loss2,  boundery_dice_loss2 = boundary_loss_func(detail2, lb)
            boundery_bce_loss += boundery_bce_loss2
            boundery_dice_loss += boundery_dice_loss2
        
        if use_boundary_4:
            boundery_bce_loss4,  boundery_dice_loss4 = boundary_loss_func(detail4, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4

        if use_boundary_8:
            boundery_bce_loss8,  boundery_dice_loss8 = boundary_loss_func(detail8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8

        # ---------------------------------------------------------------
        # Total loss = segmentation losses + boundary losses (Paper Eq. 3 / §4)
        #
        # loss = L_seg(main) + L_seg(aux16) + L_seg(aux32)
        #      + L_boundary_BCE + L_boundary_Dice
        #
        # All terms are weighted equally (no separate lambda coefficients).
        # The boundary terms only contribute during training; at inference
        # the boundary heads are not called so there is zero extra cost.
        # ---------------------------------------------------------------
        loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
        
        loss.backward()   # compute gradients via backpropagation
        optim.step()      # update all weights (network + fuse_kernel)

        loss_avg.append(loss.item())

        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())

        ## print training log message
        if (it+1)%msg_iter==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'boundery_bce_loss: {boundery_bce_loss:.4f}',
                'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it = it+1,
                max_it = max_iter,
                lr = lr,
                loss = loss_avg,
                boundery_bce_loss = loss_boundery_bce_avg,
                boundery_dice_loss = loss_boundery_dice_avg,
                time = t_intv,
                eta = eta
            )
            
            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed
            # print(boundary_loss_func.get_params())
        if (it+1)%save_iter_sep==0:# and it != 0:
            
            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')
            
            # Switch off dropout and batch-norm running-stat updates during eval
            net.eval()

            # ---------------------------------------------------------------
            # Evaluation: compute mIOU on the validation set (Paper Table 4)
            #
            # mIOU (mean Intersection over Union) measures how well the
            # predicted label mask overlaps the ground-truth mask, averaged
            # across all 19 Cityscapes classes.  Higher = better.
            #
            # Two variants are measured:
            #   mIOU50: standard evaluation at full resolution (scale=1.0)
            #   mIOU75: evaluation at 75% of input resolution (scale=0.75)
            #           — tests how well the model generalises to smaller scales
            # ---------------------------------------------------------------
            logger.info('compute the mIOU')
            with torch.no_grad():  # no gradients needed during evaluation
                single_scale1 = MscEvalV0()            # full-resolution eval
                mIOU50 = single_scale1(net, dlval, n_classes)

                single_scale2= MscEvalV0(scale=0.75)   # 75% resolution eval
                mIOU75 = single_scale2(net, dlval, n_classes)


            save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'
            .format(it+1, str(round(mIOU50,4)), str(round(mIOU75,4))))
            
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank()==0: 
                torch.save(state, save_pth)

            logger.info('training iteration {}, model saved to: {}'.format(it+1, save_pth))

            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = osp.join(save_pth_path, 'model_maxmIOU50.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank()==0: 
                    torch.save(state, save_pth)
                    
                logger.info('max mIOU model saved to: {}'.format(save_pth))
            
            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
                save_pth = osp.join(save_pth_path, 'model_maxmIOU75.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank()==0: torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))
            
            logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))
            logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))

            net.train()
    
    ## dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank()==0: torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)

if __name__ == "__main__":
    train()
