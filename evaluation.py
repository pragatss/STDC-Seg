#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from scipy import ndimage
from tqdm import tqdm
import math


def boundary_band(label_np, radius=3, ignore=255):
    # label_np: 2D int numpy array [H, W] for ONE image. Returns bool mask of the
    # band of pixels within `radius` of any class boundary.
    edge = np.zeros_like(label_np, dtype=bool)
    valid = label_np != ignore
    d = (label_np[1:, :] != label_np[:-1, :]) & valid[1:, :] & valid[:-1, :]
    edge[1:, :] |= d; edge[:-1, :] |= d
    d = (label_np[:, 1:] != label_np[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, 1:] |= d; edge[:, :-1] |= d
    if radius > 0:
        edge = ndimage.binary_dilation(edge, iterations=radius)
    edge &= valid
    return edge


class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:

            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]
  
            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()


class MscEvalBoundary(object):
    """Same as MscEvalV0, but the confusion matrix only counts pixels within
    `radius` of a ground-truth class boundary -- the thin region SBG targets,
    which whole-image mIoU averages away."""

    def __init__(self, scale=0.5, ignore_label=255, radius=3):
        self.ignore_label = ignore_label
        self.scale = scale
        self.radius = radius

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        band_pixel_count = 0
        total_pixel_count = 0
        if dist.is_initialized() and dist.get_rank() != 0:
            diter = enumerate(dl)
        else:
            diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:

            N, _, H, W = label.shape

            label = label.squeeze(1).cuda()
            size = label.size()[-2:]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

            logits = net(imgs)[0]

            logits = F.interpolate(logits, size=size,
                    mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            label_np = label.cpu().numpy()
            band_np = np.stack([
                boundary_band(label_np[n], radius=self.radius, ignore=self.ignore_label)
                for n in range(label_np.shape[0])
            ])
            # torch 1.1.0 has no real bool dtype for comparisons -- `label != ignore`
            # is uint8 (Byte), and Byte & Bool is not allowed. Match that dtype.
            band = torch.from_numpy(band_np.astype(np.uint8)).to(label.device)

            band_pixel_count += band.long().sum().item()
            total_pixel_count += band.numel()

            keep = (label != self.ignore_label) & band
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes).float()
        if dist.is_initialized():
            dist.all_reduce(hist, dist.ReduceOp.SUM)
        band_fraction = band_pixel_count / total_pixel_count
        print(f"boundary band (radius={self.radius}) covers {band_fraction*100:.2f}% of pixels "
              f"(sanity check: expect roughly 5-8% at radius=3 on Cityscapes)")
        if band_fraction > 0.5:
            print("WARNING: band covers >50% of pixels -- boundary_band logic is likely broken, "
                  "this result is not a boundary-restricted metric.")
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item(), ious


def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, use_sbg=False, use_variance=False, use_semantic=False, semantic_source='lr'):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    print('use_sbg', use_sbg)
    print('use_variance', use_variance)
    print('use_semantic', use_semantic)
    print('semantic_source', semantic_source)
    ## dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 19
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
     use_conv_last=use_conv_last, use_sbg=use_sbg, use_variance=use_variance,
     use_semantic=use_semantic, semantic_source=semantic_source)
    missing, unexpected = net.load_state_dict(torch.load(respth), strict=False)
    # Checkpoints trained before SBG existed in the code at all (not just use_sbg=False,
    # but self.sbg never constructed) have no sbg.* keys. That's fine -- sbg is unused
    # in forward() when use_sbg=False anyway. Anything else missing/unexpected is a
    # real architecture mismatch and should not be silently ignored.
    bad_missing = [k for k in missing if not k.startswith('sbg.')]
    if bad_missing or unexpected:
        raise RuntimeError(f"unexpected state_dict mismatch -- missing (non-sbg): {bad_missing}, "
                            f"unexpected: {unexpected}")
    if missing:
        print(f"checkpoint has no sbg.* weights (pre-SBG checkpoint) -- {len(missing)} keys left at random init, unused since use_sbg={use_sbg}")
    assert net.use_sbg == use_sbg and net.sbg.use_variance == use_variance and net.sbg.use_semantic == use_semantic, \
        f"flag mismatch after load_state_dict: net.use_sbg={net.use_sbg}, net.sbg.use_variance={net.sbg.use_variance}, net.sbg.use_semantic={net.sbg.use_semantic}"
    print(f"net.use_sbg={net.use_sbg}  net.sbg.use_variance={net.sbg.use_variance}  net.sbg.use_semantic={net.sbg.use_semantic}")
    net.cuda()
    net.eval()


    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        mIOU = single_scale(net, dl, 19)
    logger = logging.getLogger()
    logger.info('mIOU is: %s\n', mIOU)


CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


def evaluate_boundary(checkpoint_path, backbone, scale=0.75, radius=3,
                       use_boundary_2=False, use_boundary_4=False, use_boundary_8=True,
                       use_boundary_16=False, use_sbg=False, use_variance=False,
                       use_semantic=False, semantic_source='lr', dspth='./data'):
    # NOTE: added use_sbg here even though it wasn't in the requested signature --
    # BiSeNet needs it to reconstruct the right forward-pass wiring (same reason
    # evaluatev0's use_sbg exists: checkpoints trained before use_sbg existed as a
    # bypass flag need use_sbg=True or load_state_dict silently loads the wrong path).
    print('checkpoint_path', checkpoint_path)
    print('backbone', backbone)
    print('scale', scale)
    print('radius', radius)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    print('use_sbg', use_sbg)
    print('use_variance', use_variance)
    print('use_semantic', use_semantic)
    print('semantic_source', semantic_source)

    n_classes = 19
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
     use_sbg=use_sbg, use_variance=use_variance, use_semantic=use_semantic,
     semantic_source=semantic_source)
    missing, unexpected = net.load_state_dict(torch.load(checkpoint_path), strict=False)
    # Checkpoints trained before SBG existed in the code at all (not just use_sbg=False,
    # but self.sbg never constructed) have no sbg.* keys. That's fine -- sbg is unused
    # in forward() when use_sbg=False anyway. Anything else missing/unexpected is a
    # real architecture mismatch and should not be silently ignored.
    bad_missing = [k for k in missing if not k.startswith('sbg.')]
    if bad_missing or unexpected:
        raise RuntimeError(f"unexpected state_dict mismatch -- missing (non-sbg): {bad_missing}, "
                            f"unexpected: {unexpected}")
    if missing:
        print(f"checkpoint has no sbg.* weights (pre-SBG checkpoint) -- {len(missing)} keys left at random init, unused since use_sbg={use_sbg}")
    assert net.use_sbg == use_sbg and net.sbg.use_variance == use_variance and net.sbg.use_semantic == use_semantic, \
        f"flag mismatch after load_state_dict: net.use_sbg={net.use_sbg}, net.sbg.use_variance={net.sbg.use_variance}, net.sbg.use_semantic={net.sbg.use_semantic}"
    print(f"net.use_sbg={net.use_sbg}  net.sbg.use_variance={net.sbg.use_variance}  net.sbg.use_semantic={net.sbg.use_semantic}")
    net.cuda()
    net.eval()

    with torch.no_grad():
        full_miou = MscEvalV0(scale=scale)(net, dl, n_classes)
        boundary_miou, boundary_ious = MscEvalBoundary(scale=scale, radius=radius)(net, dl, n_classes)

    return full_miou, boundary_miou, boundary_ious


class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes = 19,
            lb_ignore = 255,
            cropsize = 1024,
            flip = True,
            *args, **kwargs):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0]-H, size[1]-W
        hst, hed = margin_h//2, margin_h//2+H
        wst, wed = margin_w//2, margin_w//2+W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5/6.
        N, C, H, W = im.size()
        long_size, short_size = (H,W) if H>W else (W,H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize*stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W-cropsize)/stride)+1
            n_y = math.ceil((H-cropsize)/stride)+1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride*iy+cropsize), min(W, stride*ix+cropsize)
                    hst, wst = hed-cropsize, wed-cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, H, W = im.size()
        new_hw = [int(H*scale), int(W*scale)]
        im = F.interpolate(im, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(im)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb==ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank()==0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                # prob = self.scale_crop_eval(imgs, sc)
                prob = self.eval_chip(imgs)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0)+np.sum(hist, axis=1)-np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(respth='./resv1_catnet/pths/', dspth='./data'):
    ## logger
    logger = logging.getLogger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)

    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 5
    n_workers = 2
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=[1], flip = False)

    ## eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))


def run_boundary_eval(checkpoint_path, backbone, use_variance, use_semantic, use_sbg=True, scale=0.75, semantic_source='lr'):
    # Reminder: pass the SAME use_sbg/use_variance/use_semantic/semantic_source this
    # checkpoint was trained with. A use_variance mismatch loads without error but
    # silently evaluates the wrong architecture (see the ARM B comment above evaluatev0's
    # call for why); a semantic_source mismatch fails loudly at load_state_dict.
    full_miou, boundary_miou_r1, _ = evaluate_boundary(
        checkpoint_path, backbone, scale=scale, radius=1,
        use_sbg=use_sbg, use_variance=use_variance, use_semantic=use_semantic,
        semantic_source=semantic_source)
    _, boundary_miou_r3, boundary_ious_r3 = evaluate_boundary(
        checkpoint_path, backbone, scale=scale, radius=3,
        use_sbg=use_sbg, use_variance=use_variance, use_semantic=use_semantic,
        semantic_source=semantic_source)

    print()
    print("=" * 64)
    print(f"full-image mIoU:              {full_miou:.4f}")
    print(f"boundary-region IoU (r=1):    {boundary_miou_r1:.4f}")
    print(f"boundary-region IoU (r=3):    {boundary_miou_r3:.4f}")
    print("=" * 64)
    print("per-class boundary IoU (r=3):")
    for name, iou in sorted(zip(CITYSCAPES_CLASSES, boundary_ious_r3.tolist()), key=lambda x: x[1]):
        print(f"  {name:15s} {iou:.4f}")


if __name__ == "__main__":
    log_dir = 'evaluation_logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    setup_logger(log_dir)
    
    #STDC1-Seg50 mIoU 0.7222
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet813', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #STDC1-Seg75 mIoU 0.7450
    # evaluatev0('./checkpoints/STDC1-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet813', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


    #STDC2-Seg50 mIoU 0.7424
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU50.pth', dspth='./data', backbone='STDCNet1446', scale=0.5, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    # #STDC2-Seg75 mIoU 0.7704
    # evaluatev0('./checkpoints/STDC2-Seg/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    # use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

    #baseline run 
    # evaluation: 0.7599517703056335
    # evaluatev0('./checkpoints/train_STDC2-Seg/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    #     use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)


#baseline run 
    # evaluation: 0.7599517703056335
    # evaluatev0('./checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    #     use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
    
    # # Boundary-region IoU (D2-style diagnostic). Flags confirmed to match the
    # # evaluatev0 run above (same checkpoint, same use_sbg/use_variance/use_semantic).
    # run_boundary_eval(
    #     checkpoint_path='./checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth',
    #     backbone='STDCNet1446',
    #     use_sbg=False, use_variance=False, use_semantic=False)

# ARM B checkpoints/train_STDC2-Seg/pths/model_maxmIOU75.pth
    # Trained before use_sbg existed as a bypass flag, back when forward() always routed
    # through self.sbg (appearance branch always on) -- so use_sbg=True here, NOT the
    # (now-default) False, or this will silently evaluate the wrong computational path.

    # evaluatev0('./checkpoints/train_STDC2-Seg_ARMB/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75,
    #     use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False,
    #     use_sbg=True, use_variance=False, use_semantic=False)
    # run_boundary_eval(
    #     checkpoint_path='./checkpoints/train_STDC2-Seg_ARMB/pths/model_maxmIOU75.pth',
    #     backbone='STDCNet1446',
    #     use_sbg=True, use_variance=False, use_semantic=False)

# # ARM C checkpoints/train_STDC2-Seg-ARM-C/pths/model_maxmIOU75.pth
#     # Trained with use_sbg=True, use_variance=True, use_semantic=False (confirmed from
#     # its training log) -- use_variance=True here, NOT False, or this silently evaluates
#     # Arm C's weights through Arm B's forward path (use_variance doesn't change any
#     # parameter shapes, so load_state_dict succeeds either way with no error).

#     evaluatev0('./checkpoints/train_STDC2-Seg-ARM-C/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75,
#         use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False,
#         use_sbg=True, use_variance=True, use_semantic=False)
#     run_boundary_eval(
#         checkpoint_path='./checkpoints/train_STDC2-Seg-ARM-C/pths/model_maxmIOU75.pth',
#         backbone='STDCNet1446',
#         use_sbg=True, use_variance=True, use_semantic=False)

# ARM D checkpoints/train_STDC2-Seg-ARM-D/pths/model_maxmIOU75.pth
    # Trained with use_sbg=True, use_variance=True, use_semantic=True (confirmed from
    # its training log) -- use_semantic=True here, NOT False. Unlike use_variance, a
    # use_semantic mismatch changes conv_sem's existence and conv_fuse's channel count,
    # so getting this wrong fails loudly at load_state_dict rather than silently.

    # evaluatev0('./checkpoints/train_STDC2-Seg-ARM-D/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75,
    #     use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False,
    #     use_sbg=True, use_variance=True, use_semantic=True)
    # run_boundary_eval(
    #     checkpoint_path='./checkpoints/train_STDC2-Seg-ARM-D/pths/model_maxmIOU75.pth',
    #     backbone='STDCNet1446',
    #     use_sbg=True, use_variance=True, use_semantic=True)

# ARM E checkpoints/train_STDC2-Seg-ARM-E/pths/model_maxmIOU75.pth
    # Trained with use_sbg=True, use_variance=True, use_semantic=True, semantic_source='hr'
    # (confirmed from its training log). semantic_source='hr' is REQUIRED here -- it builds
    # sbg_sem_head; omitting it (lr default) makes the checkpoint's sbg_sem_head.* keys
    # 'unexpected' and the load guard raises loudly (by design, not silently).

    evaluatev0('./checkpoints/train_STDC2-Seg-ARM-E/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75,
        use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False,
        use_sbg=True, use_variance=True, use_semantic=True, semantic_source='hr')
    run_boundary_eval(
        checkpoint_path='./checkpoints/train_STDC2-Seg-ARM-E/pths/model_maxmIOU75.pth',
        backbone='STDCNet1446',
        use_sbg=True, use_variance=True, use_semantic=True, semantic_source='hr')


    


