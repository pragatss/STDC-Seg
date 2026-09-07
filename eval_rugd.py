#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Score RUGD checkpoints with support-filtered mIoU.

RUGD's val split is two video recordings (park-8, trail-5), so several of the
24 classes land under a thousandth of a percent of the pixels there -- person
is 0.0014%, sign 0.0033%. Averaging those into mIoU equally with tree (43%)
means a handful of pixels swings the headline number as much as a whole
class, which is what makes the per-checkpoint metric so noisy. So report
three numbers: mIoU over every class with any support (what train.py logs),
mIoU restricted to classes above --min_support of val pixels, and the
frequency-weighted mIoU.
"""

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.model_stages import BiSeNet
from rugd import RUGD, RUGD_CLASSES


def build_hist(net, dl, n_classes, scale):
    hist = torch.zeros(n_classes, n_classes).cuda()
    with torch.no_grad():
        for imgs, label in dl:
            label = label.squeeze(1).cuda()
            size = label.shape[-2:]
            imgs = imgs.cuda()
            N, C, H, W = imgs.size()
            imgs = F.interpolate(imgs, [int(H * scale), int(W * scale)],
                                 mode='bilinear', align_corners=True)
            logits = net(imgs)[0]
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
            preds = torch.argmax(logits, dim=1)
            keep = label != 255
            hist += torch.bincount(label[keep] * n_classes + preds[keep],
                                   minlength=n_classes ** 2).view(n_classes, n_classes).float()
    return hist


def score(hist, min_support):
    """hist -> (miou_all, miou_supported, miou_freqweighted, per-class arrays)"""
    gt = hist.sum(dim=1)
    union = hist.sum(dim=0) + hist.sum(dim=1) - hist.diag()
    ious = (hist.diag() / union).cpu().numpy()
    gt = gt.cpu().numpy()
    union = union.cpu().numpy()
    support = gt / gt.sum()

    scored = union > 0                          # matches train.py / MscEvalV0
    supported = scored & (support >= min_support)
    miou_all = float(np.mean(ious[scored]))
    miou_sup = float(np.mean(ious[supported]))
    ## frequency-weighted over the same supported set, renormalised
    w = support[supported] / support[supported].sum()
    miou_fw = float(np.sum(ious[supported] * w))
    return miou_all, miou_sup, miou_fw, ious, support, scored, supported


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--pths', type=str, nargs='+', required=True,
                       help='checkpoint .pth files to score, in order')
    parse.add_argument('--rootpth', type=str, default='./data/rugd')
    parse.add_argument('--backbone', type=str, default='STDCNet1446')
    parse.add_argument('--scale', type=float, default=0.75)
    parse.add_argument('--min_support', type=float, default=0.001,
                       help='fraction of val pixels a class needs to enter the supported mean')
    parse.add_argument('--use_brh', action='store_true')
    parse.add_argument('--brh_mid', type=int, default=64)
    parse.add_argument('--batchsize', type=int, default=4)
    parse.add_argument('--n_workers', type=int, default=4)
    parse.add_argument('--per_class', action='store_true',
                       help='also print the per-class table for the last checkpoint')
    args = parse.parse_args()

    n_classes = len(RUGD_CLASSES)
    ds = RUGD(args.rootpth, mode='val')
    dl = DataLoader(ds, batch_size=args.batchsize, shuffle=False,
                    num_workers=args.n_workers, drop_last=False)

    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model='',
                  use_boundary_8=True, use_brh=args.use_brh, brh_mid=args.brh_mid)
    net.cuda().eval()

    print('%-10s %10s %10s %10s' % ('iter', 'mIoU_all', 'mIoU_sup', 'mIoU_fw'))
    rows = []
    last = None
    for pth in args.pths:
        net.load_state_dict(torch.load(pth, map_location='cpu'))
        net.cuda().eval()
        hist = build_hist(net, dl, n_classes, args.scale)
        m_all, m_sup, m_fw, ious, support, scored, supported = score(hist, args.min_support)
        tag = osp.basename(pth).replace('model_iter', '').split('_')[0].replace('.pth', '')
        print('%-10s %10.4f %10.4f %10.4f' % (tag, m_all, m_sup, m_fw))
        rows.append((m_all, m_sup, m_fw))
        last = (ious, support, scored, supported)

    if len(rows) > 1:
        a = np.array(rows)
        print('\n%-10s %10s %10s %10s' % ('', 'mIoU_all', 'mIoU_sup', 'mIoU_fw'))
        print('%-10s %10.4f %10.4f %10.4f' % ('mean', a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()))
        print('%-10s %10.4f %10.4f %10.4f' % ('std', a[:, 0].std(), a[:, 1].std(), a[:, 2].std()))
        print('%-10s %10.4f %10.4f %10.4f' % ('max', a[:, 0].max(), a[:, 1].max(), a[:, 2].max()))
        print('%-10s %10.4f %10.4f %10.4f' % ('max-mean', a[:, 0].max() - a[:, 0].mean(),
                                              a[:, 1].max() - a[:, 1].mean(),
                                              a[:, 2].max() - a[:, 2].mean()))

    if args.per_class and last is not None:
        ious, support, scored, supported = last
        print('\n%-14s %9s %8s  %s' % ('class', 'val GT%', 'IoU', 'in supported mean'))
        for c in np.argsort(-support):
            if not scored[c]:
                print('%-14s %9.4f %8s  %s' % (RUGD_CLASSES[c], 100 * support[c], 'n/a', 'absent from val'))
            else:
                print('%-14s %9.4f %8.3f  %s' % (RUGD_CLASSES[c], 100 * support[c], ious[c],
                                                 'yes' if supported[c] else 'NO  (below threshold)'))


if __name__ == '__main__':
    main()
