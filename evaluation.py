#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Boundary-region IoU evaluator for STDC-Seg.

Scores per-class IoU restricted to pixels within radius r of a ground-truth
class boundary (r=1 and r=3), plus full-image mIoU as a guard. This is the
metric the whole project treats as primary; the stock evaluation.py reports
only full-image mIoU, which dilutes boundary effects.

Usage: edit the calls at the bottom to point at your checkpoints, then
    python boundary_eval.py
Score the baseline and the GCN checkpoint the SAME way and compare.
"""
from models.model_stages import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

CLASS_NAMES = ['road','sidewalk','building','wall','fence','pole','tlight',
               'tsign','veg','terrain','sky','person','rider','car','truck',
               'bus','train','moto','bicycle']
THIN = [5, 6, 7, 12, 17, 18]   # pole, tlight, tsign, rider, moto, bicycle


def gt_boundary(label, ignore=255):
    """label: [N,H,W] long -> [N,H,W] float in {0,1}, 1 where a valid class
    boundary exists (no edge-wrap artifacts)."""
    N, H, W = label.shape
    bnd = torch.zeros((N, H, W), device=label.device)
    # horizontal neighbours
    diff = label[:, :, 1:] != label[:, :, :-1]
    valid = (label[:, :, 1:] != ignore) & (label[:, :, :-1] != ignore)
    e = (diff & valid).float()
    bnd[:, :, 1:] = torch.maximum(bnd[:, :, 1:], e)
    bnd[:, :, :-1] = torch.maximum(bnd[:, :, :-1], e)
    # vertical neighbours
    diff = label[:, 1:, :] != label[:, :-1, :]
    valid = (label[:, 1:, :] != ignore) & (label[:, :-1, :] != ignore)
    e = (diff & valid).float()
    bnd[:, 1:, :] = torch.maximum(bnd[:, 1:, :], e)
    bnd[:, :-1, :] = torch.maximum(bnd[:, :-1, :], e)
    return bnd


def dilate(bnd, r):
    """Dilate a {0,1} map by radius r via max-pool. bnd:[N,H,W] -> [N,H,W] bool."""
    k = 2 * r + 1
    d = F.max_pool2d(bnd.unsqueeze(1), kernel_size=k, stride=1, padding=r)
    return d.squeeze(1) > 0.5


@torch.no_grad()
def evaluate_boundary(respth, dspth='./data', backbone='STDCNet1446', scale=0.75,
                      use_boundary_8=True, use_ctx_gcn=False, gcn_gate='boundary',
                      radii=(1, 3), n_classes=19, ignore=255):
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval, batch_size=5, shuffle=False, num_workers=2, drop_last=False)

    net = BiSeNet(backbone=backbone, n_classes=n_classes,
                  use_boundary_2=False, use_boundary_4=False,
                  use_boundary_8=use_boundary_8, use_boundary_16=False,
                  use_conv_last=False, use_ctx_gcn=use_ctx_gcn, gcn_gate=gcn_gate)
    net.load_state_dict(torch.load(respth))
    net.cuda().eval()

    # per-radius intersection/union accumulators, plus full-image
    I = {r: torch.zeros(n_classes).cuda() for r in radii}
    U = {r: torch.zeros(n_classes).cuda() for r in radii}
    If = torch.zeros(n_classes).cuda(); Uf = torch.zeros(n_classes).cuda()

    for imgs, label in tqdm(dl):
        label = label.squeeze(1).cuda()               # [N,H,W]
        size = label.shape[-2:]
        imgs = imgs.cuda()
        N, C, H, W = imgs.size()
        new_hw = [int(H * scale), int(W * scale)]
        im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
        logits = net(im)[0]
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
        pred = torch.argmax(logits, dim=1)            # [N,H,W]

        valid = label != ignore
        bnd = gt_boundary(label, ignore)
        for r in radii:
            band = dilate(bnd, r) & valid
            for c in range(n_classes):
                pc = (pred == c) & band
                gc = (label == c) & band
                I[r][c] += (pc & gc).sum()
                U[r][c] += (pc | gc).sum()
        for c in range(n_classes):                    # full-image guard
            pc = (pred == c) & valid
            gc = (label == c) & valid
            If[c] += (pc & gc).sum()
            Uf[c] += (pc | gc).sum()

    def miou(I, U):
        iou = I / (U + 1e-6)
        present = U > 0
        return iou, iou[present].mean().item()

    print("\n==== %s ====" % respth)
    fi, fm = miou(If, Uf)
    print("full-image mIoU: %.4f" % fm)
    for r in radii:
        iou, m = miou(I[r], U[r])
        print("boundary r=%d mIoU: %.4f" % (r, m))
        thin = iou[THIN].mean().item()
        print("   thin-class r=%d mIoU: %.4f  (%s)" % (
            r, thin, ", ".join("%s %.3f" % (CLASS_NAMES[c], iou[c].item()) for c in THIN)))
    return fm


if __name__ == "__main__":
    # ---- edit these two paths, run both, compare ----
    evaluate_boundary('./checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth',
                      use_ctx_gcn=False)
    evaluate_boundary('./checkpoints/train_STDC2-Seg-GCN/pths/model_maxmIOU75.pth',
                      use_ctx_gcn=True, gcn_gate='boundary')