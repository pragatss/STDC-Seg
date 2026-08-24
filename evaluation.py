#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Boundary-region IoU evaluator for STDC-Seg  (Arm H / Arm I edition).

Scores per-class IoU restricted to pixels within radius r of a ground-truth
class boundary (r=1 and r=3), the thin-class subset, and full-image mIoU as a
regression guard. Prints a comparison table with deltas vs. the first entry.

The stock evaluation.py reports only full-image mIoU, which dilutes boundary
effects -- that is the metric that misled every earlier arm.

USAGE
    python boundary_eval.py
Edit the RUNS list at the bottom. Arm I (loss-only) needs use_brh=False,
since the architecture is unchanged. Arm H needs use_brh=True.

IMPORTANT: use_brh here MUST match what the checkpoint was TRAINED with, or
you get either a shape crash or a silently wrong number.
"""
from models.model_stages import BiSeNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'tlight',
               'tsign', 'veg', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
               'bus', 'train', 'moto', 'bicycle']
THIN = [5, 6, 7, 12, 17, 18]   # pole, tlight, tsign, rider, moto, bicycle


class MscEvalV0(object):
    # Periodic whole-image mIoU used by train.py for checkpoint selection
    # (model_maxmIOU50.pth / model_maxmIOU75.pth). Unrelated to the boundary-region
    # comparison tooling below -- kept here only because train.py imports it from
    # this module.
    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
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
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
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


def gt_boundary(label, ignore=255):
    """label: [N,H,W] long -> [N,H,W] float in {0,1}; 1 where a valid class
    boundary exists. No wrap-around artifacts, ignore pixels never make edges."""
    N, H, W = label.shape
    bnd = torch.zeros((N, H, W), device=label.device)
    d = label[:, :, 1:] != label[:, :, :-1]
    v = (label[:, :, 1:] != ignore) & (label[:, :, :-1] != ignore)
    e = (d & v).float()
    bnd[:, :, 1:] = torch.maximum(bnd[:, :, 1:], e)
    bnd[:, :, :-1] = torch.maximum(bnd[:, :, :-1], e)
    d = label[:, 1:, :] != label[:, :-1, :]
    v = (label[:, 1:, :] != ignore) & (label[:, :-1, :] != ignore)
    e = (d & v).float()
    bnd[:, 1:, :] = torch.maximum(bnd[:, 1:, :], e)
    bnd[:, :-1, :] = torch.maximum(bnd[:, :-1, :], e)
    return bnd


def dilate(bnd, r):
    """Dilate a {0,1} map by radius r via max-pool -> bool [N,H,W]."""
    k = 2 * r + 1
    return (F.max_pool2d(bnd.unsqueeze(1), kernel_size=k, stride=1,
                         padding=r).squeeze(1) > 0.5)


def load_net(respth, backbone, use_boundary_8, use_brh, brh_mid, n_classes):
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
                  use_boundary_2=False, use_boundary_4=False,
                  use_boundary_8=use_boundary_8, use_boundary_16=False,
                  use_conv_last=False,
                  use_brh=use_brh, brh_mid=brh_mid)
    sd = torch.load(respth, map_location='cpu')
    sd = sd.get('state_dict', sd)
    # defensive: strip a DDP 'module.' prefix if one is ever present
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print("  [warn] missing keys (%d): %s" % (len(missing), missing[:4]))
    if unexpected:
        print("  [warn] unexpected keys (%d): %s" % (len(unexpected), unexpected[:4]))
    if not missing and not unexpected:
        print("  checkpoint loaded cleanly (all keys matched)")
    # report res_scale so you can see whether the module actually engaged
    for k, v in sd.items():
        if 'res_scale' in k:
            print("  %s = %+.4f" % (k, float(v.flatten()[0])))
    return net


@torch.no_grad()
def evaluate_boundary(respth, dspth='./data', backbone='STDCNet1446', scale=0.75,
                      use_boundary_8=True, use_brh=False, brh_mid=64,
                      radii=(1, 3), n_classes=19, ignore=255, batchsize=5,
                      n_workers=2, verbose=True):
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval, batch_size=batchsize, shuffle=False,
                    num_workers=n_workers, drop_last=False)

    net = load_net(respth, backbone, use_boundary_8, use_brh, brh_mid, n_classes)
    net.cuda().eval()

    I = {r: torch.zeros(n_classes).cuda() for r in radii}
    U = {r: torch.zeros(n_classes).cuda() for r in radii}
    If = torch.zeros(n_classes).cuda()
    Uf = torch.zeros(n_classes).cuda()

    for imgs, label in tqdm(dl, disable=not verbose):
        label = label.squeeze(1).cuda()
        size = label.shape[-2:]
        imgs = imgs.cuda()
        N, C, H, W = imgs.size()
        im = F.interpolate(imgs, [int(H * scale), int(W * scale)],
                           mode='bilinear', align_corners=True)
        logits = net(im)[0]
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
        pred = torch.argmax(logits, dim=1)

        valid = label != ignore
        bnd = gt_boundary(label, ignore)
        for r in radii:
            band = dilate(bnd, r) & valid
            for c in range(n_classes):
                pc = (pred == c) & band
                gc = (label == c) & band
                I[r][c] += (pc & gc).sum()
                U[r][c] += (pc | gc).sum()
        for c in range(n_classes):
            pc = (pred == c) & valid
            gc = (label == c) & valid
            If[c] += (pc & gc).sum()
            Uf[c] += (pc | gc).sum()

    def miou(i, u):
        iou = i / (u + 1e-6)
        return iou, iou[u > 0].mean().item()

    res = {}
    fi, res['full'] = miou(If, Uf)
    for r in radii:
        iou, m = miou(I[r], U[r])
        res['bnd_r%d' % r] = m
        res['thin_r%d' % r] = iou[THIN].mean().item()
        res['per_class_r%d' % r] = {CLASS_NAMES[c]: iou[c].item() for c in range(n_classes)}
    return res


def report(runs, **kw):
    """runs: list of (label, checkpoint_path, dict_of_overrides)"""
    out = []
    for label, path, over in runs:
        print("\n=== %s ===\n%s" % (label, path))
        cfg = dict(kw); cfg.update(over)
        out.append((label, evaluate_boundary(path, **cfg)))

    keys = ['full', 'bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3']
    print("\n" + "=" * 74)
    print("%-22s %9s %9s %9s %9s %9s" % ("run", "full", "bnd r=1", "bnd r=3", "thin r=1", "thin r=3"))
    print("-" * 74)
    base = out[0][1]
    for label, r in out:
        print("%-22s %9.4f %9.4f %9.4f %9.4f %9.4f" % tuple([label] + [r[k] for k in keys]))
    print("-" * 74)
    for label, r in out[1:]:
        print("%-22s %+9.4f %+9.4f %+9.4f %+9.4f %+9.4f" %
              tuple(["  d vs " + out[0][0][:12]] + [r[k] - base[k] for k in keys]))
    print("=" * 74)
    print("Primary metric = boundary r=1 / r=3 and thin-class. full mIoU is a guard only.")

    # per-class thin breakdown at r=1
    print("\nthin-class IoU @ r=1")
    print("%-22s %s" % ("run", "  ".join("%8s" % CLASS_NAMES[c] for c in THIN)))
    for label, r in out:
        pc = r['per_class_r1']
        print("%-22s %s" % (label, "  ".join("%8.3f" % pc[CLASS_NAMES[c]] for c in THIN)))
    return out


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Edit paths/labels. First entry is the reference for the deltas.
    # Arm I is loss-only  -> use_brh=False
    # Arm H changes arch  -> use_brh=True
    # ---------------------------------------------------------------
    RUNS = [
        ("A baseline",
         "./checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth",
         dict(use_brh=False)),

        ("A baseline v2",
         "./checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth",
         dict(use_brh=False)),

        ("I0 lambda=1 control",
         "./checkpoints/train_STDC2-Seg-I0/pths/model_maxmIOU75.pth",
         dict(use_brh=False)),

        ("I1 lambda=3",
         "./checkpoints/train_STDC2-Seg-I1/pths/model_maxmIOU75.pth",
         dict(use_brh=False)),

        ("H1 stride-4 refine",
         "./checkpoints/train_STDC2-Seg-H1/pths/model_maxmIOU75.pth",
         dict(use_brh=True)),

        ("HI1 brh + lambda=3",
         "./checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU75.pth",
         dict(use_brh=True)),
    ]

    report(RUNS,
           dspth='./data',
           backbone='STDCNet1446',
           scale=0.75,
           use_boundary_8=True,
           brh_mid=64)