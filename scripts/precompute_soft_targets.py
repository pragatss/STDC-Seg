#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Offline pre-computation of ZeroPlane soft targets for Cityscapes training images.

Run this ONCE before training. It iterates over every Cityscapes training image,
feeds it through ZeroPlane at half resolution (512 × 1024) and saves the resulting
21-channel sem_seg map downsampled to (128 × 256) as a float16 .npy file.

Total disk usage: ~4 GB for all 2975 Cityscapes training images.

Usage:
    python scripts/precompute_soft_targets.py \
        --data_root ./data \
        --out_dir  ./soft_targets \
        --config   ZeroPlane/configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
        --opts     MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
        --resume

The output mirrors the Cityscapes leftImg8bit directory layout:
    soft_targets/train/<city>/<name>.npy

Each .npy file contains a float16 numpy array of shape (21, 128, 256).
"""

import argparse
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Soft-target resolution.  1/8 of native Cityscapes (1024 × 2048) = (128, 256).
# This is also 1/4 of the inference crop size (512 × 1024) used below.
# ---------------------------------------------------------------------------
ST_H = 128
ST_W = 256

# Resolution fed to ZeroPlane.  Using half native height/width keeps VRAM low
# while preserving the 2:1 aspect ratio that per-pixel predictions need.
INFER_H = 512
INFER_W = 1024


def parse_args():
    p = argparse.ArgumentParser(description='Pre-compute ZeroPlane soft targets')
    p.add_argument('--data_root', type=str, default='./data',
                   help='Cityscapes data root (contains leftImg8bit/ and gtFine/)')
    p.add_argument('--out_dir', type=str, default='./soft_targets',
                   help='Output directory for .npy files')
    p.add_argument('--config', type=str,
                   default='ZeroPlane/configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml',
                   help='ZeroPlane config yaml')
    p.add_argument('--opts', nargs='*',
                   default=['MODEL.WEIGHTS', './checkpoints/dust3r_encoder_released.pth'],
                   help='Detectron2 KEY VALUE overrides (same as training)')
    p.add_argument('--split', type=str, default='train',
                   help='Dataset split to process (train / val / test)')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device for ZeroPlane inference')
    p.add_argument('--resume', action='store_true',
                   help='Skip images whose .npy already exists')
    return p.parse_args()


def _build_zeroplane_predictor(config_path, opts, device):
    """Load the ZeroPlane DefaultPredictor (same logic as train.py)."""
    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    zeroplane_root = osp.join(repo_root, 'ZeroPlane')
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from ZeroPlane.ZeroPlane import add_ZeroPlane_config
    from ZeroPlane.demo.predictor import DefaultPredictor

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ZeroPlane_config(cfg)
    cfg.merge_from_file(config_path)
    if opts:
        cfg.merge_from_list(opts)
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    return DefaultPredictor(cfg)


def _load_anchors(zeroplane_root, device):
    normals = np.load(osp.join(zeroplane_root, 'cluster_anchor', 'new_mixed_normal_anchors_7.npy'))
    offsets = np.load(osp.join(zeroplane_root, 'cluster_anchor', 'new_mixed_offset_anchors_20.npy'))
    return (
        torch.tensor(normals).to(device),
        torch.tensor(offsets).to(device),
    )


def _get_coordinate_map(h, w, oh, ow, device):
    K = np.asarray([[518.86, 0, 325.58],
                    [0, 519.47, 253.74],
                    [0, 0, 1]], dtype=np.float32)
    K_inv = torch.FloatTensor(np.linalg.inv(K)).to(device)
    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * ow
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * oh
    xx = x.repeat(h, 1).to(device)
    yy = y.repeat(1, w).to(device)
    xy1 = torch.stack((xx, yy, torch.ones(h, w, dtype=torch.float32).to(device)))
    return torch.matmul(K_inv, xy1.view(3, -1))


def run_zeroplane(predictor, img_np, anchor_normals, anchor_offsets, device):
    """
    Run ZeroPlane on a single uint8 HWC RGB numpy image.
    Returns sem_seg tensor (21, H, W) on CPU, or None on failure.
    """
    h, w = img_np.shape[:2]
    k_inv_dot_xy1 = _get_coordinate_map(h, w, h, w, device)

    # DefaultPredictor expects BGR; its config uses FORMAT=RGB so it
    # reverses back to RGB internally — pass RGB as-is then let it flip.
    # (Looking at predictor.py: if format==RGB it does [:, :, ::-1]. So
    #  passing BGR gives correct behaviour — consistent with original demo.)
    img_bgr = img_np[:, :, ::-1].copy()

    anchors = {'anchor_normals': anchor_normals, 'anchor_offsets': anchor_offsets}
    with torch.no_grad():
        pred = predictor(img_bgr, anchors, k_inv_dot_xy1)

    if not isinstance(pred, dict):
        return None
    sem_seg = pred.get('sem_seg', None)
    if sem_seg is None or not torch.is_tensor(sem_seg):
        return None
    return torch.clamp(sem_seg.float(), 0.0, 1.0).cpu()


def main():
    args = parse_args()

    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    zeroplane_root = osp.join(repo_root, 'ZeroPlane')
    device = args.device

    print('[precompute] Loading ZeroPlane predictor...', flush=True)
    predictor = _build_zeroplane_predictor(args.config, args.opts, device)
    anchor_normals, anchor_offsets = _load_anchors(zeroplane_root, device)
    print('[precompute] ZeroPlane ready.', flush=True)

    img_root = osp.join(args.data_root, 'leftImg8bit', args.split)
    cities = sorted(os.listdir(img_root))

    total = sum(len(os.listdir(osp.join(img_root, c))) for c in cities)
    done = skipped = failed = 0

    for city in cities:
        city_img_dir = osp.join(img_root, city)
        city_out_dir = osp.join(args.out_dir, args.split, city)
        os.makedirs(city_out_dir, exist_ok=True)

        for fname in sorted(os.listdir(city_img_dir)):
            if not fname.endswith('_leftImg8bit.png'):
                continue
            name = fname.replace('_leftImg8bit.png', '')
            out_path = osp.join(city_out_dir, name + '.npy')

            if args.resume and osp.isfile(out_path):
                skipped += 1
                continue

            img_path = osp.join(city_img_dir, fname)
            img_pil = Image.open(img_path).convert('RGB')

            # Resize to INFER_H × INFER_W (512 × 1024) to reduce VRAM usage.
            # The 2:1 aspect ratio of Cityscapes is preserved.
            img_pil_resized = img_pil.resize((INFER_W, INFER_H), Image.BILINEAR)
            img_np = np.array(img_pil_resized, dtype=np.uint8)

            sem_seg = run_zeroplane(predictor, img_np, anchor_normals, anchor_offsets, device)

            if sem_seg is None:
                print('[precompute] WARNING: ZeroPlane returned None for {}'.format(fname), flush=True)
                failed += 1
                continue

            # Downsample from (21, INFER_H, INFER_W) to (21, ST_H, ST_W).
            sem_seg_small = torch.nn.functional.interpolate(
                sem_seg.unsqueeze(0),
                size=(ST_H, ST_W),
                mode='bilinear',
                align_corners=True,
            ).squeeze(0)

            # Save as float16 to halve storage (~1.4 MB per file, ~4 GB total).
            np.save(out_path, sem_seg_small.numpy().astype(np.float16))

            done += 1
            if done % 50 == 0:
                print('[precompute] {}/{} done  ({} skipped, {} failed)'.format(
                      done + skipped, total, skipped, failed), flush=True)

            # Free VRAM between images.
            torch.cuda.empty_cache()

    print('[precompute] Finished. done={} skipped={} failed={} total={}'.format(
          done, skipped, failed, total), flush=True)
    print('[precompute] Soft targets saved to: {}'.format(osp.join(args.out_dir, args.split)), flush=True)


if __name__ == '__main__':
    main()
