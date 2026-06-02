#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign a semantic class to each ZeroPlane plane slot using segmentation labels.

This script is meant for the exact workflow:
  - planar soft targets from scripts/generated_softtargets/
  - semantic labels from scripts/seg/

For each image:
  1. load the 21-channel ZeroPlane soft target
  2. recover the plane mask and plane-slot assignment (0..19)
  3. load the semantic trainId label image
  4. for each plane slot, collect the semantic labels inside that plane
  5. assign the plane a semantic class via majority vote or class distribution
  6. save per-plane metadata plus a per-pixel plane-semantic target

Outputs:
  out_dir/<stem>_plane_labels.json
  out_dir/<stem>_plane_semantic_hard.png
  out_dir/<stem>_plane_semantic_soft.npz
  out_dir/<stem>_plane_semantic_overlay.png
    out_dir/<stem>_plane_instance_map.png
    out_dir/<stem>_plane_instance_overlay.png
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CITYSCAPES_PALETTE = np.array([
    [128,  64, 128],   #  0 road
    [244,  35, 232],   #  1 sidewalk
    [ 70,  70,  70],   #  2 building
    [102, 102, 156],   #  3 wall
    [190, 153, 153],   #  4 fence
    [153, 153, 153],   #  5 pole
    [250, 170,  30],   #  6 traffic light
    [220, 220,   0],   #  7 traffic sign
    [107, 142,  35],   #  8 vegetation
    [152, 251, 152],   #  9 terrain
    [ 70, 130, 180],   # 10 sky
    [220,  20,  60],   # 11 person
    [255,   0,   0],   # 12 rider
    [  0,   0, 142],   # 13 car
    [  0,   0,  70],   # 14 truck
    [  0,  60, 100],   # 15 bus
    [  0,  80, 100],   # 16 train
    [  0,   0, 230],   # 17 motorcycle
    [119,  11,  32],   # 18 bicycle
], dtype=np.uint8)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle',
]

PLANE_S0 = 0.5
PLANE_S1 = 0.7310585786

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except Exception:
    FONT = ImageFont.load_default()


def parse_args():
    parser = argparse.ArgumentParser(description='Assign semantic classes to ZeroPlane plane slots')
    parser.add_argument('--soft_dir', type=str, default='./scripts/generated_softtargets',
                        help='Directory containing *.npy ZeroPlane soft targets')
    parser.add_argument('--seg_dir', type=str, default='./scripts/seg',
                        help='Directory containing *_gtFine_labelTrainIds.png')
    parser.add_argument('--image_dir', type=str, default='./scripts/images',
                        help='Optional RGB image directory for overlay rendering')
    parser.add_argument('--out_dir', type=str, default='./scripts/plane_semantic_assignments',
                        help='Output directory')
    parser.add_argument('--stem', type=str, default='',
                        help='Optional single image stem, e.g. cologne_000000_000019_leftImg8bit')
    parser.add_argument('--assignment_mode', type=str, default='majority', choices=['majority', 'distribution'],
                        help='Plane semantic assignment mode')
    parser.add_argument('--plane_threshold', type=float, default=0.50,
                        help='Minimum recovered plane probability to keep a pixel as plane')
    return parser.parse_args()


def _resolve_repo_path(path, repo_root):
    if not path or osp.isabs(path):
        return path
    return osp.normpath(osp.join(repo_root, path))


def _recover_plane_probability(soft_target):
    ch20 = soft_target[20]
    nonplane_prob = ((ch20 - PLANE_S0) / (PLANE_S1 - PLANE_S0)).clip(0.0, 1.0)
    return (1.0 - nonplane_prob).astype(np.float32)


def _resize_rgb(img_np, out_h, out_w):
    return np.array(Image.fromarray(img_np).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


def _resize_label(label_np, out_h, out_w):
    return np.array(Image.fromarray(label_np.astype(np.uint8)).resize((out_w, out_h), Image.NEAREST), dtype=np.uint8)


def _uint82bin(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def _labelcolormap(num_colors):
    cmap = np.zeros((num_colors, 3), dtype=np.uint8)
    for i in range(num_colors):
        r = 0
        g = 0
        b = 0
        idx = i
        for j in range(7):
            str_id = _uint82bin(idx)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            idx = idx >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


INSTANCE_COLORS = _labelcolormap(256)


def _render_trainid(trainid_map):
    canvas = np.zeros((trainid_map.shape[0], trainid_map.shape[1], 3), dtype=np.uint8)
    valid = (trainid_map >= 0) & (trainid_map < len(CITYSCAPES_PALETTE))
    if valid.any():
        canvas[valid] = CITYSCAPES_PALETTE[trainid_map[valid]]
    return canvas


def _build_overlay(rgb, semantic_hard):
    semantic_rgb = _render_trainid(semantic_hard)
    overlay = rgb.copy()
    valid = semantic_hard != 255
    overlay[valid] = (0.35 * rgb[valid] + 0.65 * semantic_rgb[valid]).astype(np.uint8)
    overlay[~valid] = (0.15 * rgb[~valid]).astype(np.uint8)
    return overlay


def _render_plane_instances(instance_id_map, plane_mask):
    canvas = np.zeros((instance_id_map.shape[0], instance_id_map.shape[1], 3), dtype=np.uint8)
    valid_ids = plane_mask & (instance_id_map < len(INSTANCE_COLORS))
    if valid_ids.any():
        canvas[valid_ids] = INSTANCE_COLORS[instance_id_map[valid_ids] + 1]
    return canvas


def _mask_centroid(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def _build_instance_overlay(rgb, instance_id_map, plane_mask, plane_metadata):
    instance_rgb = _render_plane_instances(instance_id_map, plane_mask)
    overlay = rgb.copy()
    overlay[plane_mask] = (0.35 * rgb[plane_mask] + 0.65 * instance_rgb[plane_mask]).astype(np.uint8)
    overlay[~plane_mask] = (0.15 * rgb[~plane_mask]).astype(np.uint8)

    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    for plane in plane_metadata:
        if plane['majority_class'] is None:
            continue
        plane_idx = plane['plane_idx']
        centroid = _mask_centroid(plane_mask & (instance_id_map == plane_idx))
        if centroid is None:
            continue
        x, y = centroid
        label = 'P{} {}'.format(plane_idx, plane['majority_name'])
        draw.text((x + 1, y + 1), label, fill=(0, 0, 0), font=FONT)
        draw.text((x, y), label, fill=tuple(INSTANCE_COLORS[plane_idx + 1].tolist()), font=FONT)
    return np.array(pil)


def _process_one(soft_path, seg_path, rgb_path, out_dir, assignment_mode, plane_threshold):
    soft_target = np.load(soft_path).astype(np.float32)
    if soft_target.shape[0] != 21:
        raise ValueError('Expected 21 channels, got {}'.format(soft_target.shape))

    semantic_label = np.array(Image.open(seg_path), dtype=np.uint8)
    out_h, out_w = soft_target.shape[1], soft_target.shape[2]
    if semantic_label.shape != (out_h, out_w):
        semantic_label = _resize_label(semantic_label, out_h, out_w)

    argmax_map = soft_target.argmax(axis=0).astype(np.uint8)
    plane_prob = _recover_plane_probability(soft_target)
    plane_mask = (argmax_map < 20) & (plane_prob >= plane_threshold)

    plane_semantic_hard = np.full((out_h, out_w), 255, dtype=np.uint8)
    plane_semantic_soft = np.zeros((19, out_h, out_w), dtype=np.float32)
    plane_idx_to_semantic_class = np.full((20,), -1, dtype=np.int16)
    plane_metadata = []

    for plane_idx in range(20):
        mask = plane_mask & (argmax_map == plane_idx)
        if not mask.any():
            continue

        labels = semantic_label[mask]
        labels = labels[(labels >= 0) & (labels < 19)]
        if labels.size == 0:
            plane_metadata.append({
                'plane_idx': int(plane_idx),
                'pixel_count': int(mask.sum()),
                'majority_class': None,
                'majority_name': None,
                'majority_fraction': 0.0,
                'class_hist': {},
            })
            continue

        hist = np.bincount(labels, minlength=19).astype(np.int64)
        maj = int(hist.argmax())
        maj_frac = float(hist[maj]) / float(labels.size)

        if assignment_mode == 'distribution':
            probs = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
            plane_semantic_soft[:, mask] = probs[:, None]
        else:
            plane_semantic_soft[maj, mask] = 1.0

        plane_semantic_hard[mask] = maj
        plane_idx_to_semantic_class[plane_idx] = maj
        plane_metadata.append({
            'plane_idx': int(plane_idx),
            'pixel_count': int(mask.sum()),
            'majority_class': maj,
            'majority_name': CLASS_NAMES[maj],
            'majority_fraction': maj_frac,
            'render_color_rgb': INSTANCE_COLORS[plane_idx + 1].tolist(),
            'class_hist': {CLASS_NAMES[i]: int(hist[i]) for i in range(19) if hist[i] > 0},
        })

    stem = osp.basename(soft_path).replace('.npy', '')
    base_stem = stem.replace('_leftImg8bit', '')

    hard_png_path = osp.join(out_dir, stem + '_plane_semantic_hard.png')
    soft_npz_path = osp.join(out_dir, stem + '_plane_semantic_soft.npz')
    json_path = osp.join(out_dir, stem + '_plane_labels.json')
    overlay_path = osp.join(out_dir, stem + '_plane_semantic_overlay.png')
    instance_map_path = osp.join(out_dir, stem + '_plane_instance_map.png')
    instance_overlay_path = osp.join(out_dir, stem + '_plane_instance_overlay.png')

    Image.fromarray(_render_trainid(plane_semantic_hard)).save(hard_png_path)
    Image.fromarray(_render_plane_instances(argmax_map, plane_mask)).save(instance_map_path)
    np.savez_compressed(
        soft_npz_path,
        plane_semantic_soft=plane_semantic_soft.astype(np.float16),
        plane_semantic_hard=plane_semantic_hard.astype(np.uint8),
        plane_mask=plane_mask.astype(np.uint8),
        plane_instance_id=argmax_map.astype(np.uint8),
        plane_idx_to_semantic_class=plane_idx_to_semantic_class,
        plane_prob=plane_prob.astype(np.float16),
    )

    with open(json_path, 'w') as f:
        json.dump({
            'stem': stem,
            'assignment_mode': assignment_mode,
            'plane_threshold': plane_threshold,
            'planes': plane_metadata,
        }, f, indent=2)

    if rgb_path and osp.isfile(rgb_path):
        rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
        if rgb.shape[:2] != (out_h, out_w):
            rgb = _resize_rgb(rgb, out_h, out_w)
        Image.fromarray(_build_overlay(rgb, plane_semantic_hard)).save(overlay_path)
        Image.fromarray(_build_instance_overlay(rgb, argmax_map, plane_mask, plane_metadata)).save(instance_overlay_path)

    print('[assign-plane-class] saved:', hard_png_path)
    print('[assign-plane-class] saved:', instance_map_path)
    print('[assign-plane-class] saved:', soft_npz_path)
    print('[assign-plane-class] saved:', json_path)
    if rgb_path and osp.isfile(rgb_path):
        print('[assign-plane-class] saved:', overlay_path)
        print('[assign-plane-class] saved:', instance_overlay_path)

    top_planes = sorted(
        [p for p in plane_metadata if p['majority_class'] is not None],
        key=lambda x: x['pixel_count'],
        reverse=True,
    )[:8]
    print('[assign-plane-class] top planes:', top_planes)


def main():
    args = parse_args()
    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

    soft_dir = _resolve_repo_path(args.soft_dir, repo_root)
    seg_dir = _resolve_repo_path(args.seg_dir, repo_root)
    image_dir = _resolve_repo_path(args.image_dir, repo_root)
    out_dir = _resolve_repo_path(args.out_dir, repo_root)
    os.makedirs(out_dir, exist_ok=True)

    if args.stem:
        soft_files = [osp.join(soft_dir, args.stem + '.npy')]
    else:
        soft_files = sorted(
            osp.join(soft_dir, f)
            for f in os.listdir(soft_dir)
            if f.endswith('.npy')
        )

    for soft_path in soft_files:
        if not osp.isfile(soft_path):
            print('[assign-plane-class] missing soft target:', soft_path)
            continue

        stem = osp.basename(soft_path).replace('.npy', '')
        base_stem = stem.replace('_leftImg8bit', '')
        seg_path = osp.join(seg_dir, base_stem + '_gtFine_labelTrainIds.png')
        rgb_path = osp.join(image_dir, stem + '.png')

        if not osp.isfile(seg_path):
            print('[assign-plane-class] missing segmentation label:', seg_path)
            continue

        if not osp.isfile(rgb_path):
            rgb_path = None

        _process_one(
            soft_path=soft_path,
            seg_path=seg_path,
            rgb_path=rgb_path,
            out_dir=out_dir,
            assignment_mode=args.assignment_mode,
            plane_threshold=args.plane_threshold,
        )


if __name__ == '__main__':
    main()
