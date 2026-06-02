#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build plane-aware semantic targets from cached ZeroPlane soft targets.

For each Cityscapes image:
  1. Load the cached 21-channel ZeroPlane soft target from soft_targets/
  2. Load the Cityscapes ground-truth semantic label
  3. Derive a plane mask / plane-slot assignment from ZeroPlane
  4. Assign each detected plane slot a semantic class based on the GT pixels
     that fall inside that plane region
  5. Save both machine-readable arrays and a debug visualization

This is intended as a first inspection/debug step before wiring the targets
into training.

Outputs per image:
  out_dir/<split>/<city>/<name>.npz
      plane_semantic_soft   (19, H, W) float16
      plane_semantic_hard   (H, W) uint8 trainId map, 255 = ignore
      plane_instance_id     (H, W) uint8, 0..19 plane slots, 20 non-plane
      plane_binary          (H, W) uint8, 1=plane, 0=non-plane
      plane_prob            (H, W) float16
      plane_class_probs     (20, 19) float16
      plane_class_majority  (20,) uint8

  debug_vis_dir/<split>/<city>/<name>_debug.png
      [RGB] | [GT semantic] | [plane instances] | [plane mask] | [plane semantic]
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

PLANE_S0 = 0.5
PLANE_S1 = 0.7310585786

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
except Exception:
    FONT = ImageFont.load_default()


def _resolve_repo_path(path, repo_root):
    if not path or osp.isabs(path):
        return path
    return osp.normpath(osp.join(repo_root, path))


def parse_args():
    parser = argparse.ArgumentParser(description='Build plane-aware semantic targets from cached ZeroPlane soft targets')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Cityscapes root containing leftImg8bit/ and gtFine/')
    parser.add_argument('--soft_targets_dir', type=str, default='./soft_targets',
                        help='Directory created by scripts/precompute_soft_targets.py')
    parser.add_argument('--out_dir', type=str, default='./plane_semantic_targets',
                        help='Output directory for .npz files')
    parser.add_argument('--debug_vis_dir', type=str, default='./plane_semantic_debug',
                        help='Output directory for debug .png files')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--assignment_mode', type=str, default='majority', choices=['majority', 'distribution'],
                        help='How to assign a semantic target to each plane slot')
    parser.add_argument('--nonplane_mode', type=str, default='ignore', choices=['ignore', 'gt'],
                        help='How to fill non-plane pixels in the saved semantic target')
    parser.add_argument('--plane_threshold', type=float, default=0.50,
                        help='Minimum recovered plane probability for a pixel to be treated as plane')
    parser.add_argument('--resume', action='store_true',
                        help='Skip images whose .npz already exists')
    parser.add_argument('--max_images', type=int, default=0,
                        help='Stop early after this many images (0 = all)')
    return parser.parse_args()


def _load_label_map(repo_root):
    with open(osp.join(repo_root, 'cityscapes_info.json'), 'r') as f:
        labels_info = json.load(f)
    return {el['id']: el['trainId'] for el in labels_info}


def _convert_to_trainid(label_arr, lb_map):
    label_arr = label_arr.copy()
    for src_id, train_id in lb_map.items():
        label_arr[label_arr == src_id] = train_id
    return label_arr


def _resize_label_nearest(label_arr, out_h, out_w):
    return np.array(
        Image.fromarray(label_arr.astype(np.uint8)).resize((out_w, out_h), Image.NEAREST),
        dtype=np.uint8,
    )


def _resize_rgb(img_rgb, out_h, out_w):
    return np.array(Image.fromarray(img_rgb).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


def uint82bin(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        idx = i
        for j in range(7):
            str_id = uint82bin(idx)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            idx = idx >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


PLANE_COLORS = labelcolormap(256)


def _render_trainid(trainid_map, black_ignore=True):
    h, w = trainid_map.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (trainid_map >= 0) & (trainid_map < len(CITYSCAPES_PALETTE))
    if valid.any():
        canvas[valid] = CITYSCAPES_PALETTE[trainid_map[valid]]
    if not black_ignore:
        canvas[~valid] = np.array([255, 255, 255], dtype=np.uint8)
    return canvas


def _render_plane_instances(plane_instance_id, plane_binary):
    h, w = plane_instance_id.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if plane_binary.any():
        canvas[plane_binary] = PLANE_COLORS[plane_instance_id[plane_binary]]
    return canvas


def _render_plane_binary(plane_binary):
    mask = (plane_binary.astype(np.uint8) * 255)
    return np.stack([mask, mask, mask], axis=2)


def _title_bar(img_np, text, bar_h=22):
    bar = np.full((bar_h, img_np.shape[1], 3), 30, dtype=np.uint8)
    pil = Image.fromarray(bar)
    ImageDraw.Draw(pil).text((4, 4), text, fill=(220, 220, 220), font=FONT)
    return np.vstack([np.array(pil), img_np])


def _recover_plane_probability(soft_target):
    ch20 = soft_target[20]
    nonplane_prob = ((ch20 - PLANE_S0) / (PLANE_S1 - PLANE_S0)).clip(0.0, 1.0)
    plane_prob = 1.0 - nonplane_prob
    return plane_prob.astype(np.float32)


def _assign_plane_semantics(soft_target, trainid_label, assignment_mode='majority',
                            nonplane_mode='ignore', plane_threshold=0.5):
    if soft_target.shape[0] != 21:
        raise ValueError('Expected soft target with 21 channels, got {}'.format(soft_target.shape))

    h, w = soft_target.shape[1], soft_target.shape[2]
    argmax_map = soft_target.argmax(axis=0).astype(np.uint8)
    plane_prob = _recover_plane_probability(soft_target)
    plane_binary = ((argmax_map < 20) & (plane_prob >= plane_threshold))

    semantic_soft = np.zeros((19, h, w), dtype=np.float32)
    semantic_hard = np.full((h, w), 255, dtype=np.uint8)
    plane_class_probs = np.zeros((20, 19), dtype=np.float32)
    plane_class_majority = np.full((20,), 255, dtype=np.uint8)

    for plane_idx in range(20):
        plane_mask = plane_binary & (argmax_map == plane_idx)
        if not plane_mask.any():
            continue

        labels = trainid_label[plane_mask]
        labels = labels[(labels >= 0) & (labels < 19)]
        if labels.size == 0:
            continue

        hist = np.bincount(labels, minlength=19).astype(np.float32)
        probs = hist / hist.sum().clip(min=1.0)
        maj = int(hist.argmax())

        plane_class_probs[plane_idx] = probs
        plane_class_majority[plane_idx] = maj
        semantic_hard[plane_mask] = maj

        if assignment_mode == 'distribution':
            semantic_soft[:, plane_mask] = probs[:, None]
        else:
            semantic_soft[maj, plane_mask] = 1.0

    if nonplane_mode == 'gt':
        nonplane_mask = ~plane_binary
        valid_nonplane = nonplane_mask & (trainid_label >= 0) & (trainid_label < 19)
        semantic_hard[valid_nonplane] = trainid_label[valid_nonplane]
        for cls_idx in range(19):
            cls_mask = valid_nonplane & (trainid_label == cls_idx)
            if cls_mask.any():
                semantic_soft[cls_idx, cls_mask] = 1.0

    return {
        'plane_semantic_soft': semantic_soft,
        'plane_semantic_hard': semantic_hard,
        'plane_instance_id': argmax_map,
        'plane_binary': plane_binary.astype(np.uint8),
        'plane_prob': plane_prob,
        'plane_class_probs': plane_class_probs,
        'plane_class_majority': plane_class_majority,
    }


def _make_debug_mosaic(rgb_path, trainid_label, outputs):
    rgb = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
    h, w = trainid_label.shape
    rgb = _resize_rgb(rgb, h, w)

    gt_vis = _render_trainid(trainid_label, black_ignore=True)
    plane_vis = _render_plane_instances(outputs['plane_instance_id'], outputs['plane_binary'].astype(bool))
    binary_vis = _render_plane_binary(outputs['plane_binary'])
    plane_sem_vis = _render_trainid(outputs['plane_semantic_hard'], black_ignore=True)

    return np.hstack([
        _title_bar(rgb, 'RGB'),
        _title_bar(gt_vis, 'GT semantic'),
        _title_bar(plane_vis, 'Plane instances'),
        _title_bar(binary_vis, 'Plane mask'),
        _title_bar(plane_sem_vis, 'Plane semantic assignment'),
    ])


def main():
    args = parse_args()

    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    args.data_root = _resolve_repo_path(args.data_root, repo_root)
    args.soft_targets_dir = _resolve_repo_path(args.soft_targets_dir, repo_root)
    args.out_dir = _resolve_repo_path(args.out_dir, repo_root)
    args.debug_vis_dir = _resolve_repo_path(args.debug_vis_dir, repo_root)

    lb_map = _load_label_map(repo_root)

    soft_root = osp.join(args.soft_targets_dir, args.split)
    gt_root = osp.join(args.data_root, 'gtFine', args.split)
    rgb_root = osp.join(args.data_root, 'leftImg8bit', args.split)

    if not osp.isdir(soft_root):
        raise FileNotFoundError('soft target split dir not found: {}'.format(soft_root))
    if not osp.isdir(gt_root):
        raise FileNotFoundError('gt split dir not found: {}'.format(gt_root))

    cities = sorted([d for d in os.listdir(soft_root) if osp.isdir(osp.join(soft_root, d))])
    done = skipped = failed = 0

    for city in cities:
        soft_city_dir = osp.join(soft_root, city)
        out_city_dir = osp.join(args.out_dir, args.split, city)
        vis_city_dir = osp.join(args.debug_vis_dir, args.split, city)
        os.makedirs(out_city_dir, exist_ok=True)
        os.makedirs(vis_city_dir, exist_ok=True)

        for fname in sorted(os.listdir(soft_city_dir)):
            if not fname.endswith('.npy'):
                continue

            stem = fname[:-4]
            soft_path = osp.join(soft_city_dir, fname)
            gt_path = osp.join(gt_root, city, stem + '_gtFine_labelIds.png')
            rgb_path = osp.join(rgb_root, city, stem + '_leftImg8bit.png')
            out_path = osp.join(out_city_dir, stem + '.npz')
            vis_path = osp.join(vis_city_dir, stem + '_debug.png')

            if args.resume and osp.isfile(out_path) and osp.isfile(vis_path):
                skipped += 1
                continue

            if not osp.isfile(gt_path):
                print('[plane-sem] WARNING missing GT:', gt_path, flush=True)
                failed += 1
                continue

            if not osp.isfile(rgb_path):
                print('[plane-sem] WARNING missing RGB:', rgb_path, flush=True)
                failed += 1
                continue

            try:
                soft_target = np.load(soft_path).astype(np.float32)
                gt_label_ids = np.array(Image.open(gt_path), dtype=np.int64)
                gt_trainid = _convert_to_trainid(gt_label_ids, lb_map).astype(np.uint8)

                out_h, out_w = soft_target.shape[1], soft_target.shape[2]
                if gt_trainid.shape != (out_h, out_w):
                    gt_trainid = _resize_label_nearest(gt_trainid, out_h, out_w)

                outputs = _assign_plane_semantics(
                    soft_target=soft_target,
                    trainid_label=gt_trainid,
                    assignment_mode=args.assignment_mode,
                    nonplane_mode=args.nonplane_mode,
                    plane_threshold=args.plane_threshold,
                )

                np.savez_compressed(
                    out_path,
                    plane_semantic_soft=outputs['plane_semantic_soft'].astype(np.float16),
                    plane_semantic_hard=outputs['plane_semantic_hard'].astype(np.uint8),
                    plane_instance_id=outputs['plane_instance_id'].astype(np.uint8),
                    plane_binary=outputs['plane_binary'].astype(np.uint8),
                    plane_prob=outputs['plane_prob'].astype(np.float16),
                    plane_class_probs=outputs['plane_class_probs'].astype(np.float16),
                    plane_class_majority=outputs['plane_class_majority'].astype(np.uint8),
                )

                mosaic = _make_debug_mosaic(rgb_path, gt_trainid, outputs)
                Image.fromarray(mosaic).save(vis_path)

                done += 1
                if done % 50 == 0:
                    print('[plane-sem] done={} skipped={} failed={}'.format(done, skipped, failed), flush=True)

                if args.max_images > 0 and done >= args.max_images:
                    print('[plane-sem] max_images={} reached'.format(args.max_images), flush=True)
                    print('[plane-sem] out_dir={}'.format(osp.join(args.out_dir, args.split)), flush=True)
                    print('[plane-sem] vis_dir={}'.format(osp.join(args.debug_vis_dir, args.split)), flush=True)
                    return

            except Exception as exc:
                print('[plane-sem] WARNING failed on {}: {}'.format(soft_path, exc), flush=True)
                failed += 1

    print('[plane-sem] Finished done={} skipped={} failed={}'.format(done, skipped, failed), flush=True)
    print('[plane-sem] out_dir={}'.format(osp.join(args.out_dir, args.split)), flush=True)
    print('[plane-sem] vis_dir={}'.format(osp.join(args.debug_vis_dir, args.split)), flush=True)


if __name__ == '__main__':
    main()
