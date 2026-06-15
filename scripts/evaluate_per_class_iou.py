#!/usr/bin/env python3
"""Compute per-class IoU for a baseline checkpoint and a target checkpoint, then compare them."""

import argparse
import csv
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from cityscapes import CityScapes
from models.model_stages import BiSeNet

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]

DEFAULT_BASELINE_WEIGHTS = './checkpoints/train_STDC1-Seg/baseline_no_plane'
DEFAULT_OUT_DIR = './predictions/per_class_iou'


def build_model(weights_path, backbone, device, use_boundary_2, use_boundary_4, use_boundary_8, use_boundary_16):
    net = BiSeNet(
        backbone=backbone,
        n_classes=19,
        use_boundary_2=use_boundary_2,
        use_boundary_4=use_boundary_4,
        use_boundary_8=use_boundary_8,
        use_boundary_16=use_boundary_16,
    )
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned[new_key] = value
    net.load_state_dict(cleaned, strict=False)
    net.to(device)
    net.eval()
    return net


def eval_collate(batch):
    ims, lbs, sts = zip(*batch)
    return default_collate(list(ims)), default_collate(list(lbs)), None


def compute_metrics(hist):
    intersection = np.diag(hist)
    pred_total = hist.sum(axis=0)
    gt_total = hist.sum(axis=1)
    union = pred_total + gt_total - intersection
    ious = intersection / np.maximum(union, 1e-10)
    accs = intersection / np.maximum(gt_total, 1e-10)
    miou = float(np.mean(ious))
    return ious, accs, miou


def infer_run_name(weights_path):
    abs_path = osp.abspath(weights_path)
    parts = abs_path.split(os.sep)
    if 'pths' in parts:
        idx = parts.index('pths')
        if idx > 0:
            return parts[idx - 1]
    return osp.splitext(osp.basename(abs_path))[0]


def checkpoint_tag_for_scale(scale):
    return '75' if scale >= 0.625 else '50'


def resolve_checkpoint_path(weights_arg, scale, prefer_best=True):
    path = osp.abspath(weights_arg)
    tag = checkpoint_tag_for_scale(scale)
    best_name = f'model_maxmIOU{tag}.pth'

    if osp.isdir(path):
        candidates = [
            osp.join(path, 'pths', best_name),
            osp.join(path, best_name),
            osp.join(path, 'pths', 'model_final.pth'),
            osp.join(path, 'model_final.pth'),
        ]
        for candidate in candidates:
            if osp.isfile(candidate):
                return candidate
        raise FileNotFoundError(f'Could not find checkpoint under {path}')

    if not osp.isfile(path):
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')

    if prefer_best and osp.basename(path) == 'model_final.pth':
        best_candidate = osp.join(osp.dirname(path), best_name)
        if osp.isfile(best_candidate):
            return best_candidate

    return path


def default_csv_path(baseline_weights, target_weights, out_dir):
    baseline_name = infer_run_name(baseline_weights)
    target_name = infer_run_name(target_weights)
    file_name = '{}_vs_{}.csv'.format(target_name, baseline_name)
    return osp.join(out_dir, file_name)


def evaluate_checkpoint(weights_path, data_root, backbone, scale, batch_size, num_workers,
                        use_boundary_2, use_boundary_4, use_boundary_8, use_boundary_16, device):
    print('weights:', osp.abspath(weights_path))
    dsval = CityScapes(data_root, mode='val')
    dl = DataLoader(
        dsval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=eval_collate,
    )

    net = build_model(
        weights_path,
        backbone,
        device,
        use_boundary_2,
        use_boundary_4,
        use_boundary_8,
        use_boundary_16,
    )

    n_classes = 19
    ignore_label = 255
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)

    with torch.no_grad():
        for imgs, label, _ in tqdm(dl, desc='eval', leave=False):
            label = label.squeeze(1).numpy().astype(np.int64)
            imgs = imgs.to(device)
            _, _, h, w = imgs.shape
            sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
            imgs = F.interpolate(imgs, (sh, sw), mode='bilinear', align_corners=True)
            logits = net(imgs)[0]
            logits = F.interpolate(logits, size=label.shape[-2:], mode='bilinear', align_corners=True)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.int64)

            keep = label != ignore_label
            merged = preds[keep] * n_classes + label[keep]
            hist += np.bincount(merged, minlength=n_classes ** 2).reshape(n_classes, n_classes)

    ious, accs, miou = compute_metrics(hist)
    return {
        'weights': weights_path,
        'ious': ious,
        'accs': accs,
        'miou': miou,
    }


def print_single_result(title, result):
    print(f'\n{title}:')
    for idx, (name, iou, acc) in enumerate(zip(CLASS_NAMES, result['ious'], result['accs'])):
        print(f'  class {idx:02d} ({name}): IoU={iou:.4f}, Acc={acc:.4f}')
    print(f"  mIoU={result['miou']:.6f}")


def print_comparison(baseline_result, target_result):
    print('\nComparison vs baseline:')
    print('  {:<2} {:<14} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
        '#', 'class', 'base IoU', 'target IoU', 'Δ IoU', 'base Acc', 'target Acc', 'Δ Acc'
    ))
    for idx, name in enumerate(CLASS_NAMES):
        base_iou = baseline_result['ious'][idx]
        tgt_iou = target_result['ious'][idx]
        base_acc = baseline_result['accs'][idx]
        tgt_acc = target_result['accs'][idx]
        print('  {:02d} {:<14} {:>10.4f} {:>10.4f} {:>+10.4f} {:>10.4f} {:>10.4f} {:>+10.4f}'.format(
            idx, name[:14], base_iou, tgt_iou, tgt_iou - base_iou, base_acc, tgt_acc, tgt_acc - base_acc
        ))
    print('\n  baseline mIoU = {:.6f}'.format(baseline_result['miou']))
    print('  target   mIoU = {:.6f}'.format(target_result['miou']))
    print('  Δ mIoU        = {:+.6f}'.format(target_result['miou'] - baseline_result['miou']))


def write_comparison_csv(csv_path, baseline_result, target_result):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'class_id', 'class_name',
            'baseline_iou', 'target_iou', 'delta_iou',
            'baseline_acc', 'target_acc', 'delta_acc',
        ])
        for idx, name in enumerate(CLASS_NAMES):
            base_iou = baseline_result['ious'][idx]
            tgt_iou = target_result['ious'][idx]
            base_acc = baseline_result['accs'][idx]
            tgt_acc = target_result['accs'][idx]
            writer.writerow([
                idx,
                name,
                '{:.6f}'.format(base_iou),
                '{:.6f}'.format(tgt_iou),
                '{:+.6f}'.format(tgt_iou - base_iou),
                '{:.6f}'.format(base_acc),
                '{:.6f}'.format(tgt_acc),
                '{:+.6f}'.format(tgt_acc - base_acc),
            ])
        writer.writerow([])
        writer.writerow(['metric', 'baseline', 'target', 'delta'])
        writer.writerow([
            'mIoU',
            '{:.6f}'.format(baseline_result['miou']),
            '{:.6f}'.format(target_result['miou']),
            '{:+.6f}'.format(target_result['miou'] - baseline_result['miou']),
        ])


def main():
    p = argparse.ArgumentParser(description='Evaluate baseline and target checkpoints and print per-class IoU comparison.')
    p.add_argument('--baseline_weights', default=DEFAULT_BASELINE_WEIGHTS)
    p.add_argument('--target_weights', required=True)
    p.add_argument('--data_root', default='./data')
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--csv_path', default='')
    p.add_argument('--backbone', default='STDCNet813', choices=['STDCNet813', 'STDCNet1446'])
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--use_boundary_2', action='store_true')
    p.add_argument('--use_boundary_4', action='store_true')
    p.add_argument('--use_boundary_8', action='store_true')
    p.add_argument('--use_boundary_16', action='store_true')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_weights = resolve_checkpoint_path(args.baseline_weights, args.scale)
    target_weights = resolve_checkpoint_path(args.target_weights, args.scale)

    print('baseline_weights:', baseline_weights)
    print('target_weights:', target_weights)
    print('data_root:', osp.abspath(args.data_root))
    print('out_dir:', osp.abspath(args.out_dir))
    print('backbone:', args.backbone)
    print('scale:', args.scale)
    print('device:', device)

    csv_path = osp.abspath(args.csv_path) if args.csv_path else osp.abspath(
        default_csv_path(baseline_weights, target_weights, args.out_dir)
    )

    baseline_result = evaluate_checkpoint(
        baseline_weights,
        args.data_root,
        args.backbone,
        args.scale,
        args.batch_size,
        args.num_workers,
        args.use_boundary_2,
        args.use_boundary_4,
        args.use_boundary_8,
        args.use_boundary_16,
        device,
    )
    target_result = evaluate_checkpoint(
        target_weights,
        args.data_root,
        args.backbone,
        args.scale,
        args.batch_size,
        args.num_workers,
        args.use_boundary_2,
        args.use_boundary_4,
        args.use_boundary_8,
        args.use_boundary_16,
        device,
    )

    print_single_result('Baseline per-class metrics', baseline_result)
    print_single_result('Target per-class metrics', target_result)
    print_comparison(baseline_result, target_result)
    write_comparison_csv(csv_path, baseline_result, target_result)
    print('\nSaved CSV:', csv_path)


if __name__ == '__main__':
    main()
