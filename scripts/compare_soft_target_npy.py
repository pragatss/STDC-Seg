#!/usr/bin/env python3
"""
Visualize and validate a precomputed soft-target `.npy` file.

This script is meant for cached ZeroPlane soft targets saved by
`scripts/precompute_soft_targets.py`. It loads one `.npy`, validates its
contents, converts it into human-readable images, and optionally places it next
to an existing predicted planar / segmentation output image.

Example:
    python scripts/compare_soft_target_npy.py \
        --npy soft_targets/train/bremen/bremen_000000_000019.npy \
        --out scripts/bremen_000000_000019_rgb.png

    # Optional full comparison view:
    python scripts/compare_soft_target_npy.py \
        --npy soft_targets/train/bremen/bremen_000000_000019.npy \
        --pred-image path/to/your/predicted_output.png \
        --out scripts/compare_bremen_000000_000019.png

Expected soft-target format:
    - shape: (21, 128, 256) or (128, 256, 21)
    - dtype: float16 / float32
    - values: typically in [0, 1]
"""

import argparse
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _uint82bin(n, count=8):
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def _labelcolormap(N):
    """ZeroPlane's own color scheme — matches visualizationBatch in disp.py."""
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0; g = 0; b = 0
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


# 256-entry palette matching ZeroPlane's labelcolormap(256).
# Channel index 0 = non-plane/background (shown as original image).
ZEROPLANE_PALETTE = _labelcolormap(256)

# Keep 21-slot alias for array indexing
CLASS_NAMES = ['plane-{:02d}'.format(i) for i in range(21)]
CLASS_NAMES[0] = '<non-plane>'
CITYSCAPES_PALETTE = ZEROPLANE_PALETTE[:21]  # backward compat alias


def _resolve_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if osp.isabs(path_value):
        return path_value
    return osp.normpath(osp.join(ROOT, path_value))


def _load_font(size: int):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
    ]
    for candidate in candidates:
        if osp.isfile(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _normalize_sem_seg(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError('Expected a 3D tensor, got shape {}'.format(array.shape))

    if array.shape[0] == len(CLASS_NAMES):
        sem_seg = array
    elif array.shape[-1] == len(CLASS_NAMES):
        sem_seg = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(
            'Expected {} channels in first or last dimension, got shape {}'.format(
                len(CLASS_NAMES), array.shape
            )
        )

    return sem_seg.astype(np.float32, copy=False)


def _to_uint8_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('RGB'), dtype=np.uint8)


def _load_pred_image(pred_image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    image = Image.open(pred_image_path)
    arr = np.asarray(image)

    if arr.ndim == 2:
        if np.issubdtype(arr.dtype, np.integer) and arr.min() >= 0 and arr.max() < len(CITYSCAPES_PALETTE):
            rgb = CITYSCAPES_PALETTE[arr]
            image = Image.fromarray(rgb)
        else:
            image = image.convert('RGB')
    elif arr.ndim == 3 and arr.shape[2] == 1:
        image = image.convert('RGB')
    else:
        image = image.convert('RGB')

    target_w, target_h = target_size
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.NEAREST)

    return _to_uint8_rgb(image)


def _make_colorized_argmax(sem_seg: np.ndarray) -> np.ndarray:
    """Colorize using ZeroPlane's labelcolormap; force ch20 (non-plane) to black."""
    argmax = sem_seg.argmax(axis=0)  # (H, W), values 0..20
    rgb = ZEROPLANE_PALETTE[argmax % 256].copy()
    rgb[argmax == 20] = 0
    return rgb


def _make_confidence_heatmap(sem_seg: np.ndarray) -> np.ndarray:
    confidence = np.clip(sem_seg.max(axis=0), 0.0, 1.0)
    return _simple_heatmap(confidence)


def _make_sum_heatmap(sem_seg: np.ndarray) -> np.ndarray:
    sum_map = sem_seg.sum(axis=0)
    max_value = float(sum_map.max()) if sum_map.size else 1.0
    denom = max(max_value, 1e-6)
    normalized = np.clip(sum_map / denom, 0.0, 1.0)
    return _simple_heatmap(normalized)


def _simple_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 * values - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(2.0 * values - 1.0) * 1.6, 0.0, 1.0)
    b = np.clip(1.2 - 1.5 * values, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _make_entropy_heatmap(sem_seg: np.ndarray) -> np.ndarray:
    clipped = np.clip(sem_seg, 1e-8, None)
    probs = clipped / np.clip(clipped.sum(axis=0, keepdims=True), 1e-8, None)
    entropy = -(probs * np.log(probs)).sum(axis=0)
    entropy /= math.log(sem_seg.shape[0])
    return _simple_heatmap(entropy)


def _validation_report(sem_seg: np.ndarray, original_dtype: str) -> Dict[str, object]:
    c, h, w = sem_seg.shape
    argmax = sem_seg.argmax(axis=0)
    max_probs = sem_seg.max(axis=0)
    sum_map = sem_seg.sum(axis=0)
    unique, counts = np.unique(argmax, return_counts=True)

    class_distribution: List[Dict[str, object]] = []
    total_pixels = int(argmax.size)
    for cls_idx, count in sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: -item[1]):
        class_distribution.append({
            'class_id': int(cls_idx),
            'class_name': CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx),
            'pixels': int(count),
            'fraction': float(count / total_pixels),
            'mean_score': float(sem_seg[cls_idx].mean()),
        })

    return {
        'shape': [int(c), int(h), int(w)],
        'dtype_on_disk': original_dtype,
        'dtype_loaded': str(sem_seg.dtype),
        'global_min': float(sem_seg.min()),
        'global_max': float(sem_seg.max()),
        'global_mean': float(sem_seg.mean()),
        'per_pixel_sum_min': float(sum_map.min()),
        'per_pixel_sum_max': float(sum_map.max()),
        'per_pixel_sum_mean': float(sum_map.mean()),
        'top_confidence_min': float(max_probs.min()),
        'top_confidence_max': float(max_probs.max()),
        'top_confidence_mean': float(max_probs.mean()),
        'top_classes': class_distribution[:10],
        'all_values_finite': bool(np.isfinite(sem_seg).all()),
        'values_in_0_1': bool((sem_seg >= 0.0).all() and (sem_seg <= 1.0).all()),
    }


def _add_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    font = _load_font(15)
    width = image_rgb.shape[1]
    bar = Image.new('RGB', (width, 28), (28, 28, 28))
    draw = ImageDraw.Draw(bar)
    draw.text((8, 6), title, fill=(230, 230, 230), font=font)
    return np.vstack([np.asarray(bar, dtype=np.uint8), image_rgb])


def _draw_stats_panel(report: Dict[str, object], width: int, height: int) -> np.ndarray:
    font = _load_font(14)
    small_font = _load_font(13)
    panel = Image.new('RGB', (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(panel)

    lines = [
        'Validation summary',
        'shape: {}'.format(tuple(report['shape'])),
        'dtype: {} -> {}'.format(report['dtype_on_disk'], report['dtype_loaded']),
        'global min/max: {:.4f} / {:.4f}'.format(report['global_min'], report['global_max']),
        'global mean: {:.4f}'.format(report['global_mean']),
        'sum min/max: {:.4f} / {:.4f}'.format(report['per_pixel_sum_min'], report['per_pixel_sum_max']),
        'sum mean: {:.4f}'.format(report['per_pixel_sum_mean']),
        'top conf min/max: {:.4f} / {:.4f}'.format(report['top_confidence_min'], report['top_confidence_max']),
        'top conf mean: {:.4f}'.format(report['top_confidence_mean']),
        'finite: {}'.format(report['all_values_finite']),
        'within [0,1]: {}'.format(report['values_in_0_1']),
        '',
        'Top argmax classes:',
    ]

    y = 10
    for index, line in enumerate(lines):
        draw.text((10, y), line, fill=(235, 235, 235), font=font if index == 0 else small_font)
        y += 20

    for item in report['top_classes'][:8]:
        cls_id = int(item['class_id'])
        color = tuple(CITYSCAPES_PALETTE[cls_id % len(CITYSCAPES_PALETTE)].tolist())
        text = '  {:02d} {:<14} {:6.2f}% mean={:.4f}'.format(
            cls_id,
            item['class_name'][:14],
            100.0 * float(item['fraction']),
            float(item['mean_score']),
        )
        draw.text((10, y), text, fill=color, font=small_font)
        y += 18
        if y > height - 20:
            break

    return np.asarray(panel, dtype=np.uint8)


def _pad_to_height(image_rgb: np.ndarray, height: int) -> np.ndarray:
    if image_rgb.shape[0] >= height:
        return image_rgb
    pad = np.zeros((height - image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
    return np.vstack([image_rgb, pad])


def _build_mosaic(panels: List[np.ndarray], columns: int) -> np.ndarray:
    rows = []
    for row_start in range(0, len(panels), columns):
        row_panels = panels[row_start:row_start + columns]
        target_h = max(panel.shape[0] for panel in row_panels)
        padded = [_pad_to_height(panel, target_h) for panel in row_panels]
        if len(padded) < columns:
            blank_width = padded[0].shape[1]
            blank = np.zeros((target_h, blank_width, 3), dtype=np.uint8)
            padded.extend([blank] * (columns - len(padded)))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def _auto_find_source_image(npy_path: str, data_root: str) -> str:
    """
    Given soft_targets/train/bremen/bremen_000000_000019.npy,
    try to find data/leftImg8bit/train/bremen/bremen_000000_000019_leftImg8bit.png.
    """
    npy_abs = osp.abspath(npy_path)
    stem = osp.splitext(osp.basename(npy_abs))[0]   # e.g. bremen_000000_000019
    city = osp.basename(osp.dirname(npy_abs))        # e.g. bremen
    split = osp.basename(osp.dirname(osp.dirname(npy_abs)))  # e.g. train

    candidate_roots = [
        data_root,
        osp.join(ROOT, 'data'),
        osp.join(ROOT, 'data', 'all_data'),
    ]
    for root in candidate_roots:
        candidate = osp.join(root, 'leftImg8bit', split, city,
                             '{}_leftImg8bit.png'.format(stem))
        if osp.isfile(candidate):
            return candidate
    return ''


def _blend_seg_with_image(seg_rgb: np.ndarray, original_rgb: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Blend exactly as ZeroPlane's visualizationBatch:
       - plane pixels  (argmax != 0): seg*0.7 + image*0.3
       - non-plane pixels (argmax == 0): original image unchanged
    seg_rgb is already colorized via _make_colorized_argmax.
    """
    if original_rgb.shape[:2] != seg_rgb.shape[:2]:
        orig_pil = Image.fromarray(original_rgb).resize(
            (seg_rgb.shape[1], seg_rgb.shape[0]), Image.BILINEAR)
        original_rgb = np.asarray(orig_pil, dtype=np.uint8)

    blended = (alpha * seg_rgb.astype(np.float32) +
               (1.0 - alpha) * original_rgb.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Non-plane mask: where seg color == [0,0,0] (palette index 0 = black)
    plane_mask = (seg_rgb.sum(axis=2) > 0).astype(np.uint8)[:, :, np.newaxis]
    return blended * plane_mask + original_rgb * (1 - plane_mask)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a soft-target .npy to RGB, optionally with comparison panels')
    parser.add_argument('--npy', required=True, help='Path to the soft-target .npy file')
    parser.add_argument('--pred-image', default='', help='Optional predicted planar/segmentation output image to compare against')
    parser.add_argument('--original-image', default='', help='Optional original RGB image for comparison panel only (no blending)')
    parser.add_argument('--data-root', default='', help='Cityscapes data root (unused for rgb-only output)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Deprecated: rgb-only output is no longer blended')
    parser.add_argument('--out', default='', help='Output comparison image path (.png)')
    parser.add_argument('--report', default='', help='Optional JSON report path')
    parser.add_argument('--columns', type=int, default=3, help='Number of panels per row in the output mosaic')
    parser.add_argument('--rgb-only', action='store_true', help='Save only the colorized argmax RGB image (blended if source image found)')
    return parser.parse_args()


def main():
    args = parse_args()

    npy_path = _resolve_path(args.npy)
    pred_image_path = _resolve_path(args.pred_image) if args.pred_image else ''
    original_image_path = _resolve_path(args.original_image) if args.original_image else ''
    data_root = _resolve_path(args.data_root) if args.data_root else ''

    if not osp.isfile(npy_path):
        raise FileNotFoundError('Soft target file does not exist: {}'.format(npy_path))
    if pred_image_path and not osp.isfile(pred_image_path):
        raise FileNotFoundError('Predicted image does not exist: {}'.format(pred_image_path))
    if original_image_path and not osp.isfile(original_image_path):
        raise FileNotFoundError('Original image does not exist: {}'.format(original_image_path))

    raw = np.load(npy_path)
    report = _validation_report(_normalize_sem_seg(raw), str(raw.dtype))
    sem_seg = _normalize_sem_seg(raw)
    _, h, w = sem_seg.shape

    argmax_rgb = _make_colorized_argmax(sem_seg)
    rgb_only = args.rgb_only or (not pred_image_path)

    default_base = osp.splitext(osp.basename(npy_path))[0]
    if rgb_only:
        out_path = _resolve_path(args.out) if args.out else osp.join(ROOT, 'scripts', '{}_rgb.png'.format(default_base))
        os.makedirs(osp.dirname(out_path), exist_ok=True)

        Image.fromarray(argmax_rgb).save(out_path)
        print('Loaded: {}'.format(npy_path))
        print('Saved RGB image to: {}'.format(out_path))
        print('Shape: {}'.format(tuple(report['shape'])))
        print('Value range: {:.4f} .. {:.4f}'.format(report['global_min'], report['global_max']))
        return

    confidence_rgb = _make_confidence_heatmap(sem_seg)
    sum_rgb = _make_sum_heatmap(sem_seg)
    entropy_rgb = _make_entropy_heatmap(sem_seg)

    panels = []
    if original_image_path:
        original_rgb = _load_pred_image(original_image_path, (w, h))
        panels.append(_add_title(original_rgb, 'Original image'))

    if pred_image_path:
        pred_rgb = _load_pred_image(pred_image_path, (w, h))
        panels.append(_add_title(pred_rgb, 'Provided predicted output'))

    panels.extend([
        _add_title(argmax_rgb, 'Soft target argmax (colorized)'),
        _add_title(confidence_rgb, 'Top-class confidence'),
        _add_title(sum_rgb, 'Per-pixel channel sum'),
        _add_title(entropy_rgb, 'Normalized entropy'),
        _draw_stats_panel(report, width=w, height=max(h + 28, 260)),
    ])

    mosaic = _build_mosaic(panels, columns=max(1, args.columns))

    out_path = _resolve_path(args.out) if args.out else osp.join(ROOT, 'scripts', '{}_compare.png'.format(default_base))
    report_path = _resolve_path(args.report) if args.report else osp.join(ROOT, 'scripts', '{}_report.json'.format(default_base))

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    os.makedirs(osp.dirname(report_path), exist_ok=True)

    Image.fromarray(mosaic).save(out_path)
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    print('Loaded: {}'.format(npy_path))
    print('Saved comparison image to: {}'.format(out_path))
    print('Saved validation report to: {}'.format(report_path))
    print('Shape: {}'.format(tuple(report['shape'])))
    print('Value range: {:.4f} .. {:.4f}'.format(report['global_min'], report['global_max']))
    print('Per-pixel sum range: {:.4f} .. {:.4f}'.format(report['per_pixel_sum_min'], report['per_pixel_sum_max']))
    print('Top confidence mean: {:.4f}'.format(report['top_confidence_mean']))
    print('Top classes:')
    for item in report['top_classes'][:5]:
        print('  {:02d} {:<14} {:6.2f}%'.format(
            int(item['class_id']),
            item['class_name'],
            100.0 * float(item['fraction']),
        ))


if __name__ == '__main__':
    main()
