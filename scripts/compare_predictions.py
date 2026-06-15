#!/usr/bin/env python3
"""
Side-by-side comparison of two STDC-Seg model predictions.

Produces a grid for each image:
    [ RGB input | Model A overlay | Model B overlay ]

Usage:
    conda run -n stdcseg18 --no-capture-output python scripts/compare_predictions.py \
        --model_a predictions/baseline_no_plane \
        --model_b predictions/plane_setcriterion_w005_delay20k \
        --out_dir predictions/comparison_baseline_vs_w005_delay20k
"""

import argparse
import glob
import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DEFAULT_IMAGES_DIR = osp.join(ROOT, 'scripts', 'images')
FONT_SIZE = 22
BORDER = 4                 # white border between panels


def load_image(path):
    return np.array(Image.open(path).convert('RGB'))


def find_overlays(pred_dir):
    """Return {stem: path} for all overlay images under pred_dir/overlay/."""
    overlay_dir = osp.join(pred_dir, 'overlay')
    if not osp.isdir(overlay_dir):
        # Fallback: check pred_dir/color/ or pred_dir itself
        for sub in ('color', ''):
            d = osp.join(pred_dir, sub) if sub else pred_dir
            paths = sorted(glob.glob(osp.join(d, '*.png')) +
                           glob.glob(osp.join(d, '*.jpg')))
            if paths:
                overlay_dir = d
                break
    paths = sorted(glob.glob(osp.join(overlay_dir, '*.png')) +
                   glob.glob(osp.join(overlay_dir, '*.jpg')))
    return {osp.splitext(osp.basename(p))[0]: p for p in paths}


def draw_label_on_image(img_array, text, padding=6):
    """Draw text label directly onto the top-left of the image with a dark background box."""
    img = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Dark semi-transparent box behind text
    box_x1, box_y1 = padding, padding
    box_x2, box_y2 = padding + tw + padding, padding + th + padding
    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=(0, 0, 0))
    draw.text((box_x1 + padding, box_y1 + padding), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def make_border(h, w):
    return np.ones((h, BORDER, 3), dtype=np.uint8) * 255


def build_comparison(rgb, overlay_a, overlay_b, label_rgb, label_a, label_b):
    col_rgb = draw_label_on_image(rgb,       label_rgb)
    col_a   = draw_label_on_image(overlay_a, label_a)
    col_b   = draw_label_on_image(overlay_b, label_b)

    h = col_rgb.shape[0]
    w = col_rgb.shape[1]
    grid = np.hstack([
        col_rgb,
        make_border(h, w),
        col_a,
        make_border(h, w),
        col_b,
    ])
    return grid


def parse_args():
    p = argparse.ArgumentParser(description='Side-by-side prediction comparison.')
    p.add_argument('--model_a',    required=True, help='predictions dir for model A (baseline)')
    p.add_argument('--model_b',    required=True, help='predictions dir for model B (plane aux)')
    p.add_argument('--images_dir', default=DEFAULT_IMAGES_DIR, help='original RGB images directory')
    p.add_argument('--out_dir',    default='', help='output directory')
    p.add_argument('--label_a',    default='', help='label for model A column (default: dir name)')
    p.add_argument('--label_b',    default='', help='label for model B column (default: dir name)')
    return p.parse_args()


def main():
    args = parse_args()

    model_a_dir = osp.abspath(args.model_a)
    model_b_dir = osp.abspath(args.model_b)
    images_dir  = osp.abspath(args.images_dir)

    label_a = args.label_a or osp.basename(model_a_dir)
    label_b = args.label_b or osp.basename(model_b_dir)

    out_dir = osp.abspath(args.out_dir) if args.out_dir else osp.join(
        ROOT, 'predictions',
        'comparison_{}__vs__{}'.format(osp.basename(model_a_dir), osp.basename(model_b_dir))
    )
    os.makedirs(out_dir, exist_ok=True)

    overlays_a = find_overlays(model_a_dir)
    overlays_b = find_overlays(model_b_dir)

    # Find stems present in both sets
    common_stems = sorted(set(overlays_a.keys()) & set(overlays_b.keys()))
    if not common_stems:
        raise ValueError(
            'No common images found between:\n  {}\n  {}'.format(model_a_dir, model_b_dir)
        )

    print('[compare] model A  =', model_a_dir, '({} images)'.format(len(overlays_a)))
    print('[compare] model B  =', model_b_dir, '({} images)'.format(len(overlays_b)))
    print('[compare] common   =', len(common_stems), 'images')
    print('[compare] out_dir  =', out_dir)

    # Try to find matching RGB originals
    rgb_paths = {}
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for p in glob.glob(osp.join(images_dir, ext)):
            rgb_paths[osp.splitext(osp.basename(p))[0]] = p

    for stem in common_stems:
        ov_a = load_image(overlays_a[stem])
        ov_b = load_image(overlays_b[stem])

        if stem in rgb_paths:
            rgb = load_image(rgb_paths[stem])
            # Resize RGB to match overlay size if needed
            if rgb.shape[:2] != ov_a.shape[:2]:
                rgb = np.array(Image.fromarray(rgb).resize(
                    (ov_a.shape[1], ov_a.shape[0]), Image.BILINEAR))
        else:
            # No RGB available: use a grey placeholder
            rgb = np.full_like(ov_a, 128)

        # Resize B to match A if they differ
        if ov_b.shape[:2] != ov_a.shape[:2]:
            ov_b = np.array(Image.fromarray(ov_b).resize(
                (ov_a.shape[1], ov_a.shape[0]), Image.BILINEAR))

        grid = build_comparison(rgb, ov_a, ov_b,
                                label_rgb='Input',
                                label_a=label_a,
                                label_b=label_b)

        out_path = osp.join(out_dir, stem + '.png')
        Image.fromarray(grid).save(out_path)
        print('[saved]', stem)

    print('[done] comparisons saved to', out_dir)


if __name__ == '__main__':
    main()
