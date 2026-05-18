#!/usr/bin/env python3
"""
Visualize a few generated soft targets from scripts/generated_softtargets.

Output:
  scripts/softtarget_previews_black/

Each preview shows:
  [RGB input] | [soft-target plane instances]

Rules:
- Plane queries 0..19 get distinct colors
- Non-plane query 20 is pure black

Usage:
  cd /home/husky/Downloads/Pragat/STDC-Seg-Mod
  conda run -n stdcseg18 --no-capture-output python scripts/visualize_generated_softtargets.py
"""

import glob
import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
IMAGES_DIR = osp.join(ROOT, 'scripts', 'images')
SOFT_DIR = osp.join(ROOT, 'scripts', 'generated_softtargets')
OUT_DIR = osp.join(ROOT, 'scripts', 'softtarget_previews_black')

# Hardcoded: visualize a small subset only.
MAX_IMAGES = 6
PANEL_H, PANEL_W = 256, 512
NUM_QUERIES = 20  # 0..19 plane, 20 non-plane

os.makedirs(OUT_DIR, exist_ok=True)

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
except Exception:
    FONT = ImageFont.load_default()


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


def title_bar(img_np, text, bar_h=22):
    bar = np.full((bar_h, img_np.shape[1], 3), 30, dtype=np.uint8)
    pil = Image.fromarray(bar)
    ImageDraw.Draw(pil).text((4, 4), text, fill=(220, 220, 220), font=FONT)
    return np.vstack([np.array(pil), img_np])


def resize_rgb(img_np, h, w):
    return np.array(Image.fromarray(img_np).resize((w, h), Image.BILINEAR))


def render_softtarget_black(argmax_map, num_queries=NUM_QUERIES):
    """Plane queries get colors, non-plane stays black."""
    colors = labelcolormap(256)
    canvas = np.zeros((argmax_map.shape[0], argmax_map.shape[1], 3), dtype=np.uint8)
    plane_mask = argmax_map < num_queries
    plane_ids = argmax_map.copy().astype(np.int32)
    plane_ids[~plane_mask] = 0

    if plane_mask.any():
        plane_colors = np.stack([
            colors[plane_ids, 0],
            colors[plane_ids, 1],
            colors[plane_ids, 2],
        ], axis=2)
        canvas[plane_mask] = plane_colors[plane_mask]
    return canvas


def main():
    image_paths = sorted(glob.glob(osp.join(IMAGES_DIR, '*.png')) +
                         glob.glob(osp.join(IMAGES_DIR, '*.jpg')))
    image_paths = image_paths[:MAX_IMAGES]
    saved_mosaics = []

    if not image_paths:
        print('No images found in', IMAGES_DIR)
        return

    for img_path in image_paths:
        stem = osp.splitext(osp.basename(img_path))[0]
        soft_path = osp.join(SOFT_DIR, stem + '.npy')
        if not osp.isfile(soft_path):
            print(f'[SKIP] missing soft target: {soft_path}')
            continue

        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        img_rgb = resize_rgb(img_rgb, PANEL_H, PANEL_W)

        soft = np.load(soft_path).astype(np.float32)  # (21,H,W)
        argmax_map = soft.argmax(axis=0).astype(np.int32)
        soft_rgb = render_softtarget_black(argmax_map)
        soft_rgb = resize_rgb(soft_rgb, PANEL_H, PANEL_W)

        n_plane_queries = len(np.unique(argmax_map[argmax_map < NUM_QUERIES]))
        nonplane_frac = float((argmax_map == NUM_QUERIES).mean()) * 100.0

        mosaic = np.hstack([
            title_bar(img_rgb, 'RGB Input'),
            title_bar(soft_rgb, f'Soft-target plane instances | queries={n_plane_queries} | non-plane black ({nonplane_frac:.1f}% np)'),
        ])
        saved_mosaics.append(mosaic)

        out_path = osp.join(OUT_DIR, stem + '_softtarget_black.png')
        Image.fromarray(mosaic).save(out_path)
        print('saved:', out_path)

    if saved_mosaics:
        cols = 2
        rows = (len(saved_mosaics) + cols - 1) // cols
        tile_h = max(img.shape[0] for img in saved_mosaics)
        tile_w = max(img.shape[1] for img in saved_mosaics)

        padded = []
        for img in saved_mosaics:
            canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            canvas[:img.shape[0], :img.shape[1]] = img
            padded.append(canvas)

        while len(padded) < rows * cols:
            padded.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

        row_imgs = []
        for r in range(rows):
            row_imgs.append(np.hstack(padded[r * cols:(r + 1) * cols]))
        combined = np.vstack(row_imgs)

        combined_path = osp.join(OUT_DIR, 'combined_softtarget_black_contact_sheet.png')
        Image.fromarray(combined).save(combined_path)
        print('saved:', combined_path)

    print('Done. Previews saved to:', OUT_DIR)


if __name__ == '__main__':
    main()
