#!/usr/bin/env python3
"""
Visualize a few generated soft targets from scripts/generated_softtargets.

Output:
    scripts/combined_with_stdcpred/

Each preview shows:
    [RGB input] | [soft-target plane instances] | [STDC prediction (color)] | [STDC overlay]

Rules:
- Plane queries 0..19 get distinct colors
- Non-plane query 20 is pure black

Usage:
  cd /home/husky/Downloads/Pragat/STDC-Seg-Mod
    conda run -n stdcseg18 --no-capture-output python scripts/visualize_generated_softtargets.py
"""

import argparse
import glob
import os
import os.path as osp

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms

from models.model_stages import BiSeNet

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
IMAGES_DIR = osp.join(ROOT, 'scripts', 'images')
SOFT_DIR = osp.join(ROOT, 'scripts', 'generated_softtargets')
OUT_DIR = osp.join(ROOT, 'scripts', 'combined_with_stdcpred')

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


def cityscapes_palette():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ], dtype=np.uint8)


def build_model(ckpt_path, backbone, device):
    net = BiSeNet(
        backbone=backbone,
        n_classes=19,
        use_boundary_2=False,
        use_boundary_4=False,
        use_boundary_8=True,
        use_boundary_16=False,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        cleaned[nk] = v
    net.load_state_dict(cleaned, strict=False)
    net.to(device)
    net.eval()
    return net


def stdc_predict(net, img_rgb, device, scale):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_pil = Image.fromarray(img_rgb)
    x = to_tensor(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        h, w = x.shape[-2:]
        sh, sw = int(h * scale), int(w * scale)
        x_scaled = torch.nn.functional.interpolate(
            x, size=(sh, sw), mode='bilinear', align_corners=True
        )
        logits = net(x_scaled)[0]
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode='bilinear', align_corners=True
        )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def colorize_pred(pred, palette):
    return palette[pred]


def blend_rgb(img_rgb, seg_rgb, alpha=0.5):
    img_f = img_rgb.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)
    return np.clip((1.0 - alpha) * img_f + alpha * seg_f, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Visualize soft targets together with STDC segmentation predictions.')
    parser.add_argument('--images_dir', default=IMAGES_DIR)
    parser.add_argument('--soft_dir', default=SOFT_DIR)
    parser.add_argument('--out_dir', default=OUT_DIR)
    parser.add_argument('--weights', default=osp.join(ROOT, 'checkpoints', 'STDC1-Seg', 'model_maxmIOU75.pth'))
    parser.add_argument('--backbone', default='STDCNet813', choices=['STDCNet813', 'STDCNet1446'])
    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--max_images', type=int, default=MAX_IMAGES)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading STDC checkpoint: {args.weights} | backbone={args.backbone} | device={device}')
    net = build_model(args.weights, args.backbone, device)
    palette = cityscapes_palette()

    image_paths = sorted(glob.glob(osp.join(args.images_dir, '*.png')) +
                         glob.glob(osp.join(args.images_dir, '*.jpg')))
    image_paths = image_paths[:args.max_images]
    saved_mosaics = []

    if not image_paths:
        print('No images found in', args.images_dir)
        return

    for img_path in image_paths:
        stem = osp.splitext(osp.basename(img_path))[0]
        soft_path = osp.join(args.soft_dir, stem + '.npy')
        if not osp.isfile(soft_path):
            print(f'[SKIP] missing soft target: {soft_path}')
            continue

        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        img_rgb = resize_rgb(img_rgb, PANEL_H, PANEL_W)

        soft = np.load(soft_path).astype(np.float32)  # (21,H,W)
        argmax_map = soft.argmax(axis=0).astype(np.int32)
        soft_rgb = render_softtarget_black(argmax_map)
        soft_rgb = resize_rgb(soft_rgb, PANEL_H, PANEL_W)

        pred = stdc_predict(net, img_rgb, device, args.scale)
        pred_rgb = colorize_pred(pred, palette)
        pred_rgb = resize_rgb(pred_rgb, PANEL_H, PANEL_W)
        overlay = blend_rgb(img_rgb, colorize_pred(pred, palette), alpha=args.alpha)
        overlay = resize_rgb(overlay, PANEL_H, PANEL_W)

        n_plane_queries = len(np.unique(argmax_map[argmax_map < NUM_QUERIES]))
        nonplane_frac = float((argmax_map == NUM_QUERIES).mean()) * 100.0

        mosaic = np.hstack([
            title_bar(img_rgb, 'RGB Input'),
            title_bar(soft_rgb, f'Soft-target plane instances | queries={n_plane_queries} | non-plane black ({nonplane_frac:.1f}% np)'),
            title_bar(pred_rgb, 'STDC Prediction (19 classes)'),
            title_bar(overlay, f'STDC Overlay alpha={args.alpha:.2f}'),
        ])
        saved_mosaics.append(mosaic)

        out_path = osp.join(args.out_dir, stem + '_softtarget_with_stdc.png')
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

        combined_path = osp.join(args.out_dir, 'combined_softtarget_with_stdc_contact_sheet.png')
        Image.fromarray(combined).save(combined_path)
        print('saved:', combined_path)

    print('Done. Previews saved to:', args.out_dir)


if __name__ == '__main__':
    main()
