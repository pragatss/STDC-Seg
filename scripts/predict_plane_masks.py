#!/usr/bin/env python3
"""
Visualize per-query planar mask predictions from the SetCriterion planar head.

For each image, produces:
  <out_dir>/plane_argmax/<stem>.png   — each pixel coloured by winning query index
  <out_dir>/plane_overlay/<stem>.png  — above blended onto the RGB input
  <out_dir>/plane_active/<stem>.png   — only queries not predicted as "no-object"

Usage:
    conda run -n stdcseg18 --no-capture-output python scripts/predict_plane_masks.py \
        --weights checkpoints/train_STDC1-Seg/plane_setcriterion_w005_delay20k \
        --out_dir predictions/plane_masks_w005_delay20k

    # compare two models side-by-side:
    conda run -n stdcseg18 --no-capture-output python scripts/predict_plane_masks.py \
        --weights checkpoints/train_STDC1-Seg/plane_setcriterion_w005_delay20k \
        --compare_weights checkpoints/train_STDC1-Seg/baseline_no_plane \
        --out_dir predictions/plane_masks_comparison
"""

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model_stages import BiSeNet

DEFAULT_IMAGES_DIR = osp.join(ROOT, 'scripts', 'images')
FONT_SIZE = 20
BORDER = 4


# ── distinct colour palette for up to 21 planar queries ─────────────────────
QUERY_PALETTE = np.array([
    [230,  25,  75],  [60, 180,  75],  [255, 225,  25],  [  0, 130, 200],
    [245, 130,  48],  [145,  30, 180],  [ 70, 240, 240],  [240,  50, 230],
    [210, 245,  60],  [250, 190, 212],  [  0, 128, 128],  [220, 190, 255],
    [170, 110,  40],  [255, 250, 200],  [128,   0,   0],  [170, 255, 195],
    [128, 128,   0],  [255, 215, 180],  [  0,   0, 128],  [128, 128, 128],
    [255, 255, 255],  # slot 20 — no-object (white)
], dtype=np.uint8)

NOOBJ_COLOR = np.array([30, 30, 30], dtype=np.uint8)  # dark grey = no-object area


def load_font():
    try:
        return ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def draw_label(img_array, text, padding=5):
    img = Image.fromarray(img_array.copy())
    draw = ImageDraw.Draw(img)
    font = load_font()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([padding, padding, padding + tw + padding, padding + th + padding],
                   fill=(0, 0, 0))
    draw.text((padding * 2, padding * 2), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def border(h):
    return np.ones((h, BORDER, 3), dtype=np.uint8) * 200


def blend(rgb, mask_rgb, alpha=0.55):
    return np.clip((1 - alpha) * rgb.astype(np.float32) +
                   alpha * mask_rgb.astype(np.float32), 0, 255).astype(np.uint8)


def resolve_ckpt(weights_arg, scale=0.5):
    path = osp.abspath(weights_arg)
    tag = '75' if scale >= 0.625 else '50'
    if osp.isdir(path):
        for candidate in [
            osp.join(path, 'pths', f'model_maxmIOU{tag}.pth'),
            osp.join(path, f'model_maxmIOU{tag}.pth'),
            osp.join(path, 'pths', 'model_final.pth'),
        ]:
            if osp.isfile(candidate):
                return candidate
        raise FileNotFoundError(f'No checkpoint found under {path}')
    if not osp.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def build_model(ckpt_path, backbone, device,
                num_queries=20, num_classes=20,
                hidden_dim=256, decoder_layers=10):
    net = BiSeNet(
        backbone=backbone,
        n_classes=19,
        use_boundary_2=False,
        use_boundary_4=False,
        use_boundary_8=False,
        use_boundary_16=False,
        use_plane_aux=True,
        plane_aux_tap='fuse',
        plane_aux_mid=64,
        plane_aux_mode='setcriterion',
        plane_aux_num_queries=num_queries,
        plane_aux_num_classes=num_classes,
        plane_aux_hidden_dim=hidden_dim,
        plane_aux_decoder_layers=decoder_layers,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print(f'[load] {len(missing)} missing keys')
    net.to(device).eval()
    return net


def preprocess(img_rgb):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tfm(Image.fromarray(img_rgb)).unsqueeze(0)


def run_inference(net, img_rgb, device, scale=0.5):
    """Returns (pred_masks, pred_logits) both upsampled to original image size."""
    x = preprocess(img_rgb).to(device)
    H, W = x.shape[-2:]
    sh, sw = max(1, int(H * scale)), max(1, int(W * scale))
    x_s = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=True)

    with torch.no_grad():
        net_out = net(x_s)

    # Unpack: no boundary heads, with plane_aux → (out, out16, out32, plane_aux_out, soft_target)
    if isinstance(net_out, (tuple, list)) and len(net_out) >= 4:
        plane_aux_out = net_out[-2]   # second-to-last
    else:
        raise RuntimeError('Unexpected net output format: {}'.format(
            [type(o).__name__ for o in net_out] if isinstance(net_out, (tuple, list)) else type(net_out)))

    if not isinstance(plane_aux_out, dict):
        raise RuntimeError(
            'plane_aux_out is not a dict — is --plane_aux_mode setcriterion set in the loaded model?')

    pred_masks  = plane_aux_out['pred_masks']   # (1, Q, h, w)
    pred_logits = plane_aux_out['pred_logits']  # (1, Q, C+1)

    # Upsample masks to original image size
    pred_masks = F.interpolate(pred_masks, size=(H, W),
                               mode='bilinear', align_corners=True)

    return pred_masks.squeeze(0), pred_logits.squeeze(0)   # (Q,H,W), (Q,C+1)


def masks_to_rgb(pred_masks, pred_logits, num_classes, active_only=False):
    """
    pred_masks : (Q, H, W) — raw logits
    pred_logits: (Q, C+1)  — classification logits

    Returns:
        argmax_rgb  — each pixel coloured by winning query index
        active_rgb  — same but no-object queries shown as dark grey
    """
    Q, H, W = pred_masks.shape
    palette = QUERY_PALETTE[:Q]

    # Winning query per pixel (argmax over sigmoid scores)
    mask_probs = torch.sigmoid(pred_masks)           # (Q,H,W)
    winner = mask_probs.argmax(dim=0).cpu().numpy()  # (H,W) int in [0,Q)

    argmax_rgb = palette[winner]                     # (H,W,3)

    # Which queries predict a foreground class (not no-object)?
    cls_pred = pred_logits.argmax(dim=-1).cpu().numpy()  # (Q,) each in [0, C]
    is_active = cls_pred < num_classes               # True = foreground

    active_winner = winner.copy()
    for q in range(Q):
        if not is_active[q]:
            active_winner[active_winner == q] = Q   # mark as noobj

    active_rgb = palette[np.minimum(active_winner, Q - 1)].copy()
    active_rgb[active_winner == Q] = NOOBJ_COLOR

    return argmax_rgb, active_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights',          required=True,
                   help='checkpoint dir or .pth for the plane-aux model')
    p.add_argument('--compare_weights',  default='',
                   help='optional second model checkpoint (baseline) for side-by-side')
    p.add_argument('--images_dir',       default=DEFAULT_IMAGES_DIR)
    p.add_argument('--out_dir',          default='')
    p.add_argument('--backbone',         default='STDCNet813',
                   choices=['STDCNet813', 'STDCNet1446'])
    p.add_argument('--scale',            type=float, default=0.5)
    p.add_argument('--alpha',            type=float, default=0.55,
                   help='overlay opacity')
    p.add_argument('--num_queries',      type=int, default=20)
    p.add_argument('--num_classes',      type=int, default=20)
    p.add_argument('--hidden_dim',       type=int, default=256)
    p.add_argument('--decoder_layers',   type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_a = resolve_ckpt(args.weights, args.scale)
    run_name = osp.basename(osp.dirname(ckpt_a) if osp.basename(ckpt_a).endswith('.pth')
                            else ckpt_a)
    out_dir = osp.abspath(args.out_dir) if args.out_dir else osp.join(
        ROOT, 'predictions', 'plane_masks_' + osp.basename(
            args.weights.rstrip('/').rstrip('\\')))

    argmax_dir  = osp.join(out_dir, 'plane_argmax')
    overlay_dir = osp.join(out_dir, 'plane_overlay')
    active_dir  = osp.join(out_dir, 'plane_active')
    for d in (argmax_dir, overlay_dir, active_dir):
        os.makedirs(d, exist_ok=True)

    images_dir = osp.abspath(args.images_dir)
    image_paths = sorted(
        glob.glob(osp.join(images_dir, '*.png')) +
        glob.glob(osp.join(images_dir, '*.jpg'))
    )
    if not image_paths:
        raise FileNotFoundError('No images in ' + images_dir)

    print('[plane_masks] loading model A:', ckpt_a)
    net_a = build_model(ckpt_a, args.backbone, device,
                        args.num_queries, args.num_classes,
                        args.hidden_dim, args.decoder_layers)

    # Optional baseline segmentation model for side-by-side strip
    net_b = None
    if args.compare_weights:
        from scripts.predict_images import (
            resolve_checkpoint_path, build_model as build_seg_model,
            predict_mask, cityscapes_palette,
        )
        ckpt_b = resolve_checkpoint_path(args.compare_weights, args.scale)
        print('[plane_masks] loading comparison model B:', ckpt_b)
        net_b = build_seg_model(ckpt_b, args.backbone, device)
        seg_palette = cityscapes_palette()

    for img_path in image_paths:
        stem = osp.splitext(osp.basename(img_path))[0]
        img_rgb = np.array(Image.open(img_path).convert('RGB'))

        pred_masks, pred_logits = run_inference(net_a, img_rgb, device, args.scale)
        argmax_rgb, active_rgb = masks_to_rgb(pred_masks, pred_logits, args.num_classes)

        argmax_overlay = blend(img_rgb, argmax_rgb, args.alpha)
        active_overlay = blend(img_rgb, active_rgb, args.alpha)

        Image.fromarray(argmax_rgb).save(osp.join(argmax_dir, stem + '.png'))
        Image.fromarray(argmax_overlay).save(osp.join(overlay_dir, stem + '.png'))
        Image.fromarray(active_overlay).save(osp.join(active_dir, stem + '.png'))

        print('[saved]', stem)

    print('[done] plane mask visualizations saved to', out_dir)


if __name__ == '__main__':
    main()
