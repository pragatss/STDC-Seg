#!/usr/bin/env python3
"""
Run predictions from two checkpoints and save all available head outputs.

  baseline_no_plane checkpoint  →  seg predictions only (no plane head)
  plane-trained checkpoint      →  seg predictions  +  plane predictions

Output layout:
  predictions/independent_heads/
    baseline_no_plane/
      seg/color/<stem>.png       — argmax, Cityscapes palette
      seg/overlay/<stem>.png     — blended onto RGB
    plane_setcriterion_w005_delay20k/
      seg/color/<stem>.png
      seg/overlay/<stem>.png
      plane/argmax/<stem>.png    — each pixel coloured by winning query
      plane/overlay/<stem>.png   — plane argmax blended onto RGB
      plane/active/<stem>.png    — no-object queries shown dark-grey

Example:
    conda run -n stdcseg18 --no-capture-output python scripts/predict_both_heads.py

    # use a different plane checkpoint:
    conda run -n stdcseg18 --no-capture-output python scripts/predict_both_heads.py \\
        --plane_weights checkpoints/train_STDC1-Seg/plane_setcriterion_w003_delay20k
"""

import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model_stages import BiSeNet

DEFAULT_IMAGES_DIR    = osp.join(ROOT, 'scripts', 'images')
DEFAULT_OUT_ROOT      = osp.join(ROOT, 'predictions', 'independent_heads')
DEFAULT_SEG_WEIGHTS   = osp.join(ROOT, 'checkpoints', 'train_STDC1-Seg', 'baseline_no_plane')
DEFAULT_PLANE_WEIGHTS = osp.join(ROOT, 'checkpoints', 'train_STDC1-Seg', 'plane_setcriterion_w005_delay20k')


# ── colour palettes ───────────────────────────────────────────────────────────

CS_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

QUERY_PALETTE = np.array([
    [230,  25,  75], [ 60, 180,  75], [255, 225,  25], [  0, 130, 200],
    [245, 130,  48], [145,  30, 180], [ 70, 240, 240], [240,  50, 230],
    [210, 245,  60], [250, 190, 212], [  0, 128, 128], [220, 190, 255],
    [170, 110,  40], [255, 250, 200], [128,   0,   0], [170, 255, 195],
    [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128],
    [255, 255, 255],  # slot 20 — no-object
], dtype=np.uint8)

NOOBJ_COLOR = np.array([30, 30, 30], dtype=np.uint8)


# ── helpers ───────────────────────────────────────────────────────────────────

def blend(img_rgb, color_rgb, alpha=0.5):
    return np.clip(
        (1 - alpha) * img_rgb.astype(np.float32) +
        alpha * color_rgb.astype(np.float32), 0, 255
    ).astype(np.uint8)


def resolve_ckpt(weights_arg, scale=0.5):
    path = osp.abspath(weights_arg)
    tag = '75' if scale >= 0.625 else '50'
    if osp.isdir(path):
        for candidate in [
            osp.join(path, 'pths', f'model_maxmIOU{tag}.pth'),
            osp.join(path, f'model_maxmIOU{tag}.pth'),
            osp.join(path, 'pths', 'model_final.pth'),
            osp.join(path, 'model_final.pth'),
        ]:
            if osp.isfile(candidate):
                return candidate
        raise FileNotFoundError(f'No checkpoint found under {path}')
    if not osp.isfile(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return path


def load_state(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}


def build_seg_only_model(ckpt_path, backbone, device):
    """Baseline: no plane head."""
    net = BiSeNet(
        backbone=backbone, n_classes=19,
        use_boundary_2=False, use_boundary_4=False,
        use_boundary_8=False, use_boundary_16=False,
    )
    missing, unexpected = net.load_state_dict(load_state(ckpt_path), strict=False)
    if missing:
        print(f'  [seg-only load] {len(missing)} missing keys')
    if unexpected:
        print(f'  [seg-only load] {len(unexpected)} unexpected keys')
    return net.to(device).eval()


def build_plane_model(ckpt_path, backbone, device,
                      num_queries, num_classes, hidden_dim, decoder_layers):
    """Plane-trained: backbone + seg head + plane head."""
    net = BiSeNet(
        backbone=backbone, n_classes=19,
        use_boundary_2=False, use_boundary_4=False,
        use_boundary_8=False, use_boundary_16=False,
        use_plane_aux=True,
        plane_aux_tap='fuse',
        plane_aux_mid=64,
        plane_aux_mode='setcriterion',
        plane_aux_num_queries=num_queries,
        plane_aux_num_classes=num_classes,
        plane_aux_hidden_dim=hidden_dim,
        plane_aux_decoder_layers=decoder_layers,
    )
    missing, unexpected = net.load_state_dict(load_state(ckpt_path), strict=False)
    if missing:
        print(f'  [plane load] {len(missing)} missing keys')
    if unexpected:
        print(f'  [plane load] {len(unexpected)} unexpected keys')
    return net.to(device).eval()


def preprocess(img_rgb):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return tfm(Image.fromarray(img_rgb)).unsqueeze(0)


def run_forward(net, img_rgb, device, scale):
    """Single forward pass. Returns raw net_out tuple and original (H, W)."""
    x = preprocess(img_rgb).to(device)
    H, W = x.shape[-2:]
    x_s = F.interpolate(x, size=(max(1, int(H * scale)), max(1, int(W * scale))),
                        mode='bilinear', align_corners=True)
    with torch.no_grad():
        net_out = net(x_s)
    return net_out, H, W


def extract_seg(net_out, H, W):
    """Returns (H, W) uint8 argmax label map."""
    logits = net_out[0]
    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def extract_plane(net_out, H, W):
    """Returns (masks (Q,H,W), logits (Q,C+1)) or None if no plane output."""
    plane_dict = next(
        (item for item in net_out if isinstance(item, dict) and 'pred_masks' in item),
        None,
    )
    if plane_dict is None:
        return None, None
    masks  = F.interpolate(plane_dict['pred_masks'], size=(H, W),
                           mode='bilinear', align_corners=True).squeeze(0)
    logits = plane_dict['pred_logits'].squeeze(0)
    return masks, logits


def plane_to_rgb(masks, logits, num_classes):
    """Returns (argmax_rgb, active_rgb) each (H, W, 3) uint8."""
    Q       = masks.shape[0]
    palette = QUERY_PALETTE[:Q]
    winner  = torch.sigmoid(masks).argmax(dim=0).cpu().numpy()
    argmax_rgb = palette[winner]

    is_active = logits.argmax(dim=-1).cpu().numpy() < num_classes
    active_w  = winner.copy()
    for q in range(Q):
        if not is_active[q]:
            active_w[active_w == q] = Q
    active_rgb = palette[np.minimum(active_w, Q - 1)].copy()
    active_rgb[active_w == Q] = NOOBJ_COLOR
    return argmax_rgb, active_rgb


def save_seg(img_rgb, seg_pred, out_root, stem, alpha):
    color   = CS_PALETTE[seg_pred]
    overlay = blend(img_rgb, color, alpha)
    Image.fromarray(color).save(osp.join(out_root, 'seg', 'color',   stem + '.png'))
    Image.fromarray(overlay).save(osp.join(out_root, 'seg', 'overlay', stem + '.png'))


def save_plane(img_rgb, masks, logits, out_root, stem, alpha, num_classes):
    argmax_rgb, active_rgb = plane_to_rgb(masks, logits, num_classes)
    Image.fromarray(argmax_rgb).save(osp.join(out_root, 'plane', 'argmax',  stem + '.png'))
    Image.fromarray(blend(img_rgb, argmax_rgb, alpha)).save(
        osp.join(out_root, 'plane', 'overlay', stem + '.png'))
    Image.fromarray(blend(img_rgb, active_rgb, alpha)).save(
        osp.join(out_root, 'plane', 'active',  stem + '.png'))


def makedirs_for(out_root, heads):
    for head in heads:
        if head == 'seg':
            os.makedirs(osp.join(out_root, 'seg', 'color'),   exist_ok=True)
            os.makedirs(osp.join(out_root, 'seg', 'overlay'), exist_ok=True)
        elif head == 'plane':
            os.makedirs(osp.join(out_root, 'plane', 'argmax'),  exist_ok=True)
            os.makedirs(osp.join(out_root, 'plane', 'overlay'), exist_ok=True)
            os.makedirs(osp.join(out_root, 'plane', 'active'),  exist_ok=True)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seg_weights',    default=DEFAULT_SEG_WEIGHTS,
                   help='Seg-only checkpoint (default: baseline_no_plane)')
    p.add_argument('--plane_weights',  default=DEFAULT_PLANE_WEIGHTS,
                   help='Plane-trained checkpoint (default: plane_setcriterion_w005_delay20k)')
    p.add_argument('--images_dir',     default=DEFAULT_IMAGES_DIR)
    p.add_argument('--out_dir',        default=DEFAULT_OUT_ROOT)
    p.add_argument('--backbone',       default='STDCNet813',
                   choices=['STDCNet813', 'STDCNet1446'])
    p.add_argument('--scale',          type=float, default=0.5)
    p.add_argument('--alpha',          type=float, default=0.5)
    p.add_argument('--num_queries',    type=int, default=20)
    p.add_argument('--num_classes',    type=int, default=20)
    p.add_argument('--hidden_dim',     type=int, default=256)
    p.add_argument('--decoder_layers', type=int, default=10)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seg_ckpt   = resolve_ckpt(args.seg_weights,   args.scale)
    plane_ckpt = resolve_ckpt(args.plane_weights, args.scale)

    seg_run   = osp.basename(args.seg_weights.rstrip('/\\'))
    plane_run = osp.basename(args.plane_weights.rstrip('/\\'))

    seg_out_root   = osp.join(args.out_dir, seg_run)
    plane_out_root = osp.join(args.out_dir, plane_run)

    makedirs_for(seg_out_root,   ['seg'])
    makedirs_for(plane_out_root, ['seg', 'plane'])

    image_paths = sorted(
        glob.glob(osp.join(args.images_dir, '*.png')) +
        glob.glob(osp.join(args.images_dir, '*.jpg')) +
        glob.glob(osp.join(args.images_dir, '*.jpeg'))
    )
    if not image_paths:
        raise FileNotFoundError(f'No images found in {args.images_dir}')

    print(f'\n[independent_heads]')
    print(f'  baseline checkpoint : {seg_ckpt}')
    print(f'  plane checkpoint    : {plane_ckpt}')
    print(f'  images              : {args.images_dir}  ({len(image_paths)} images)')
    print(f'  output root         : {args.out_dir}\n')

    print(f'[loading {seg_run} ...]')
    seg_net = build_seg_only_model(seg_ckpt, args.backbone, device)

    print(f'[loading {plane_run} ...]')
    plane_net = build_plane_model(plane_ckpt, args.backbone, device,
                                  args.num_queries, args.num_classes,
                                  args.hidden_dim, args.decoder_layers)
    print()

    for img_path in image_paths:
        stem    = osp.splitext(osp.basename(img_path))[0]
        img_rgb = np.array(Image.open(img_path).convert('RGB'))

        # ── baseline: seg only ────────────────────────────────────────────────
        out, H, W = run_forward(seg_net, img_rgb, device, args.scale)
        save_seg(img_rgb, extract_seg(out, H, W), seg_out_root, stem, args.alpha)

        # ── plane checkpoint: seg + plane ─────────────────────────────────────
        out, H, W = run_forward(plane_net, img_rgb, device, args.scale)
        save_seg(img_rgb, extract_seg(out, H, W), plane_out_root, stem, args.alpha)
        masks, logits = extract_plane(out, H, W)
        if masks is not None:
            save_plane(img_rgb, masks, logits, plane_out_root, stem, args.alpha, args.num_classes)

        print(f'  [saved] {stem}')

    print(f'\n[done]')
    print(f'  {seg_run}/seg/')
    print(f'  {plane_run}/seg/')
    print(f'  {plane_run}/plane/')


if __name__ == '__main__':
    main()
