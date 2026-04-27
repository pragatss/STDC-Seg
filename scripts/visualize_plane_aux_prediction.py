#!/usr/bin/env python3
"""
Visualize ZeroPlane soft-target predictions on a real RGB image.

Produces a side-by-side figure:
  [Input Image] | [Argmax Class Map] | [Top-class Confidence Heatmap]

Usage:
    python scripts/visualize_plane_aux_prediction.py \
        --image path/to/image.png \
        --config ZeroPlane/configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml \
        --opts MODEL.WEIGHTS ./checkpoints/dust3r_encoder_released.pth \
        --out scripts/viz_out.png

The script reuses the same ZeroPlane inference path as precompute_soft_targets.py
so what you see here is exactly what gets stored in the soft_targets/ directory
during training.
"""

import argparse
import os
import os.path as osp
import sys

# ── If --device cpu is requested, hide all GPUs before any CUDA-aware
#    library (torch, detectron2) is imported.  This prevents detectron2's
#    internal preprocessing from silently sending tensors to CUDA while the
#    model weights sit on CPU.
if '--device' in sys.argv:
    _dev_idx = sys.argv.index('--device')
    if _dev_idx + 1 < len(sys.argv) and sys.argv[_dev_idx + 1] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from PIL import Image
import torch

# ── repo root on sys.path ────────────────────────────────────────────────────
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Cityscapes 19-class palette (+ 2 extra slots for ZeroPlane's 21 channels)
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
    [  0, 200,   0],   # 19 <extra>
    [200,   0, 200],   # 20 <extra>
], dtype=np.uint8)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle', '<extra-19>', '<extra-20>',
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_predictor(config_path, opts, device):
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


def _load_anchors(device):
    zp_root = osp.join(ROOT, 'ZeroPlane')
    normals = np.load(osp.join(zp_root, 'cluster_anchor', 'new_mixed_normal_anchors_7.npy'))
    offsets = np.load(osp.join(zp_root, 'cluster_anchor', 'new_mixed_offset_anchors_20.npy'))
    return torch.tensor(normals).to(device), torch.tensor(offsets).to(device)


def _get_coordinate_map(h, w, device):
    K = np.asarray([[518.86, 0, 325.58],
                    [0, 519.47, 253.74],
                    [0, 0, 1]], dtype=np.float32)
    K_inv = torch.FloatTensor(np.linalg.inv(K)).to(device)
    x = torch.arange(w, dtype=torch.float32).view(1, w)
    y = torch.arange(h, dtype=torch.float32).view(h, 1)
    xx = x.repeat(h, 1).to(device)
    yy = y.repeat(1, w).to(device)
    xy1 = torch.stack((xx, yy, torch.ones(h, w, dtype=torch.float32, device=device)))
    return torch.matmul(K_inv, xy1.view(3, -1))


def run_zeroplane(predictor, img_np, anchor_normals, anchor_offsets, device):
    """img_np: uint8 HWC RGB.  Returns (21, H, W) float32 CPU tensor."""
    h, w = img_np.shape[:2]
    k_inv_dot = _get_coordinate_map(h, w, device)
    img_bgr = img_np[:, :, ::-1].copy()
    anchors = {'anchor_normals': anchor_normals, 'anchor_offsets': anchor_offsets}
    with torch.no_grad():
        pred = predictor(img_bgr, anchors, k_inv_dot)
    if not isinstance(pred, dict):
        raise RuntimeError('ZeroPlane returned unexpected type: {}'.format(type(pred)))
    sem_seg = pred.get('sem_seg')
    if sem_seg is None:
        raise RuntimeError('ZeroPlane output has no sem_seg key')
    return torch.clamp(sem_seg.float(), 0.0, 1.0).cpu()


def colorize_argmax(sem_seg):
    """sem_seg: (C, H, W) float.  Returns (H, W, 3) uint8 RGB."""
    argmax = sem_seg.argmax(dim=0).numpy()          # (H, W)
    rgb = CITYSCAPES_PALETTE[argmax % len(CITYSCAPES_PALETTE)]
    return rgb


def confidence_heatmap(sem_seg):
    """Returns max-probability heatmap as (H, W, 3) uint8 RGB (viridis-like)."""
    conf = sem_seg.max(dim=0).values.numpy()        # (H, W)  in [0, 1]
    conf_uint8 = (conf * 255).clip(0, 255).astype(np.uint8)
    try:
        import cv2  # noqa: F401
        heatmap_bgr = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_VIRIDIS)
        return heatmap_bgr[:, :, ::-1]              # BGR → RGB
    except ImportError:
        # Fallback: greyscale repeated to 3 channels
        grey = np.stack([conf_uint8] * 3, axis=-1)
        return grey


def add_legend(img_rgb, sem_seg, top_k=5):
    """Burn a small class-legend (top-k by mean probability) onto img_rgb."""
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return img_rgb

    # top-k classes by average probability across the image
    mean_probs = sem_seg.mean(dim=(1, 2)).numpy()   # (C,)
    top_indices = mean_probs.argsort()[::-1][:top_k]

    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    y = 8
    for idx in top_indices:
        color = tuple(CITYSCAPES_PALETTE[idx % len(CITYSCAPES_PALETTE)].tolist())
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        label = '{}: {:.3f}'.format(name, float(mean_probs[idx]))
        # shadow
        draw.text((11, y + 1), label, fill=(0, 0, 0), font=font)
        draw.text((10, y), label, fill=color, font=font)
        y += 18

    return np.array(pil)


def make_panel(title, img_rgb, font=None):
    """Add a title bar above an RGB numpy image."""
    try:
        from PIL import ImageDraw, ImageFont
        pil = Image.fromarray(img_rgb)
        bar = Image.new('RGB', (pil.width, 22), (30, 30, 30))
        draw = ImageDraw.Draw(bar)
        if font is None:
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
            except Exception:
                font = ImageFont.load_default()
        draw.text((4, 4), title, fill=(220, 220, 220), font=font)
        return np.vstack([np.array(bar), img_rgb])
    except ImportError:
        return img_rgb


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Visualize ZeroPlane soft-target prediction on a real image')
    p.add_argument('--image', required=True, help='Path to input RGB image')
    p.add_argument('--config', default='ZeroPlane/configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml')
    p.add_argument('--opts', nargs='*', default=['MODEL.WEIGHTS', './checkpoints/dust3r_encoder_released.pth'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--infer-height', type=int, default=512)
    p.add_argument('--infer-width', type=int, default=1024)
    p.add_argument('--out', default='scripts/viz_plane_aux.png', help='Output image path')
    p.add_argument('--no-display', action='store_true', help='Skip plt.show(); just save')
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve relative paths from repo root ────────────────────────────────
    config_path = args.config if osp.isabs(args.config) else osp.join(ROOT, args.config)
    opts = list(args.opts) if args.opts else []
    for i, o in enumerate(opts):
        if opts[i - 1] == 'MODEL.WEIGHTS' and not osp.isabs(o):
            opts[i] = osp.join(ROOT, o)

    device = args.device

    # ── load image ───────────────────────────────────────────────────────────
    img_pil = Image.open(args.image).convert('RGB')
    orig_w, orig_h = img_pil.size
    print('Input image: {} ({}×{})'.format(args.image, orig_w, orig_h))

    img_resized = img_pil.resize((args.infer_width, args.infer_height), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.uint8)

    # ── run ZeroPlane ─────────────────────────────────────────────────────────
    print('Loading ZeroPlane predictor from {} ...'.format(config_path))
    predictor = _build_predictor(config_path, opts, device)
    anchor_normals, anchor_offsets = _load_anchors(device)
    print('Running inference ...')
    sem_seg = run_zeroplane(predictor, img_np, anchor_normals, anchor_offsets, device)
    # sem_seg: (21, infer_H, infer_W) float32 CPU

    print('sem_seg shape:', tuple(sem_seg.shape))
    print('sem_seg min/max: {:.4f} / {:.4f}'.format(float(sem_seg.min()), float(sem_seg.max())))

    argmax_ids = sem_seg.argmax(dim=0).numpy()
    unique, counts = np.unique(argmax_ids, return_counts=True)
    print('Dominant classes (argmax):')
    for uid, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
        name = CLASS_NAMES[uid] if uid < len(CLASS_NAMES) else str(uid)
        print('  class {:2d} ({:15s}): {:.1f}% of pixels'.format(
            uid, name, 100.0 * cnt / argmax_ids.size))

    # ── build visualisation panels ───────────────────────────────────────────
    H, W = args.infer_height, args.infer_width

    panel_input   = make_panel('Input Image',            img_np)
    panel_argmax  = make_panel('Argmax Class Map',       add_legend(colorize_argmax(sem_seg), sem_seg))
    panel_conf    = make_panel('Top-class Confidence',   confidence_heatmap(sem_seg))

    # Ensure all panels are the same height (they should be but just in case)
    ph = max(p.shape[0] for p in [panel_input, panel_argmax, panel_conf])
    def pad_height(img, target_h):
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    mosaic = np.hstack([
        pad_height(panel_input,  ph),
        pad_height(panel_argmax, ph),
        pad_height(panel_conf,   ph),
    ])

    out_path = args.out if osp.isabs(args.out) else osp.join(ROOT, args.out)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    Image.fromarray(mosaic).save(out_path)
    print('Saved visualisation to: {}'.format(out_path))

    if not args.no_display:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 5))
            plt.imshow(mosaic)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print('(matplotlib display skipped: {})'.format(e))


if __name__ == '__main__':
    main()
