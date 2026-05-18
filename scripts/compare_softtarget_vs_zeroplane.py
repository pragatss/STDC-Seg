#!/usr/bin/env python3
"""
For each image in scripts/images/ :
    1. Run ZeroPlane to get a soft-target (21, H, W)
    2. Render the live prediction with the exact demo-style argmax + blend logic
    3. Downsample and save a training-style soft target
    4. Render that stored soft target back into a demo-style plane overlay
        5. Save a combined comparison to scripts/combined/

            [RGB Input] | [Live ZeroPlane render] | [Stored soft-target render]
            | [Soft-target only] | [Disagreement]

Usage (from repo root):
    conda run -n stdcseg18 python scripts/compare_softtarget_vs_zeroplane.py
"""

import os, sys, glob, csv
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

# ── paths (hardcoded) ────────────────────────────────────────────────────────
ROOT        = osp.abspath(osp.join(osp.dirname(__file__), '..'))
IMAGES_DIR  = osp.join(ROOT, 'scripts', 'images')
OUT_DIR     = osp.join(ROOT, 'scripts', 'combined')
SOFT_DIR    = osp.join(ROOT, 'scripts', 'generated_softtargets')
CONFIG      = osp.join(ROOT, 'ZeroPlane', 'configs', 'ZeroPlaneNYUV2',
                       'dust3r_large_dpt_bs16_50ep.yaml')
WEIGHTS     = osp.join(ROOT, 'checkpoints', 'dust3r_encoder_released.pth')
# Match the public ZeroPlane demo path exactly.
INFER_H, INFER_W = 192, 256
ORIG_H, ORIG_W = 480, 640
ST_H, ST_W = 128, 256
PANEL_H, PANEL_W = 256, 512          # display size for each panel

# ── soft-target constants (double-sigmoid range) ─────────────────────────────
S0, S1 = 0.5, 0.7310585786

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SOFT_DIR, exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── ZeroPlane setup ──────────────────────────────────────────────────────────
import torch

def _build_predictor(device):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from ZeroPlane.ZeroPlane import add_ZeroPlane_config
    from ZeroPlane.demo.predictor import DefaultPredictor
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ZeroPlane_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.merge_from_list(['MODEL.WEIGHTS', WEIGHTS])
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return DefaultPredictor(cfg)

def _load_anchors(device):
    zp = osp.join(ROOT, 'ZeroPlane')
    normals = np.load(osp.join(zp, 'cluster_anchor', 'new_mixed_normal_anchors_7.npy'))
    offsets = np.load(osp.join(zp, 'cluster_anchor', 'new_mixed_offset_anchors_20.npy'))
    return torch.tensor(normals).to(device), torch.tensor(offsets).to(device)

def _coord_map(h, w, oh, ow, device):
    K = np.array([[518.86, 0, 325.58], [0, 519.47, 253.74], [0, 0, 1]], np.float32)
    K_inv = torch.FloatTensor(np.linalg.inv(K)).to(device)
    xx = (torch.arange(w, dtype=torch.float32).view(1, w) / w * ow).repeat(h, 1).to(device)
    yy = (torch.arange(h, dtype=torch.float32).view(h, 1) / h * oh).repeat(1, w).to(device)
    xy1 = torch.stack([xx, yy, torch.ones(h, w, dtype=torch.float32, device=device)])
    return torch.matmul(K_inv, xy1.view(3, -1))

def run_zeroplane(predictor, img_np, normals, offsets, device):
    """img_np: uint8 HWC RGB → returns (21, H, W) float32 CPU."""
    h, w = img_np.shape[:2]
    coord = _coord_map(h, w, ORIG_H, ORIG_W, device)
    img_bgr = img_np[:, :, ::-1].copy()
    with torch.no_grad():
        pred = predictor(img_bgr, {'anchor_normals': normals, 'anchor_offsets': offsets}, coord)
    sem = pred['sem_seg']
    return sem.float().clamp(0, 1).cpu().numpy()   # (21, H, W)

# ── visualisation helpers ────────────────────────────────────────────────────

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


def plane_instance_vis(argmax, rgb_np, num_queries=20):
    """Exact ZeroPlane demo rendering from utils/disp.py.
    argmax: (H, W) int32 — winner query index per pixel (0-19=plane, 20=non-plane)."""
    segmentation = argmax.copy().astype(np.int32)
    segmentation += 1
    segmentation[segmentation >= (num_queries + 1)] = 0

    colors = labelcolormap(256)
    seg = np.stack([
        colors[segmentation, 0],
        colors[segmentation, 1],
        colors[segmentation, 2],
    ], axis=2)

    blend_seg = (seg * 0.7 + rgb_np.astype(np.float32) * 0.3).astype(np.uint8)
    seg_mask = (segmentation > 0).astype(np.uint8)[:, :, np.newaxis]
    blend_seg = blend_seg * seg_mask + rgb_np.astype(np.uint8) * (1 - seg_mask)

    n_plane_queries = len(np.unique(argmax[argmax < num_queries]))
    return blend_seg.astype(np.uint8), n_plane_queries


def compare_argmax_quality(argmax_ref, argmax_soft, num_queries=20):
    """Compare stored soft-target argmax against a reference argmax at the same resolution."""
    same_query = (argmax_ref == argmax_soft)
    ref_plane = argmax_ref < num_queries
    soft_plane = argmax_soft < num_queries
    same_plane_nonplane = (ref_plane == soft_plane)
    both_plane = ref_plane & soft_plane
    plane_union = ref_plane | soft_plane

    return {
        'pixel_acc': float(same_query.mean()),
        'plane_nonplane_acc': float(same_plane_nonplane.mean()),
        'both_plane_frac': float(both_plane.mean()),
        'plane_union_frac': float(plane_union.mean()),
        'plane_query_acc_on_both_plane': float(same_query[both_plane].mean()) if both_plane.any() else 1.0,
        'changed_any_frac': float((~same_query).mean()),
        'changed_within_plane_frac': float(((~same_query) & plane_union).mean()),
    }


def disagreement_vis(argmax_ref, argmax_soft):
    """Red = mismatch, dark = exact match."""
    mismatch = argmax_ref != argmax_soft
    canvas = np.zeros((argmax_ref.shape[0], argmax_ref.shape[1], 3), dtype=np.uint8)
    canvas[mismatch] = np.array([220, 40, 40], dtype=np.uint8)
    canvas[~mismatch] = np.array([25, 25, 25], dtype=np.uint8)
    return canvas


def softtarget_only_vis(argmax_map, num_queries=20):
    """Plane queries get demo colors, non-plane stays black."""
    colors = labelcolormap(256)
    canvas = np.zeros((argmax_map.shape[0], argmax_map.shape[1], 3), dtype=np.uint8)
    plane_mask = argmax_map < num_queries
    if plane_mask.any():
        plane_ids = argmax_map.copy().astype(np.int32)
        plane_ids[~plane_mask] = 0
        plane_colors = np.stack([
            colors[plane_ids, 0],
            colors[plane_ids, 1],
            colors[plane_ids, 2],
        ], axis=2)
        canvas[plane_mask] = plane_colors[plane_mask]
    return canvas


def resize_rgb(img_np, h, w):
    return np.array(Image.fromarray(img_np).resize((w, h), Image.BILINEAR))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print('Loading ZeroPlane predictor ...')
    predictor = _build_predictor(device)
    normals, offsets = _load_anchors(device)

    image_paths = sorted(glob.glob(osp.join(IMAGES_DIR, '*.png')) +
                         glob.glob(osp.join(IMAGES_DIR, '*.jpg')))
    summary_rows = []

    for img_path in image_paths:
        print(f'Processing {osp.basename(img_path)} ...')

        # ── load & resize RGB exactly like demo.py ─────────────────────────
        img_pil = Image.open(img_path).convert('RGB')
        img_infer = np.array(img_pil.resize((INFER_W, INFER_H), Image.BILINEAR), dtype=np.uint8)

        # ── run ZeroPlane ──────────────────────────────────────────────────
        sem_np = run_zeroplane(predictor, img_infer, normals, offsets, device)
        # sem_np: (21, INFER_H, INFER_W)

        # ── panel A: RGB input ─────────────────────────────────────────────
        panel_rgb = resize_rgb(img_infer, PANEL_H, PANEL_W)

        # ── panel B: live ZeroPlane render from full-resolution argmax ─────
        argmax_live = sem_np.argmax(axis=0).astype(np.int32)
        argmax_live_panel = np.array(
            Image.fromarray(argmax_live.astype(np.uint8)).resize((PANEL_W, PANEL_H), Image.NEAREST)
        ).astype(np.int32)
        panel_blend, n_live = plane_instance_vis(
            argmax_live_panel,
            resize_rgb(img_infer, PANEL_H, PANEL_W)
        )

        # ── panel C: stored soft-target render after training-style downsample ──
        sem_small = F.interpolate(
            torch.from_numpy(sem_np).unsqueeze(0),
            size=(ST_H, ST_W),
            mode='bilinear',
            align_corners=True,
        ).squeeze(0).cpu().numpy().astype(np.float16)

        soft_name = osp.splitext(osp.basename(img_path))[0] + '.npy'
        soft_path = osp.join(SOFT_DIR, soft_name)
        np.save(soft_path, sem_small)

        argmax_live_ref_small = np.array(
            Image.fromarray(argmax_live.astype(np.uint8)).resize((ST_W, ST_H), Image.NEAREST)
        ).astype(np.int32)
        argmax_soft = sem_small.argmax(axis=0).astype(np.int32)
        metrics = compare_argmax_quality(argmax_live_ref_small, argmax_soft)
        argmax_soft_panel = np.array(
            Image.fromarray(argmax_soft.astype(np.uint8)).resize((PANEL_W, PANEL_H), Image.NEAREST)
        ).astype(np.int32)
        softtarget_only_panel = np.array(
            Image.fromarray(softtarget_only_vis(argmax_soft)).resize((PANEL_W, PANEL_H), Image.NEAREST)
        )
        disagreement_panel = np.array(
            Image.fromarray(disagreement_vis(argmax_live_ref_small, argmax_soft)).resize(
                (PANEL_W, PANEL_H), Image.NEAREST
            )
        )
        instance_vis, n_inst = plane_instance_vis(
            argmax_soft_panel,
            resize_rgb(img_infer, PANEL_H, PANEL_W)
        )

        mosaic = np.hstack([
            title_bar(panel_rgb,    'RGB Input'),
            title_bar(panel_blend,  f'Live ZeroPlane demo-style render (queries={n_live})'),
            title_bar(instance_vis, f'Stored soft-target render {ST_H}x{ST_W} (queries={n_inst})'),
            title_bar(softtarget_only_panel, 'Soft-target only (non-plane black)'),
            title_bar(disagreement_panel,
                      'Disagreement red | px={:.1f}% plane/non-plane={:.1f}% plane-query={:.1f}%'.format(
                          100.0 * metrics['pixel_acc'],
                          100.0 * metrics['plane_nonplane_acc'],
                          100.0 * metrics['plane_query_acc_on_both_plane'])),
        ])

        stem = osp.splitext(osp.basename(img_path))[0]
        out_path = osp.join(OUT_DIR, f'{stem}_comparison.png')
        Image.fromarray(mosaic).save(out_path)
        print(f'  → saved {out_path}')
        print(f'  → saved {soft_path}')
        print('  → quality: pixel_acc={:.4f} plane_nonplane_acc={:.4f} plane_query_acc_on_both_plane={:.4f} changed_any={:.4f}'.format(
            metrics['pixel_acc'],
            metrics['plane_nonplane_acc'],
            metrics['plane_query_acc_on_both_plane'],
            metrics['changed_any_frac']))

        summary_rows.append({
            'image': osp.basename(img_path),
            'soft_path': soft_name,
            'pixel_acc': metrics['pixel_acc'],
            'plane_nonplane_acc': metrics['plane_nonplane_acc'],
            'plane_query_acc_on_both_plane': metrics['plane_query_acc_on_both_plane'],
            'both_plane_frac': metrics['both_plane_frac'],
            'plane_union_frac': metrics['plane_union_frac'],
            'changed_any_frac': metrics['changed_any_frac'],
            'changed_within_plane_frac': metrics['changed_within_plane_frac'],
        })

    summary_path = osp.join(OUT_DIR, 'quality_summary.csv')
    if summary_rows:
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\nSaved quality summary to: {summary_path}')

    print(f'\nDone. {len(image_paths)} images processed.')


if __name__ == '__main__':
    main()
