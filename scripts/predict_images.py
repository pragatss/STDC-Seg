#!/usr/bin/env python3
"""
Run STDC semantic segmentation inference on all images in scripts/images and save
colorized predictions plus RGB overlays.

Examples:
    conda run -n stdcseg18 --no-capture-output python scripts/predict_images.py \
        --weights checkpoints/train_STDC1-Seg/baseline_no_plane

    conda run -n stdcseg18 --no-capture-output python scripts/predict_images.py \
        --weights checkpoints/train_STDC1-Seg/plane_setcriterion_w002_delay20k \
        --out_dir predictions/plane_setcriterion_w002_delay20k
"""

import argparse
import glob
import os
import os.path as osp

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.model_stages import BiSeNet

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DEFAULT_IMAGES_DIR = osp.join(ROOT, 'scripts', 'images')
DEFAULT_PREDICTIONS_DIR = osp.join(ROOT, 'predictions')


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


def infer_run_name(weights_path):
    abs_path = osp.abspath(weights_path)
    parts = abs_path.split(os.sep)
    if 'pths' in parts:
        idx = parts.index('pths')
        if idx > 0:
            return parts[idx - 1]
    return osp.splitext(osp.basename(abs_path))[0]


def build_model(weights_path, backbone, device):
    net = BiSeNet(
        backbone=backbone,
        n_classes=19,
        use_boundary_2=False,
        use_boundary_4=False,
        use_boundary_8=False,
        use_boundary_16=False,
    )
    state = torch.load(weights_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    cleaned = {}
    for key, value in state.items():
        new_key = key[7:] if key.startswith('module.') else key
        cleaned[new_key] = value
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        print('[load] missing keys:', len(missing))
    if unexpected:
        print('[load] unexpected keys:', len(unexpected))
    net.to(device)
    net.eval()
    return net


def preprocess_image(img_rgb):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return to_tensor(Image.fromarray(img_rgb)).unsqueeze(0)


def predict_mask(net, img_rgb, device, scale):
    x = preprocess_image(img_rgb).to(device)
    with torch.no_grad():
        h, w = x.shape[-2:]
        sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
        x_scaled = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=True)
        logits = net(x_scaled)[0]
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def blend_rgb(img_rgb, seg_rgb, alpha=0.5):
    img_f = img_rgb.astype(np.float32)
    seg_f = seg_rgb.astype(np.float32)
    return np.clip((1.0 - alpha) * img_f + alpha * seg_f, 0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='Run STDC predictions on a folder of RGB images.')
    parser.add_argument('--weights', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--images_dir', default=DEFAULT_IMAGES_DIR, help='Directory with input .png/.jpg images')
    parser.add_argument('--out_dir', default='', help='Output directory; default is predictions/<run_name>')
    parser.add_argument('--backbone', default='STDCNet813', choices=['STDCNet813', 'STDCNet1446'])
    parser.add_argument('--scale', type=float, default=0.5, help='Inference scale before upsampling back')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay opacity')
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = resolve_checkpoint_path(args.weights, args.scale)
    images_dir = osp.abspath(args.images_dir)
    run_name = infer_run_name(weights_path)
    out_dir = osp.abspath(args.out_dir) if args.out_dir else osp.join(DEFAULT_PREDICTIONS_DIR, run_name)
    color_dir = osp.join(out_dir, 'color')
    overlay_dir = osp.join(out_dir, 'overlay')
    label_dir = osp.join(out_dir, 'label_id')

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(osp.join(images_dir, '*.png')) +
        glob.glob(osp.join(images_dir, '*.jpg')) +
        glob.glob(osp.join(images_dir, '*.jpeg'))
    )
    if not image_paths:
        raise FileNotFoundError('No images found in {}'.format(images_dir))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    palette = cityscapes_palette()

    print('[predict] weights =', weights_path)
    print('[predict] images  =', images_dir)
    print('[predict] out_dir =', out_dir)
    print('[predict] device  =', device)
    net = build_model(weights_path, args.backbone, device)

    for img_path in image_paths:
        stem = osp.splitext(osp.basename(img_path))[0]
        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        pred = predict_mask(net, img_rgb, device, args.scale)
        pred_rgb = palette[pred]
        overlay = blend_rgb(img_rgb, pred_rgb, alpha=args.alpha)

        Image.fromarray(pred.astype(np.uint8)).save(osp.join(label_dir, stem + '.png'))
        Image.fromarray(pred_rgb).save(osp.join(color_dir, stem + '.png'))
        Image.fromarray(overlay).save(osp.join(overlay_dir, stem + '.png'))
        print('[saved]', stem)

    print('[done] predictions saved to', out_dir)


if __name__ == '__main__':
    main()
