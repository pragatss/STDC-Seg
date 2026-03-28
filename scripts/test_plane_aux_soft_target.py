#!/usr/bin/env python3

import argparse
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.model_stages import BiSeNet


def _resolve_path(path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(ROOT, path_value))


def _resolve_model_weights_in_opts(opts):
    if not opts:
        return opts
    resolved = list(opts)
    idx = 0
    while idx + 1 < len(resolved):
        if resolved[idx] == 'MODEL.WEIGHTS':
            resolved[idx + 1] = _resolve_path(resolved[idx + 1])
            break
        idx += 2
    return resolved


def main():
    parser = argparse.ArgumentParser(description='Test only plane_aux_soft_target generation path')
    parser.add_argument('--backbone', default='STDCNet813', choices=['STDCNet813', 'STDCNet1446'])
    parser.add_argument('--n-classes', type=int, default=19)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--print-full', action='store_true', help='Print full soft target tensor')
    parser.add_argument('--config-file', type=str, default='ZeroPlane-ref/configs/ZeroPlaneNYUV2/dust3r_large_dpt_bs16_50ep.yaml')
    parser.add_argument('--opts', nargs='*', default=['MODEL.WEIGHTS', './checkpoints/dust3r_encoder_released.pth'])
    parser.add_argument('--zeroplane-ckpt', type=str, default='', help='Optional explicit ckpt path; overrides MODEL.WEIGHTS in --opts')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')

    device = torch.device(args.device)

    from train import load_zeroplane_soft_target_fn, _extract_model_weights_from_opts

    resolved_config = _resolve_path(args.config_file)
    resolved_opts = _resolve_model_weights_in_opts(args.opts)
    ckpt_from_opts = _extract_model_weights_from_opts(resolved_opts)
    resolved_ckpt = _resolve_path(args.zeroplane_ckpt) if args.zeroplane_ckpt else ckpt_from_opts

    if not resolved_ckpt:
        raise RuntimeError('No checkpoint provided. Pass --zeroplane-ckpt or MODEL.WEIGHTS in --opts.')

    zeroplane_soft_target_fn = load_zeroplane_soft_target_fn(
        ckpt_path=resolved_ckpt,
        config_path=resolved_config,
        config_opts=resolved_opts,
    )

    model = BiSeNet(
        backbone=args.backbone,
        n_classes=args.n_classes,
        use_plane_aux=True,
        plane_aux_soft_target_only_debug=True,
        zeroplane_soft_target_fn=zeroplane_soft_target_fn,
    ).to(device)
    model.eval()

    image = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    with torch.no_grad():
        outputs = model(image)

    feat_out, feat_out16, feat_out32, plane_aux_logits, plane_aux_soft_target = outputs

    print('Main output shape:', tuple(feat_out.shape))
    print('Aux16 output shape:', tuple(feat_out16.shape))
    print('Aux32 output shape:', tuple(feat_out32.shape))
    print('plane_aux_logits is None:', plane_aux_logits is None)

    if plane_aux_soft_target is None:
        raise RuntimeError('plane_aux_soft_target is None; expected a generated tensor')

    print('plane_aux_soft_target shape:', tuple(plane_aux_soft_target.shape))
    print('plane_aux_soft_target dtype:', plane_aux_soft_target.dtype)
    print('plane_aux_soft_target min/max:', float(plane_aux_soft_target.min()), float(plane_aux_soft_target.max()))

    if args.print_full:
        print('plane_aux_soft_target tensor:')
        print(plane_aux_soft_target)

    assert plane_aux_soft_target.shape[-2:] == (args.height, args.width), 'Soft target not resized to input HxW'
    assert float(plane_aux_soft_target.min()) >= 0.0 and float(plane_aux_soft_target.max()) <= 1.0, 'Soft target values not clamped to [0,1]'

    print('PASS: plane_aux_soft_target generation path works as expected')


if __name__ == '__main__':
    main()
