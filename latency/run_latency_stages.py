from __future__ import division

import os
import sys
import argparse
import torch
import numpy as np

sys.path.append("../")

from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
from models.model_stages_trt import BiSeNet

INPUT_DIMS = {
    512: (1, 3, 512, 1024),
    768: (1, 3, 768, 1536),
    1024: (1, 3, 1024, 2048),
    720: (1, 3, 720, 960),
}

# name -> (checkpoint dir under ../checkpoints, use_brh) for the current,
# architecture-compatible experiment set documented in commands.txt.
# (ARM-*/GCN/ARMB checkpoints use older module names no longer in
# models/model_stages*.py and can't be loaded here.)
SWEEP = [
    ('Baseline2', 'train_STDC2-Seg-Baseline2', False),
    ('I0 (bnd_weight=1)', 'train_STDC2-Seg-I0', False),
    ('I1 (bnd_weight=3)', 'train_STDC2-Seg-I1', False),
    ('H1 (brh)', 'train_STDC2-Seg-H1', True),
    ('HI1 (brh + bnd_weight=3)', 'train_STDC2-Seg-HI1', True),
]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=None,
                    help='path to a .pth file, or a checkpoint dir containing pths/model_maxmIOU<scale>.pth')
    p.add_argument('--backbone', type=str, default='STDCNet1446')
    p.add_argument('--n_classes', type=int, default=19)
    p.add_argument('--use_brh', type=str2bool, default=False)
    p.add_argument('--brh_mid', type=int, default=64)
    p.add_argument('--input_size', type=int, choices=sorted(INPUT_DIMS), default=512)
    p.add_argument('--miou_scale', type=int, choices=(50, 75), default=50,
                    help='which model_maxmIOU<scale>.pth to pick when --ckpt is a checkpoint dir '
                         '(this is the val-time eval scale used to pick the best checkpoint during '
                         'training, unrelated to --input_size, the deployment resolution used here)')
    p.add_argument('--sweep', action='store_true',
                    help='ignore --ckpt/--use_brh and benchmark the known current checkpoint set')
    return p.parse_args()


def resolve_ckpt(path, miou_scale):
    if os.path.isfile(path):
        return path
    cand = os.path.join(path, 'pths', 'model_maxmIOU{}.pth'.format(miou_scale))
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(path, 'pths', 'model_final.pth')
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError('no checkpoint found under {}'.format(path))


def run_one(ckpt_path, backbone, n_classes, use_brh, brh_mid, input_size, miou_scale):
    ckpt_path = resolve_ckpt(ckpt_path, miou_scale)
    model = BiSeNet(backbone=backbone, n_classes=n_classes, use_boundary_8=True,
                     input_size=input_size, use_brh=use_brh, brh_mid=brh_mid)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.cuda()

    latency = compute_latency(model, INPUT_DIMS[input_size])
    fps = 1000. / latency
    print('{}: {:.2f} FPS ({:.3f} ms)'.format(ckpt_path, fps, latency))
    return fps, latency


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if args.sweep:
        results = []
        for label, ckpt_dir, use_brh in SWEEP:
            ckpt_path = os.path.join('../checkpoints', ckpt_dir)
            fps, latency = run_one(ckpt_path, args.backbone, args.n_classes,
                                    use_brh, args.brh_mid, args.input_size, args.miou_scale)
            results.append((label, fps, latency))

        print('\n==== latency sweep @ input_size={} ===='.format(args.input_size))
        for label, fps, latency in results:
            print('{:28s} {:8.2f} FPS  {:8.3f} ms'.format(label, fps, latency))
        return

    if args.ckpt is None:
        raise SystemExit('--ckpt is required unless --sweep is passed')
    run_one(args.ckpt, args.backbone, args.n_classes, args.use_brh, args.brh_mid,
            args.input_size, args.miou_scale)


if __name__ == '__main__':
    main()

# ckbone:  STDCNet1446
# =========Speed Testing=========
# 100%|████████████████████████████████████████████████████████| 180/180 [00:12<00:00, 14.74it/s]
# ../checkpoints/train_STDC2-Seg-H1/pths/model_maxmIOU50.pth: 14.52 FPS (68.877 ms)
# BiSeNet backbone:  STDCNet1446
# backbone:  STDCNet1446
# =========Speed Testing=========
# 100%|████████████████████████████████████████████████████████| 181/181 [00:12<00:00, 14.63it/s]
# ../checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU50.pth: 14.41 FPS (69.374 ms)

# ==== latency sweep @ input_size=512 ====
# Baseline2                       17.09 FPS    58.522 ms
# I0 (bnd_weight=1)               15.42 FPS    64.832 ms
# I1 (bnd_weight=3)               15.72 FPS    63.624 ms
# H1 (brh)                        14.52 FPS    68.877 ms
# HI1 (brh + bnd_weight=3)        14.41 FPS    69.374 ms