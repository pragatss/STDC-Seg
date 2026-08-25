#!/usr/bin/env python
"""
Clean latency protocol for the equal-latency Pareto comparison
(stock baseline @ several input scales vs I1 / H1 / HI1 @ 0.75).

Uses models.model_stages.BiSeNet -- the SAME class and forward path
evaluation.py scores accuracy with -- not the export-oriented
model_stages_trt.BiSeNet, so latency is measured on the exact graph that
produced the accuracy numbers, including the (small, constant across arms)
overhead of the aux heads that forward() always computes.

Protocol (this is what fixes the earlier contamination bug, where a
loss-only arm that is provably architecturally identical to baseline
measured 5.7ms slower -- a GPU-state artifact, not a real cost):

  1. Every config is built and loaded ONCE, up front, in a fixed order,
     each with its own warm-up (discarded) so cudnn.benchmark autotuning
     and clock ramp-up happen before any timed sample.
  2. Timing then proceeds in R rounds. Each round visits every config
     once, in a FRESH random order, so any drift over the session (GPU
     thermal state, other processes coming and going) gets spread across
     all configs instead of biasing whichever one happens to run first
     or last.
  3. Each timed sample = wall-clock mean over K back-to-back forward
     passes, bracketed by torch.cuda.synchronize() (single-iteration
     timing is dominated by Python/launch overhead, not the kernel).
  4. Report mean +/- std (and min/max) per config over the R round-level
     samples; raw per-round samples are written to CSV for audit.

This does NOT lock GPU clocks -- that needs root, not available in this
environment. If you have sudo, run before this script:
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -lgc <min_clock>,<max_clock>   # e.g. the max from
                                                    # `nvidia-smi -q -d SUPPORTED_CLOCKS`
Without a clock lock, treat absolute ms/FPS as approximate. The
interleaved design still protects the RELATIVE comparisons the Pareto
claim rests on from any one-sided drift, but does not reduce variance
the way a clock lock would -- expect wider std than a locked run.

The script refuses to run on a busy GPU unless --force is passed, since
absolute numbers under contention are not meaningful (this is exactly
the failure mode that produced the earlier 5.7ms artifact).

USAGE
    cd latency
    PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" python measure_pareto_latency.py
Edit CONFIGS below to add/remove checkpoints or scales.
"""
import argparse
import csv
import os
import random
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_stages import BiSeNet

CITYSCAPES_HW = (1024, 2048)  # (H, W) native Cityscapes resolution


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('bad bool: %r' % v)


def scale_hw(scale):
    h, w = CITYSCAPES_HW
    return int(h * scale), int(w * scale)


def gpu_state():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total,clocks.current.sm,clocks.max.sm',
            '--format=csv,noheader,nounits',
        ]).decode().strip()
        util, mem_used, mem_total, clk, clk_max = [x.strip() for x in out.split(',')]
        return dict(util=int(util), mem_used=int(mem_used), mem_total=int(mem_total),
                    clk=int(clk), clk_max=int(clk_max))
    except Exception as e:
        print('  [warn] could not read nvidia-smi state: %s' % e)
        return None


def build_model(backbone, n_classes, use_boundary_8, use_brh, brh_mid, ckpt):
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
                  use_boundary_2=False, use_boundary_4=False,
                  use_boundary_8=use_boundary_8, use_boundary_16=False,
                  use_conv_last=False, use_brh=use_brh, brh_mid=brh_mid)
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'checkpoint %s did not load cleanly (missing=%d unexpected=%d) -- '
            'use_brh probably does not match how it was trained' %
            (ckpt, len(missing), len(unexpected)))
    return net.cuda().eval()


def time_block(model, x, k):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(k):
            model(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / k * 1000.0  # ms/iter


# name -> (checkpoint, use_brh, scale). Baseline architecture is timed at
# each candidate scale to build the "stock, just upscale" curve; the
# checkpoint weights used for baseline latency are arbitrary (weights don't
# affect forward-pass latency, only architecture + input shape do) so any
# baseline checkpoint works here -- the *accuracy* sweep is what needs both
# baseline replicates, not this one.
CONFIGS = [
    dict(label='baseline@0.75', ckpt='../checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth', use_brh=False, scale=0.75),
    dict(label='baseline@0.80', ckpt='../checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth', use_brh=False, scale=0.80),
    dict(label='baseline@0.85', ckpt='../checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth', use_brh=False, scale=0.85),
    dict(label='baseline@0.90', ckpt='../checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth', use_brh=False, scale=0.90),
    dict(label='I1@0.75',       ckpt='../checkpoints/train_STDC2-Seg-I1/pths/model_maxmIOU75.pth',        use_brh=False, scale=0.75),
    dict(label='H1@0.75',       ckpt='../checkpoints/train_STDC2-Seg-H1/pths/model_maxmIOU75.pth',        use_brh=True,  scale=0.75),
    dict(label='HI1@0.75',      ckpt='../checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU75.pth',       use_brh=True,  scale=0.75),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='STDCNet1446')
    p.add_argument('--n_classes', type=int, default=19)
    p.add_argument('--use_boundary_8', type=str2bool, default=True)
    p.add_argument('--brh_mid', type=int, default=64)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--rounds', type=int, default=20, help='R: interleaved passes over all configs')
    p.add_argument('--iters_per_sample', type=int, default=10, help='K: fwd passes averaged per sample')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='pareto_latency.csv')
    p.add_argument('--force', action='store_true', help='proceed even if the GPU looks busy')
    args = p.parse_args()

    state = gpu_state()
    if state is not None:
        print('GPU state before run: util=%d%% mem=%d/%dMiB clock=%d/%dMHz' %
              (state['util'], state['mem_used'], state['mem_total'], state['clk'], state['clk_max']))
        if state['util'] > 20 or state['mem_used'] > 0.3 * state['mem_total']:
            msg = ('GPU looks busy (another process is likely running). Absolute latency '
                   'numbers will not be trustworthy under contention.')
            if not args.force:
                raise SystemExit(msg + ' Re-run with --force to proceed anyway.')
            print('  [warn] ' + msg + ' Proceeding because --force was passed.')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    configs = [dict(c) for c in CONFIGS]
    print('\nBuilding %d model configs and warming each up (%d iters, discarded)...' %
          (len(configs), args.warmup))
    for cfg in configs:
        h, w = scale_hw(cfg['scale'])
        cfg['hw'] = (h, w)
        cfg['model'] = build_model(args.backbone, args.n_classes, args.use_boundary_8,
                                    cfg['use_brh'], args.brh_mid, cfg['ckpt'])
        cfg['input'] = torch.randn(1, 3, h, w, device='cuda')
        time_block(cfg['model'], cfg['input'], args.warmup)  # warm-up, discarded
        print('  %-14s  input=%dx%d  ckpt=%s' % (cfg['label'], h, w, cfg['ckpt']))

    rng = random.Random(args.seed)
    samples = {cfg['label']: [] for cfg in configs}

    print('\nRunning %d interleaved rounds (%d fwd passes/sample, random order each round)...' %
          (args.rounds, args.iters_per_sample))
    raw_rows = []
    for r in range(args.rounds):
        order = configs[:]
        rng.shuffle(order)
        for cfg in order:
            ms = time_block(cfg['model'], cfg['input'], args.iters_per_sample)
            samples[cfg['label']].append(ms)
            raw_rows.append(dict(round=r, label=cfg['label'], ms=ms))
        print('  round %2d/%d done' % (r + 1, args.rounds))

    state_after = gpu_state()
    if state_after is not None:
        print('GPU state after run: util=%d%% mem=%d/%dMiB clock=%d/%dMHz' %
              (state_after['util'], state_after['mem_used'], state_after['mem_total'],
               state_after['clk'], state_after['clk_max']))

    print('\n%-14s %10s %10s %10s %10s %6s' % ('label', 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'FPS'))
    rows = []
    for cfg in configs:
        xs = samples[cfg['label']]
        mean = statistics.mean(xs)
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print('%-14s %10.3f %10.3f %10.3f %10.3f %6.2f' %
              (cfg['label'], mean, std, min(xs), max(xs), 1000.0 / mean))
        rows.append(dict(label=cfg['label'], scale=cfg['scale'], use_brh=cfg['use_brh'],
                          h=cfg['hw'][0], w=cfg['hw'][1], mean_ms=mean, std_ms=std,
                          min_ms=min(xs), max_ms=max(xs), n_samples=len(xs)))

    with open(args.out, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    raw_path = os.path.splitext(args.out)[0] + '_raw.csv'
    with open(raw_path, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['round', 'label', 'ms'])
        wtr.writeheader()
        wtr.writerows(raw_rows)
    print('\nWrote %s and %s' % (args.out, raw_path))


if __name__ == '__main__':
    main()
