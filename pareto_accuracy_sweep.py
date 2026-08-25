#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Accuracy half of the equal-latency Pareto comparison: re-evaluate the stock
baseline checkpoint(s) at several input scales to get the (scale -> boundary
IoU) curve that measure_pareto_latency.py's baseline@<scale> rows supply the
latency half of.

Uses BOTH on-disk baseline replicates (Baseline, Baseline2) at each scale,
so each point on the curve carries n=2 (not n=1) -- still short of the n=3
used for the reported baseline-mean numbers at scale=0.75 (that third
replicate isn't on disk in this checkout), but better than treating a single
checkpoint's eval as if it had no replicate noise. Add a third baseline
checkpoint path below once one is available and re-run.

Deterministic: CityScapes val mode has no augmentation and the DataLoader
uses shuffle=False, so re-running this script on the same checkpoint/scale
reproduces the same numbers -- unlike latency, there's no need for repeated
timing samples here, just one pass per (checkpoint, scale).

USAGE
    PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" python pareto_accuracy_sweep.py
"""
import csv

from evaluation import report

BASELINES = [
    ('B1', './checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth'),
    ('B2', './checkpoints/train_STDC2-Seg-Baseline2/pths/model_maxmIOU75.pth'),
]
SCALES = (0.75, 0.80, 0.85, 0.90)

RUNS = []
for rep_label, ckpt in BASELINES:
    for scale in SCALES:
        RUNS.append(('baseline-%s@%.2f' % (rep_label, scale), ckpt,
                      dict(use_brh=False, scale=scale)))


if __name__ == '__main__':
    out = report(RUNS, dspth='./data', backbone='STDCNet1446',
                 use_boundary_8=True, brh_mid=64)

    with open('pareto_accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'scale', 'full', 'bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3'])
        for (label, ckpt, over), (label2, res) in zip(RUNS, out):
            assert label == label2
            w.writerow([label, over['scale'], res['full'], res['bnd_r1'],
                        res['bnd_r3'], res['thin_r1'], res['thin_r3']])
    print('\nWrote pareto_accuracy.csv')
