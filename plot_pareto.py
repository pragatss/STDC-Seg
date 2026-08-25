#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Plot the equal-latency Pareto figure: x = latency (ms), y = boundary IoU
r=1, for the stock-baseline-at-various-scales curve vs the I1/H1/HI1
treatment points.

Reads:
    latency/pareto_latency.csv   (from measure_pareto_latency.py)
    pareto_accuracy.csv          (from pareto_accuracy_sweep.py)

Treatment-arm accuracy (I1/H1/HI1) and the n=3 baseline@0.75 reference are
hardcoded below from the numbers already reported in this experiment
(scale=0.75, Cityscapes val) -- re-run evaluation.py's own RUNS list if you
want those regenerated rather than taken as given.

USAGE
    python plot_pareto.py
Writes pareto.png.
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- given, already-reported numbers (scale=0.75) --------------------------
NOISE_R1 = 0.0032  # measured noise floor, bnd r=1, from 3 baseline training replicates
BASELINE_N3_R1 = 0.3580  # baseline mean (n=3) bnd r=1, for reference against the fresh B1/B2 point
TREATMENTS = {
    # label: (bnd_r1, is_brh)
    'I1 (loss only)':    (0.3778, False),
    'H1 (BRH only)':     (0.3757, True),
    'HI1 (both)':        (0.3977, True),
}
LAT_LABEL_FOR_TREATMENT = {
    'I1 (loss only)': 'I1@0.75',
    'H1 (BRH only)':  'H1@0.75',
    'HI1 (both)':     'HI1@0.75',
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    lat_rows = read_csv('latency/pareto_latency.csv')
    lat_by_label = {r['label']: r for r in lat_rows}

    acc_rows = read_csv('pareto_accuracy.csv')
    by_scale = defaultdict(list)
    for r in acc_rows:
        by_scale[float(r['scale'])].append(float(r['bnd_r1']))

    curve_x, curve_y, curve_yerr, curve_n, curve_scale = [], [], [], [], []
    for scale in sorted(by_scale):
        label = 'baseline@%.2f' % scale
        if label not in lat_by_label:
            print('  [warn] no latency row for %s, skipping' % label)
            continue
        latency = float(lat_by_label[label]['mean_ms'])
        vals = by_scale[scale]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        curve_x.append(latency)
        curve_y.append(mean)
        curve_yerr.append(std)
        curve_n.append(len(vals))
        curve_scale.append(scale)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.errorbar(curve_x, curve_y, yerr=curve_yerr, marker='o', color='0.35',
                linewidth=1.5, capsize=3, label='stock baseline (upscale)', zorder=2)
    for x, y, n, s in zip(curve_x, curve_y, curve_n, curve_scale):
        ax.annotate('%.2f (n=%d)' % (s, n), (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=8, color='0.35')

    # n=3 reference point at scale=0.75, plotted separately from the fresh
    # n=2 recompute above so any discrepancy between them is visible rather
    # than silently averaged away.
    if 'baseline@0.75' in lat_by_label:
        x75 = float(lat_by_label['baseline@0.75']['mean_ms'])
        ax.errorbar([x75], [BASELINE_N3_R1], yerr=[NOISE_R1], marker='s', color='0.6',
                    linestyle='none', capsize=3, label='baseline (n=3, reported)', zorder=2)

    colors = {'I1 (loss only)': 'tab:blue', 'H1 (BRH only)': 'tab:orange', 'HI1 (both)': 'tab:red'}
    for label, (y, is_brh) in TREATMENTS.items():
        lat_label = LAT_LABEL_FOR_TREATMENT[label]
        if lat_label not in lat_by_label:
            print('  [warn] no latency row for %s, skipping' % lat_label)
            continue
        x = float(lat_by_label[lat_label]['mean_ms'])
        xerr = float(lat_by_label[lat_label]['std_ms'])
        ax.errorbar([x], [y], xerr=[xerr], yerr=[NOISE_R1], marker='D',
                    color=colors[label], linestyle='none', capsize=3, label=label, zorder=3)

    ax.set_xlabel('latency, ms/frame (512-input-scale forward pass, FP32, PyTorch)')
    ax.set_ylabel('boundary IoU, r=1 (Cityscapes val)')
    ax.set_title('Equal-latency Pareto: BRH/loss arms vs. stock-at-higher-scale')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('pareto.png', dpi=200)
    print('Wrote pareto.png')


if __name__ == '__main__':
    main()
