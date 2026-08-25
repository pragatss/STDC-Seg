#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Equal-cost Pareto figures: stock-baseline-at-various-scales curve vs the
I1/H1/HI1 treatment points, on two cost axes:

  pareto_flops.png     GFLOPs (primary -- deterministic, reproducible,
                        doesn't depend on GPU/thermal state at all)
  pareto_latency.png   wall-clock ms/frame (secondary/supplementary --
                        included because a real-time paper needs a wall-clock
                        number too, but caveat the noise explicitly)

Scales are the 32-divisible corrected set (see pareto_accuracy_sweep.py's
docstring for why: the original 0.80/0.85/0.90 sweep hit a non-integer
nearest-neighbor upsample ratio at STDCNet1446's stride-32 feature map,
producing a non-monotonic boundary-IoU artifact).

Reads:
    pareto_flops.csv             (from pareto_flops.py)
    pareto_accuracy.csv          (from pareto_accuracy_sweep.py)
    latency/pareto_latency.csv   (from measure_pareto_latency.py)

Treatment-arm accuracy (I1/H1/HI1) is hardcoded below from the numbers
already reported for this experiment (scale=0.75, Cityscapes val).

USAGE
    python plot_pareto.py
"""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_R1 = 0.0032  # measured noise floor, bnd r=1, from 3 baseline training replicates
TREATMENTS = {
    # label: bnd_r1
    'I1 (loss only)': 0.3778,
    'H1 (BRH only)':  0.3757,
    'HI1 (both)':     0.3977,
}
TREATMENT_ARCH = {  # which architecture's FLOPs/latency to use
    'I1 (loss only)': 'stock',
    'H1 (BRH only)':  'brh',
    'HI1 (both)':     'brh',
}
COLORS = {'I1 (loss only)': 'tab:blue', 'H1 (BRH only)': 'tab:orange', 'HI1 (both)': 'tab:red'}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def accuracy_curve():
    """scale -> (mean bnd_r1, std, n) across baseline replicates."""
    rows = read_csv('pareto_accuracy.csv')
    by_scale = defaultdict(list)
    for r in rows:
        by_scale[round(float(r['scale']), 5)].append(float(r['bnd_r1']))
    out = {}
    for scale, vals in by_scale.items():
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        out[scale] = (mean, std, len(vals))
    return out


def make_plot(x_curve, x_treat, xlabel, title, out_path, acc_curve, treat_xerr=None):
    scales = sorted(x_curve)
    xs = [x_curve[s] for s in scales]
    ys = [acc_curve[s][0] for s in scales]
    yerr = [acc_curve[s][1] for s in scales]
    ns = [acc_curve[s][2] for s in scales]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(xs, ys, yerr=yerr, marker='o', color='0.35', linewidth=1.5,
                capsize=3, label='stock baseline (upscale)', zorder=2)
    for x, y, s, n in zip(xs, ys, scales, ns):
        ax.annotate('%.4g (n=%d)' % (s, n), (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=8, color='0.35')

    for label, y in TREATMENTS.items():
        if label not in x_treat:
            continue
        x = x_treat[label]
        xerr = treat_xerr[label] if treat_xerr else None
        ax.errorbar([x], [y], xerr=[xerr] if xerr else None, yerr=[NOISE_R1], marker='D',
                    color=COLORS[label], linestyle='none', capsize=3, label=label, zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('boundary IoU, r=1 (Cityscapes val)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print('Wrote %s' % out_path)


def main():
    acc_curve = accuracy_curve()

    # --- FLOPs (primary) ---------------------------------------------------
    flops_rows = read_csv('pareto_flops.csv')
    flops_by_scale = {round(float(r['scale']), 5): r for r in flops_rows}
    x_curve_flops = {s: float(flops_by_scale[s]['gflops_stock']) for s in acc_curve if s in flops_by_scale}
    ref = flops_by_scale[round(0.75, 5)]
    x_treat_flops = {
        label: float(ref['gflops_stock'] if arch == 'stock' else ref['gflops_brh'])
        for label, arch in TREATMENT_ARCH.items()
    }
    make_plot(x_curve_flops, x_treat_flops, 'GFLOPs (thop, full multi-head forward, 1x3xHxW)',
              'Equal-FLOPs Pareto: BRH/loss arms vs. stock-at-higher-scale',
              'pareto_flops.png', acc_curve)

    # --- latency (secondary) -----------------------------------------------
    lat_rows = read_csv('latency/pareto_latency.csv')
    lat_by_label = {r['label']: r for r in lat_rows}
    x_curve_lat, xerr_curve_lat = {}, {}
    for s in acc_curve:
        label = 'baseline@%s' % (('%.5f' % s).rstrip('0').rstrip('.'))
        # try a couple of formattings since float->str isn't canonical
        candidates = [label, 'baseline@%s' % s, 'baseline@%.4g' % s]
        row = next((lat_by_label[c] for c in candidates if c in lat_by_label), None)
        if row is None:
            print('  [warn] no latency row matched scale %s, skipping from latency plot' % s)
            continue
        x_curve_lat[s] = float(row['mean_ms'])

    lat_label_for_treatment = {'I1 (loss only)': 'I1@0.75', 'H1 (BRH only)': 'H1@0.75', 'HI1 (both)': 'HI1@0.75'}
    x_treat_lat, xerr_treat_lat = {}, {}
    for label, lat_label in lat_label_for_treatment.items():
        if lat_label in lat_by_label:
            x_treat_lat[label] = float(lat_by_label[lat_label]['mean_ms'])
            xerr_treat_lat[label] = float(lat_by_label[lat_label]['std_ms'])

    make_plot(x_curve_lat, x_treat_lat,
              'latency, ms/frame (PyTorch FP32, GPU clocks locked but contention-free only '
              'when noted -- see caveats)',
              'Equal-latency Pareto (supplementary -- higher noise floor than FLOPs)',
              'pareto_latency.png', acc_curve, treat_xerr=xerr_treat_lat)


if __name__ == '__main__':
    main()
