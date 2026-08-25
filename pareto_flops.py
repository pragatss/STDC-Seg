#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Deterministic cost metrics (params, GFLOPs) for baseline vs BRH architecture,
to use as the primary cost axis instead of throttled/noisy wall-clock
latency. Unlike latency, these numbers don't depend on GPU state at all --
same machine, same run, every time.

I1 is architecturally byte-identical to baseline (loss-only change), so its
params/FLOPs are identical to baseline by construction -- not computed
separately. HI1 shares H1's architecture (BRH + loss change), so HI1's
params/FLOPs equal H1's.

Reports at each of the scales used in the corrected accuracy/latency sweep,
since FLOPs scale with input resolution and the whole point is comparing
BRH's fixed architectural cost against the STOCK baseline's cost at various
test-time resolutions.

USAGE
    PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" python pareto_flops.py
Writes pareto_flops.csv.
"""
import csv

import torch
from thop import profile

from models.model_stages import BiSeNet

CITYSCAPES_HW = (1024, 2048)
SCALES = (0.75, 0.78125, 0.8125, 0.875, 0.9375, 1.0)


def build(use_brh, brh_mid=64):
    net = BiSeNet(backbone='STDCNet1446', n_classes=19,
                  use_boundary_2=False, use_boundary_4=False,
                  use_boundary_8=True, use_boundary_16=False,
                  use_conv_last=False, use_brh=use_brh, brh_mid=brh_mid)
    return net.eval()


def count(net, h, w):
    x = torch.randn(1, 3, h, w)
    flops, params = profile(net, inputs=(x,), verbose=False)
    return flops, params


def main():
    H0, W0 = CITYSCAPES_HW
    net_stock = build(use_brh=False)
    net_brh = build(use_brh=True)

    rows = []
    print('%-10s %10s %12s %12s %10s %10s %8s' %
          ('scale', 'input', 'GFLOPs(stk)', 'GFLOPs(brh)', 'params(stk)', 'params(brh)', '%FLOPs'))
    for s in SCALES:
        h, w = int(H0 * s), int(W0 * s)
        f_stock, p_stock = count(net_stock, h, w)
        f_brh, p_brh = count(net_brh, h, w)
        pct = 100.0 * (f_brh - f_stock) / f_stock
        print('%-10.4f %4dx%-5d %12.3f %12.3f %10.3fM %10.3fM %+7.2f%%' %
              (s, h, w, f_stock / 1e9, f_brh / 1e9, p_stock / 1e6, p_brh / 1e6, pct))
        rows.append(dict(scale=s, h=h, w=w,
                          gflops_stock=f_stock / 1e9, gflops_brh=f_brh / 1e9,
                          params_stock_M=p_stock / 1e6, params_brh_M=p_brh / 1e6,
                          pct_flops_overhead=pct))

    print('\nparam delta (brh - stock): %.4fM' % ((p_brh - p_stock) / 1e6))

    with open('pareto_flops.csv', 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print('\nWrote pareto_flops.csv')
    print('I1 == stock (loss-only, architecturally identical). HI1 == BRH (loss change does not touch params/FLOPs).')


if __name__ == '__main__':
    main()
