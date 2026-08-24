# STDC2-Seg boundary-refinement experiments — results

All runs: STDC2 backbone (`STDCNet1446`), Cityscapes, `scale=0.75`, single GPU
(`--nproc_per_node=1`), `--max_iter 60000`, `--use_boundary_8 True`, pretrained
from `checkpoints/STDCNet1446_76.47.tar`. Evaluated with `evaluation.py`'s
`report()` (boundary-region IoU restricted to pixels within radius r of a
ground-truth class boundary; thin-class subset = pole, traffic light, traffic
sign, rider, motorcycle, bicycle).

## Noise floor

Two identical-config baseline runs (`A baseline` vs `A baseline v2`, no seed
pinning in `train.py`) give a measured run-to-run noise floor:

| metric | noise floor |
|---|---|
| full-image mIoU | ~0.9 pt |
| boundary IoU r=1 | ~0.3 pt |
| boundary IoU r=3 | ~0.5 pt |
| thin-class IoU r=1 | ~0.3 pt |
| thin-class IoU r=3 | ~0.55 pt |

Any delta smaller than this cannot be distinguished from noise on a single run.

## Runs

| label | respath | config |
|---|---|---|
| A baseline | `checkpoints/train_STDC2-Seg-Baseline/` | stock, no BRH, no boundary-weighted loss |
| A baseline v2 | `checkpoints/train_STDC2-Seg-Baseline2/` | identical config, independent run (noise-floor check) |
| I0 (control) | `checkpoints/train_STDC2-Seg-I0/` | `--bnd_weight 1.0` (== stock loss, should be a no-op) |
| I1 | `checkpoints/train_STDC2-Seg-I1/` | `--bnd_weight 3.0` |
| H1 | `checkpoints/train_STDC2-Seg-H1/` | `--use_brh True` (BoundaryRefine, stride-4 refine head) |
| HI1 | `checkpoints/train_STDC2-Seg-HI1/` | `--use_brh True --bnd_weight 3.0` (stacked) |

## Full comparison table

| run | full mIoU | bnd r=1 | bnd r=3 | thin r=1 | thin r=3 |
|---|---|---|---|---|---|
| A baseline | 0.7600 | 0.3595 | 0.4513 | 0.3340 | 0.4218 |
| A baseline v2 | 0.7512 | 0.3563 | 0.4465 | 0.3309 | 0.4163 |
| I0 λ=1 (control) | 0.7635 | 0.3583 | 0.4495 | 0.3299 | 0.4170 |
| I1 λ=3 | 0.7704 | 0.3778 | 0.4768 | 0.3543 | 0.4503 |
| H1 (BRH) | 0.7660 | 0.3757 | 0.4708 | 0.3540 | 0.4453 |
| **HI1 (BRH + λ=3)** | **0.7725** | **0.3977** | **0.4982** | **0.3862** | **0.4839** |

## Deltas vs. `A baseline`, with noise-floor verdicts

| run | bnd r=1 Δ | bnd r=3 Δ | thin r=1 Δ | thin r=3 Δ | verdict |
|---|---|---|---|---|---|
| I0 (control) | −0.12 | −0.18 | −0.41 | −0.48 | **noise** (as expected — λ=1 is a no-op; sanity-checks the methodology) |
| I1 (λ=3) | +1.83 | +2.55 | +2.03 | +2.85 | **real** — 6-8× the noise floor |
| H1 (BRH) | +1.62 | +1.95 | +2.00 | +2.34 | **real** — 4-6× the noise floor |
| HI1 (stacked) | +3.82 | +4.70 | +5.22 | +6.21 | **real, and large** — 8-20× the noise floor |

(Deltas in points, e.g. `+1.83` = `+0.0183` on the raw IoU scale.)

## Stacking: additive or synergistic?

Comparing HI1's actual gain to the naive sum of I1-alone + H1-alone:

| metric | I1 alone | H1 alone | naive sum | HI1 actual | vs. sum |
|---|---|---|---|---|---|
| bnd r=1 | +1.83 | +1.62 | +3.45 | +3.82 | +0.37 |
| bnd r=3 | +2.55 | +1.95 | +4.50 | +4.70 | +0.20 |
| thin r=1 | +2.03 | +2.00 | +4.03 | +5.22 | **+1.19** |
| thin r=3 | +2.85 | +2.34 | +5.19 | +6.21 | **+1.02** |

HI1 beats the naive additive sum on every metric — mildly on whole-boundary
metrics, substantially on thin-class metrics. Consistent with a mechanistic
clue from checkpoint loading: `H1` alone converges to `brh.res_scale = -2.16`,
but `HI1` converges to `brh.res_scale = +2.34` — **opposite sign**. The BRH
module is being used differently depending on whether the boundary-weighted
loss is also present, suggesting genuine interaction rather than two
independent, overlapping gains.

## Per-class thin IoU @ r=1 (baseline vs. best result, HI1)

| class | baseline | HI1 | Δ |
|---|---|---|---|
| pole | 0.335 | 0.448 | **+0.113** |
| traffic sign | 0.367 | 0.439 | **+0.072** |
| traffic light | 0.337 | 0.392 | **+0.055** |
| bicycle | 0.377 | 0.409 | +0.032 |
| motorcycle | 0.264 | 0.287 | +0.023 (rider/moto are rare classes with a higher individual noise floor — hold these two more loosely) |
| rider | 0.324 | 0.343 | +0.019 |

## Bottom line

`HI1` (BRH + boundary-weighted loss, λ=3) is the best result across every
metric measured, and the combination outperforms the naive sum of its two
components — particularly on the thin-class subset both interventions
specifically target. `I0`'s deltas landing inside the noise band (as its
design predicts, since `λ=1` reproduces the stock loss exactly) is a working
sanity check on this whole comparison methodology, which is what gives the
I1/H1/HI1 results above the noise floor real credibility.

## Caveats

- No seed pinning in `train.py` — the noise-floor table above is the
  practical consequence, estimated from n=2 (single baseline pair). Treat it
  as a rough estimate, not a precise variance.
- Single GPU (`--nproc_per_node=1`) throughout. `InPlaceABNSync` is designed
  for cross-GPU synchronized BatchNorm; on 1 GPU it degenerates to plain BN
  over a smaller effective batch (16 vs. the original recipe's 48 on 3 GPUs).
  This project's own baseline (0.75-0.76 mIoU75) is below the STDC2-Seg75
  paper/README number (0.7704) for this reason — all comparisons above are
  internally consistent (same hardware, same handicap, throughout), but
  should not be read as directly comparable to the paper's published numbers.
