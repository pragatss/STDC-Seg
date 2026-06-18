# Plane Head Training Observer

## Current Run: `plane_setcriterion_w015_layers16`

| Parameter | Previous (`w005_delay20k`) | Current |
|---|---|---|
| `plane_loss_weight` | 0.05 | **0.15** |
| `plane_aux_start_iter` | 20000 | **0** |
| `plane_aux_decoder_layers` | 10 | **16** |

### Command
```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod CUDA_VISIBLE_DEVICES=0 conda run -n stdcseg18 --no-capture-output torchrun --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC1-Seg/plane_setcriterion_w015_layers16 \
  --backbone STDCNet813 \
  --mode train \
  --use_boundary_8 false \
  --use_boundary_4 false \
  --use_boundary_2 false \
  --use_boundary_16 false \
  --use_plane_aux true \
  --plane_aux_mode setcriterion \
  --plane_aux_tap fuse \
  --plane_aux_num_queries 20 \
  --plane_aux_num_classes 20 \
  --plane_aux_hidden_dim 256 \
  --plane_aux_decoder_layers 16 \
  --plane_set_cost_class 2.0 \
  --plane_set_cost_mask 5.0 \
  --plane_set_cost_dice 5.0 \
  --plane_set_num_points 3072 \
  --plane_set_oversample_ratio 3.0 \
  --plane_set_importance_sample_ratio 0.75 \
  --plane_set_eos_coef 0.1 \
  --plane_set_weight_ce 2.0 \
  --plane_set_weight_mask 5.0 \
  --plane_set_weight_dice 5.0 \
  --soft_targets_dir soft_targets \
  --plane_loss_weight 0.15 \
  --plane_aux_start_iter 0 \
  --n_img_per_gpu 4 \
  --max_iter 60000 \
  --save_iter_sep 1000
```

---

## What to Watch

### Signal 1 — Does `plane_aux_loss` drop in the first 10k iters?
- **Good:** `17 → 14 → 11 → 9 → ...` (actively decreasing)
- **Bad:** `17 → 17 → 17 → 17` (flatline) → see Scenario A below

### Signal 2 — Does `plane_aux_loss` break below 6.9 by iter 30-40k?
`6.9` is the ceiling the previous run hit and never broke through.
- **Good:** loss reaches 5.0, 4.0, or lower
- **Bad:** plateaus again at ~6.9 → see Scenario B below

### Signal 3 — Does seg mIOU recover by iter 60k?
Starting plane loss from iter 0 will cause an early mIOU dip — that's expected.
- **Reference:** `baseline_no_plane` finished at mIOU50 ~0.53
- **Good:** finishes at mIOU50 >= 0.50
- **Bad:** finishes at mIOU50 < 0.45 → see Scenario C below

---

## Next Steps if Bad

### Scenario A — `plane_aux_loss` flatlines at ~17 from iter 0
**Cause:** soft targets not loading, plane loss not being applied.

Check the soft targets directory:
```bash
ls soft_targets/ | head -5
# should show .npy files, one per Cityscapes training image
```
Fix: verify the directory exists and rerun the same command.

---

### Scenario B — `plane_aux_loss` drops to ~6.9 then plateaus again
**Cause:** model capacity or soft target quality is the bottleneck.

**Option B1 — More decoder layers (capacity):**
```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod CUDA_VISIBLE_DEVICES=0 conda run -n stdcseg18 --no-capture-output torchrun --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC1-Seg/plane_setcriterion_w015_layers20 \
  --backbone STDCNet813 --mode train \
  --use_boundary_8 false --use_boundary_4 false \
  --use_boundary_2 false --use_boundary_16 false \
  --use_plane_aux true --plane_aux_mode setcriterion --plane_aux_tap fuse \
  --plane_aux_num_queries 20 --plane_aux_num_classes 20 \
  --plane_aux_hidden_dim 256 --plane_aux_decoder_layers 20 \
  --plane_set_cost_class 2.0 --plane_set_cost_mask 5.0 --plane_set_cost_dice 5.0 \
  --plane_set_num_points 3072 --plane_set_oversample_ratio 3.0 \
  --plane_set_importance_sample_ratio 0.75 --plane_set_eos_coef 0.1 \
  --plane_set_weight_ce 2.0 --plane_set_weight_mask 5.0 --plane_set_weight_dice 5.0 \
  --soft_targets_dir soft_targets \
  --plane_loss_weight 0.15 --plane_aux_start_iter 0 \
  --n_img_per_gpu 4 --max_iter 60000 --save_iter_sep 1000
```

**Option B2 — Higher weight (stronger signal):**

Same as the current command but with:
```
--plane_loss_weight 0.25
--respath checkpoints/train_STDC1-Seg/plane_setcriterion_w025_layers16
```

---

### Scenario C — Seg mIOU finishes below 0.45 (weight too aggressive)
**Cause:** plane loss at 0.15 is overwhelming seg training.

**Fix:** reduce weight and add a short delay:
```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod CUDA_VISIBLE_DEVICES=0 conda run -n stdcseg18 --no-capture-output torchrun --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC1-Seg/plane_setcriterion_w010_delay5k_layers16 \
  --backbone STDCNet813 --mode train \
  --use_boundary_8 false --use_boundary_4 false \
  --use_boundary_2 false --use_boundary_16 false \
  --use_plane_aux true --plane_aux_mode setcriterion --plane_aux_tap fuse \
  --plane_aux_num_queries 20 --plane_aux_num_classes 20 \
  --plane_aux_hidden_dim 256 --plane_aux_decoder_layers 16 \
  --plane_set_cost_class 2.0 --plane_set_cost_mask 5.0 --plane_set_cost_dice 5.0 \
  --plane_set_num_points 3072 --plane_set_oversample_ratio 3.0 \
  --plane_set_importance_sample_ratio 0.75 --plane_set_eos_coef 0.1 \
  --plane_set_weight_ce 2.0 --plane_set_weight_mask 5.0 --plane_set_weight_dice 5.0 \
  --soft_targets_dir soft_targets \
  --plane_loss_weight 0.10 --plane_aux_start_iter 5000 \
  --n_img_per_gpu 4 --max_iter 60000 --save_iter_sep 1000
```

---

## Reference: `w005_delay20k` Results (previous run)

| Metric | Value |
|---|---|
| `plane_aux_loss` at turn-on (iter 20k) | ~17.5 |
| `plane_aux_loss` at end (iter 60k) | ~6.9 (plateau) |
| Plane % of total loss at turn-on | 19% |
| Plane % of total loss at end | 10% |
| Final mIOU50 | 0.531 |
| Final mIOU75 | 0.565 |
| Plane predictions | Partially learned, visually poor — queries not specialized |
