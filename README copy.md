## Run predict images

### Baseline best-checkpoint predictions

```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod conda run -n stdcseg18 --no-capture-output python scripts/predict_images.py --weights checkpoints/train_STDC1-Seg/baseline_no_plane --out_dir predictions/baseline_no_plane --images_dir scripts/images --backbone STDCNet813 --scale 0.5
```

### Plane model best-checkpoint predictions

```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod conda run -n stdcseg18 --no-capture-output python scripts/predict_images.py --weights checkpoints/train_STDC1-Seg/plane_setcriterion_w002_delay20k --out_dir predictions/plane_setcriterion_w002_delay20k --images_dir scripts/images --backbone STDCNet813 --scale 0.5
```



## Per-class IoU comparison

### Best-vs-best comparison against baseline

```bash
PYTHONPATH=/home/husky/Downloads/Pragat/STDC-Seg-Mod conda run -n stdcseg18 --no-capture-output python scripts/evaluate_per_class_iou.py --target_weights checkpoints/train_STDC1-Seg/plane_setcriterion_w002_delay20k --data_root ./data --backbone STDCNet813 --scale 0.5
```

### Notes

- `scripts/predict_images.py` prefers `model_maxmIOU50.pth` for `--scale 0.5` and `model_maxmIOU75.pth` for `--scale 0.75`.
- `scripts/evaluate_per_class_iou.py` compares best-vs-best by default when passed a run directory.
- CSV output is written under `predictions/per_class_iou/`.
