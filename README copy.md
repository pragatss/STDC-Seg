export CUDA_VISIBLE_DEVICES=0
PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" /home/husky/anaconda3/envs/stdcseg/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC2-Seg-GCN/ \
  --backbone STDCNet1446 \
  --mode train \
  --n_workers_train 12 \
  --n_workers_val 1 \
  --max_iter 60000 \
  --use_boundary_8 True \
  --pretrain_path checkpoints/STDCNet1446_76.47.tar \
  --use_ctx_gcn True \
  --gcn_gate boundary


d be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  net.load_state_dict(torch.load(respth))
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:57<00:00,  1.75it/s]

==== ./checkpoints/train_STDC2-Seg-Baseline/pths/model_maxmIOU75.pth ====
full-image mIoU: 0.7600
boundary r=1 mIoU: 0.3595
   thin-class r=1 mIoU: 0.3340  (pole 0.335, tlight 0.337, tsign 0.367, rider 0.324, moto 0.264, bicycle 0.377)
boundary r=3 mIoU: 0.4513
   thin-class r=3 mIoU: 0.4218  (pole 0.430, tlight 0.435, tsign 0.479, rider 0.400, moto 0.325, bicycle 0.462)
self.mode val
self.len val 500
100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [00:55<00:00,  1.79it/s]

==== ./checkpoints/train_STDC2-Seg-GCN/pths/model_maxmIOU75.pth ====
full-image mIoU: 0.7504
boundary r=1 mIoU: 0.3560
   thin-class r=1 mIoU: 0.3347  (pole 0.333, tlight 0.345, tsign 0.362, rider 0.332, moto 0.262, bicycle 0.374)
boundary r=3 mIoU: 0.4462
   thin-class r=3 mIoU: 0.4215  (pole 0.427, tlight 0.442, tsign 0.473, rider 0.407, moto 0.322, bicycle 0.458)