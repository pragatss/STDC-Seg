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
