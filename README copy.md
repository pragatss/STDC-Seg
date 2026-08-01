baseline
python -m torch.distributed.launch --nproc_per_node=1 train.py --respath checkpoints/train_STDC2-Seg/ --backbone STDCNet1446 --mode train --n_workers_train 12 --n_workers_val 1 --max_iter 60000 --use_boundary_8 True --pretrain_path checkpoints/STDCNet1446_76.47.tar

res on train:
max mIOU model saved to: checkpoints/train_STDC2-Seg/pths/model_maxmIOU50.pth
max mIOU model saved to: checkpoints/train_STDC2-Seg/pths/model_maxmIOU75.pth
mIOU50 is: 0.7217114567756653, mIOU75 is: 0.7599517703056335
maxmIOU50 is: 0.7217114567756653, maxmIOU75 is: 0.7599517703056335.

res on eval
evaluatev0('./checkpoints/train_STDC2-Seg/pths/model_maxmIOU75.pth', dspth='./data', backbone='STDCNet1446', scale=0.75, 
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
mIOU is: 0.7599517703056335

ARM B, checkpoint 4 use_variance false use_segmentation false
export CUDA_VISIBLE_DEVICES=0
/home/husky/anaconda3/envs/stdcseg/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC2-Seg-ARM-B/ \
  --backbone STDCNet1446 \
  --mode train \
  --n_workers_train 12 \
  --n_workers_val 1 \
  --max_iter 60000 \
  --use_boundary_8 True \
  --pretrain_path checkpoints/STDCNet1446_76.47.tar
res on train:
max mIOU model saved to: checkpoints/train_STDC2-Seg/pths/model_maxmIOU50.pth
mIOU50 is: 0.7217464447021484, mIOU75 is: 0.7517090439796448
maxmIOU50 is: 0.7217464447021484, maxmIOU75 is: 0.7522354125976562.
training done, model saved to: checkpoints/train_STDC2-Seg/pths/model_final.pth

eval run
CUDA_VISIBLE_DEVICES=0 python evaluation.py
mIOU is: 0.752235472202301

ARM C, checkpoint 5 use_variance true use_segmentation false
export CUDA_VISIBLE_DEVICES=0
/home/husky/anaconda3/envs/stdcseg/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC2-Seg-ARM-C/ \
  --backbone STDCNet1446 \
  --mode train \
  --n_workers_train 12 \
  --n_workers_val 1 \
  --max_iter 60000 \
  --use_boundary_8 True \
  --pretrain_path checkpoints/STDCNet1446_76.47.tar \
  --use_sbg True \
  --use_variance True

  mOU50_0.7241_mIOU75_0.7544.pth
max mIOU model saved to: checkpoints/train_STDC2-Seg-ARM-C/pths/model_maxmIOU50.pth
mIOU50 is: 0.7240659594535828, mIOU75 is: 0.7544452548027039
maxmIOU50 is: 0.7240659594535828, maxmIOU75 is: 0.7546665668487549.

#ARM D
export CUDA_VISIBLE_DEVICES=0
PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" /home/husky/anaconda3/envs/stdcseg/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC2-Seg-ARM-D/ \
  --backbone STDCNet1446 \
  --mode train \
  --n_workers_train 12 \
  --n_workers_val 1 \
  --max_iter 60000 \
  --use_boundary_8 True \
  --pretrain_path checkpoints/STDCNet1446_76.47.tar \
  --use_sbg True \
  --use_variance True \
  --use_semantic True

  training iteration 60000, model saved to: checkpoints/train_STDC2-Seg-ARM-D/pths/model_iter60000_mIOU50_0.7252_mIOU75_0.7541.pth
max mIOU model saved to: checkpoints/train_STDC2-Seg-ARM-D/pths/model_maxmIOU75.pth
mIOU50 is: 0.7251720428466797, mIOU75 is: 0.7541453838348389
maxmIOU50 is: 0.7253044843673706, maxmIOU75 is: 0.7541453838348389.
training done, model saved to: checkpoints/train_STDC2-Seg-ARM-D/pths/model_final.pth
epoch:  324


ARM E(hr)
export CUDA_VISIBLE_DEVICES=0
PATH="/home/husky/anaconda3/envs/stdcseg18/bin:$PATH" /home/husky/anaconda3/envs/stdcseg/bin/python -m torch.distributed.launch --nproc_per_node=1 train.py \
  --respath checkpoints/train_STDC2-Seg-ARM-E/ \
  --backbone STDCNet1446 \
  --mode train \
  --n_workers_train 12 \
  --n_workers_val 1 \
  --max_iter 60000 \
  --use_boundary_8 True \
  --pretrain_path checkpoints/STDCNet1446_76.47.tar \
  --use_sbg True \
  --use_variance True \
  --use_semantic True \
  --semantic_source hr
.pth
max mIOU model saved to: checkpoints/train_STDC2-Seg-ARM-E/pths/model_maxmIOU50.pth
max mIOU model saved to: checkpoints/train_STDC2-Seg-ARM-E/pths/model_maxmIOU75.pth
mIOU50 is: 0.7251105904579163, mIOU75 is: 0.755883514881134
maxmIOU50 is: 0.7251105904579163, maxmIOU75 is: 0.755883514881134.
training done, model saved to: checkpoints/train_STDC2-Seg-ARM-E/pths/model_final.pth