#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import imageio

from transform import *


# SYNTHIA-RAND-CITYSCAPES raw class id -> Cityscapes trainId.
# Verified against the id_to_trainid tables used by AdaptSegNet/CLAN and
# valeoai/DADA for SYNTHIA->Cityscapes domain adaptation. Anything not
# listed here (void, lanemarking, parking-slot, road-work, ...) has no
# Cityscapes trainId counterpart and is left as ignore_lb.
SYNTHIA_ID_TO_TRAINID = {
    3: 0,    # road
    4: 1,    # sidewalk
    2: 2,    # building
    21: 3,   # wall
    5: 4,    # fence
    7: 5,    # pole
    15: 6,   # traffic light
    9: 7,    # traffic sign
    6: 8,    # vegetation
    16: 9,   # terrain
    1: 10,   # sky
    10: 11,  # person
    17: 12,  # rider
    8: 13,   # car
    18: 14,  # truck
    19: 15,  # bus
    20: 16,  # train
    12: 17,  # motorcycle
    11: 18,  # bicycle
}


class Synthia(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train',
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5),
    *args, **kwargs):
        super(Synthia, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255
        self.id_to_trainid = SYNTHIA_ID_TO_TRAINID

        ## on-disk layout mirrors Cityscapes: RGB/<split> and
        ## GT/LABELS/<split>, pre-split at the same ratios as this repo's
        ## Cityscapes train/val/test (59.5%/10%/30.5%).
        splits = ('train', 'val') if mode == 'trainval' else (mode,)
        names = []
        self.imgs = {}
        self.labels = {}
        for split in splits:
            impth = osp.join(rootpth, 'RGB', split)
            gtpth = osp.join(rootpth, 'GT', 'LABELS', split)
            im_names = set(os.listdir(impth))
            gt_names = set(os.listdir(gtpth))
            split_names = sorted(im_names & gt_names)
            assert len(split_names) > 0, 'no overlapping filenames found under {} and {}'.format(impth, gtpth)
            names.extend(split_names)
            self.imgs.update({n: osp.join(impth, n) for n in split_names})
            self.labels.update({n: osp.join(gtpth, n) for n in split_names})

        self.imnames = names
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        ## SYNTHIA GT/LABELS pngs are 16-bit dual-channel (class id in
        ## channel 0, instance id in channel 1); plain PIL misreads them,
        ## so decode via the freeimage plugin like other SYNTHIA loaders do.
        raw_label = imageio.imread(lbpth, format='PNG-FI')[:, :, 0]
        label = Image.fromarray(raw_label.astype(np.uint8))
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        out = np.full_like(label, self.ignore_lb)
        for k, v in self.id_to_trainid.items():
            out[label == k] = v
        return out



if __name__ == "__main__":
    from tqdm import tqdm
    ds = Synthia('./data/SYNTHIA', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(set(uni))
