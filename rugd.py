#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np

from transform import *


# RUGD class titles in the order used as trainIds (0..23). This is the class
# order of the DatasetNinja meta.json for RUGD, which is alphabetical and
# drops the official "void" class -- void pixels are simply left unlabelled
# by the annotations and become ignore_lb.
RUGD_CLASSES = [
    'asphalt',       # 0
    'bicycle',       # 1
    'bridge',        # 2
    'building',      # 3
    'bush',          # 4
    'concrete',      # 5
    'container',     # 6
    'dirt',          # 7
    'fence',         # 8
    'grass',         # 9
    'gravel',        # 10
    'log',           # 11
    'mulch',         # 12
    'person',        # 13
    'picnic-table',  # 14
    'pole',          # 15
    'rock',          # 16
    'rock-bed',      # 17
    'sand',          # 18
    'sign',          # 19
    'sky',           # 20
    'tree',          # 21
    'vehicle',       # 22
    'water',         # 23
]


class RUGD(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train',
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5),
    *args, **kwargs):
        super(RUGD, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        ## on-disk layout is the DatasetNinja/Supervisely export: <split>/img
        ## holds the RGB pngs and <split>/label holds the single-channel
        ## trainId pngs produced by prepare_rugd.py from <split>/ann.
        splits = ('train', 'val') if mode == 'trainval' else (mode,)
        names = []
        self.imgs = {}
        self.labels = {}
        for split in splits:
            impth = osp.join(rootpth, split, 'img')
            gtpth = osp.join(rootpth, split, 'label')
            assert osp.isdir(gtpth), \
                '{} not found -- run "python prepare_rugd.py" first'.format(gtpth)
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
        ## already stored as trainIds (0..23, 255=void), so no remap here.
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label


    def __len__(self):
        return self.len



if __name__ == "__main__":
    from tqdm import tqdm
    ds = RUGD('./data/rugd', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(set(uni))
