#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train',
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5),
    soft_targets_dir=None, *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255
        self.soft_targets_dir = soft_targets_dir

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## build soft-target path map (optional)
        self.soft_target_paths = {}
        if soft_targets_dir is not None:
            st_mode_dir = osp.join(soft_targets_dir, mode)
            if osp.isdir(st_mode_dir):
                for fd in os.listdir(st_mode_dir):
                    fdpth = osp.join(st_mode_dir, fd)
                    if not osp.isdir(fdpth):
                        continue
                    for fname in os.listdir(fdpth):
                        if fname.endswith('.npy'):
                            name = fname[:-4]  # strip .npy
                            self.soft_target_paths[name] = osp.join(fdpth, fname)
                print('soft_targets_dir: found {} soft target files for split={}'.format(
                      len(self.soft_target_paths), mode))
            else:
                print('WARNING: soft_targets_dir={} does not contain split dir {}'.format(
                      soft_targets_dir, mode))

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
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        # Load cached soft target (.npy) if available for this image.
        soft_target = None
        if self.soft_target_paths and fn in self.soft_target_paths:
            st_arr = np.load(self.soft_target_paths[fn]).astype(np.float32)  # (21, H_st, W_st)
            soft_target = st_arr

        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im=img, lb=label, st=soft_target)
            im_lb = self.trans_train(im_lb)
            img, label, soft_target = im_lb['im'], im_lb['lb'], im_lb.get('st', None)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)

        # Convert soft target to float32 tensor if it was loaded.
        if soft_target is not None:
            soft_target = torch.from_numpy(
                np.ascontiguousarray(soft_target).astype(np.float32)
            )  # (21, H_st, W_st)

        return img, label, soft_target


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb, *_ in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

