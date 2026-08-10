#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.util import enet_weighing
import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

def boundary_weight_map(labels, radius=3, w_bnd=3.0, ignore_lb=255):
    """1.0 everywhere, w_bnd within `radius` px of a GT class boundary."""
    lab = labels
    N, H, W = lab.shape
    bnd = torch.zeros((N, H, W), device=lab.device)
    d = lab[:,:,1:] != lab[:,:,:-1]
    v = (lab[:,:,1:] != ignore_lb) & (lab[:,:,:-1] != ignore_lb)
    e = (d & v).float()
    # torch.maximum doesn't exist in torch 1.1.0 (required here for the InPlaceABNSync
    # CUDA extension to build) -- torch.max(a, b) is the elementwise-max equivalent.
    bnd[:,:,1:]  = torch.max(bnd[:,:,1:],  e)
    bnd[:,:,:-1] = torch.max(bnd[:,:,:-1], e)
    d = lab[:,1:,:] != lab[:,:-1,:]
    v = (lab[:,1:,:] != ignore_lb) & (lab[:,:-1,:] != ignore_lb)
    e = (d & v).float()
    bnd[:,1:,:]  = torch.max(bnd[:,1:,:],  e)
    bnd[:,:-1,:] = torch.max(bnd[:,:-1,:], e)
    k = 2 * radius + 1
    band = F.max_pool2d(bnd.unsqueeze(1), kernel_size=k, stride=1, padding=radius).squeeze(1)
    return 1.0 + (w_bnd - 1.0) * (band > 0.5).float()


class BoundaryOhemCELoss(nn.Module):
    """OHEM cross-entropy that upweights pixels near GT boundaries.
       w_bnd=1.0 reproduces stock OhemCELoss exactly."""
    def __init__(self, thresh, n_min, ignore_lb=255, radius=3, w_bnd=3.0, *args, **kwargs):
        super(BoundaryOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.radius = radius
        self.w_bnd = w_bnd
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels)                       # [N,H,W]
        w = boundary_weight_map(labels, self.radius, self.w_bnd, self.ignore_lb)
        loss = (loss * w).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class WeightedOhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, num_classes, ignore_lb=255, *args, **kwargs):
        super(WeightedOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.num_classes = num_classes
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(), ignore_index=self.ignore_lb, reduction='none')
        loss = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
