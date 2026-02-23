
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import json

def dice_loss_func(input, target):
    """
    Soft Dice loss used as part of the Detail loss (Paper §3.3, Eq. 3).
    Complements BCE by directly optimising the overlap between predicted and
    ground-truth boundary regions, helping with the extreme class imbalance
    between boundary and non-boundary pixels.
    """
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).cuda()
    ones = ones.index_select(0, label.long())   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def get_boundary(gtmasks):
    # Discrete Laplacian operator used to extract binary boundary maps from
    # ground-truth segmentation masks (Paper §3.3, Fig. 6 — GT boundary generation).
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, 8, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = F.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets


class DetailAggregateLoss(nn.Module):
    """
    Detail Aggregation Learning loss — Paper §3.3 / Fig. 6.

    THE BIG PICTURE:
    The original BiSeNet kept a whole separate "Detail Branch" running at
    inference time just to preserve sharp edges.  That branch was expensive.
    This paper's solution: don't run the Detail Branch at inference at all.
    Instead, use a special loss to TEACH the network to be edge-aware during
    training, then discard the boundary heads afterwards.

    HOW THE GROUND-TRUTH EDGES ARE GENERATED (automatically, no manual labelling):
      1. Take the segmentation mask (which we already have as ground truth).
      2. Apply a Laplacian filter to it — think of this like a "find the edges"
         filter used in image processing.  Pixels where one class meets another
         will light up; interior pixels will be zero.
      3. Threshold to get a clean binary edge map (1 = edge, 0 = not edge).
      4. Repeat at 1/2 and 1/4 scales to capture thick and thin edges alike.
      5. Fuse all three scales with a small learnable weighted sum (fuse_kernel)
         into one final boundary supervision map.

    HOW THE NETWORK IS PENALISED:
      Two loss terms are used together (Paper Eq. 3):
        BCE  (Binary Cross-Entropy): standard per-pixel penalty.
        Dice: handles the class imbalance — true edges are rare pixels, so BCE
              alone wouldn't force the model to find them all.  Dice directly
              maximises the overlap between predicted and true boundaries.
    """
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        # Fixed discrete Laplacian kernel — highlights class-boundary pixels
        # in the segmentation mask (Paper §3.3, GT boundary generation step).
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        # Learnable pyramid fusion kernel (Paper §3.3 / Fig. 6).
        # Combines boundary maps from three scales (1x, 1/2x, 1/4x) into a
        # single soft GT boundary mask.  Initialised with decreasing weights
        # [6/10, 3/10, 1/10] so full-resolution boundaries dominate.
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # --- Full-resolution GT boundary (Paper §3.3, Fig. 6 step 1) ---
        # Convolve the GT mask with the Laplacian to produce a raw response;
        # clamp to [0,inf] and threshold at 0.1 to get a binary boundary map.
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        # --- Multi-scale strided Laplacian responses (Paper §3.3 / Fig. 6) ---
        # Strided convolutions naturally capture boundaries at coarser scales,
        # providing scale-aware supervision for the boundary prediction.
        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        # Upsample coarse boundary maps back to full resolution for fusion
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        # Binarise each upsampled scale
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        # --- Pyramid fusion (Paper §3.3 / Fig. 6, learnable fuse_kernel) ---
        # Stack the three finest-scale boundary maps (1x, 1/2x, 1/4x) and fuse
        # them with the learned weighted 1-D convolution to produce the final
        # soft GT boundary mask used for supervision.
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)  # weighted sum across scales

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        # --- Detail loss = BCE + Dice (Paper §3.3, Eq. 3) ---
        # BCE handles per-pixel classification; Dice mitigates the heavy
        # class imbalance between boundary and non-boundary pixels.
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss,  dice_loss

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
                nowd_params += list(module.parameters())
        return nowd_params

if __name__ == '__main__':
    torch.manual_seed(15)
    with open('../cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
    lb_map = {el['id']: el['trainId'] for el in labels_info}

    img_path = 'data/gtFine/val/frankfurt/frankfurt_000001_037705_gtFine_labelIds.png'
    img = cv2.imread(img_path, 0)
 
    label = np.zeros(img.shape, np.uint8)
    for k, v in lb_map.items():
        label[img == k] = v

    img_tensor = torch.from_numpy(label).cuda()
    img_tensor = torch.unsqueeze(img_tensor, 0).type(torch.cuda.FloatTensor)
   

    detailAggregateLoss = DetailAggregateLoss()
    for param in detailAggregateLoss.parameters():
        print(param)

    bce_loss,  dice_loss = detailAggregateLoss(torch.unsqueeze(img_tensor, 0), img_tensor)
    print(bce_loss,  dice_loss)