#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


from nets.stdcnet import STDCNet1446, STDCNet813
from modules.bn import InPlaceABNSync as BatchNorm2d
# BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module (ARM) — Paper §3.2 / Fig. 4.

    What problem does this solve?
    The deep layers of the backbone produce very coarse (1/16 and 1/32 scale)
    but semantically rich feature maps.  Some channels in those maps are more
    useful than others.  The ARM learns to WEIGHT each channel automatically:

      1. Take the feature map from the backbone at a coarse scale.
      2. Squeeze it to a single number per channel (global average pool).
      3. Pass through a small 1×1 conv + sigmoid to get a weight between 0 and 1
         for each channel — think of it as "how important is this channel?".
      4. Multiply the original feature map by those weights channel-by-channel.

    The result is a refined feature map that emphasises the most useful channels
    and suppresses the noisy ones, helping the Context Path capture better
    semantic context without adding much compute.
    """
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        # self.bn_atten = BatchNorm2d(out_chan)
        self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        # Global average pool -> 1x1 attention weights (ARM, Paper §3.2)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        # Element-wise channel recalibration of the feature map
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class PlaneAuxHead(nn.Module):
    def __init__(self, in_chan, mid_chan=64, n_classes=21, zeroplane_model=None, zeroplane_soft_target_fn=None, *args, **kwargs):
        super(PlaneAuxHead, self).__init__()
        # Match ZeroPlane sem_seg output shape: [num_plane_slots + non_plane, H, W].
        self.pred_head = BiSeNetOutput(in_chan, mid_chan, n_classes)
        self.zeroplane_model = zeroplane_model
        self.zeroplane_soft_target_fn = zeroplane_soft_target_fn
        self._fwd_call_count = 0  # used to gate verbose per-iter logs
        print('[PlaneAuxHead.__init__] in_chan={}, mid_chan={}, n_classes={}, '
              'zeroplane_model={}, zeroplane_soft_target_fn={}'.format(
              in_chan, mid_chan, n_classes,
              type(zeroplane_model).__name__ if zeroplane_model is not None else None,
              type(zeroplane_soft_target_fn).__name__ if zeroplane_soft_target_fn is not None else None),
              flush=True)

        if isinstance(self.zeroplane_model, nn.Module):
            self.zeroplane_model.eval()
            print('[PlaneAuxHead.__init__] zeroplane_model set to eval()', flush=True)
            # for param in self.zeroplane_model.parameters():
            #     param.requires_grad = False
    
    def _predict_zeroplane_soft_target(self, image=None, zeroplane_inputs=None, pred_logits=None):
        _verbose = (self._fwd_call_count <= 1)
        if self.zeroplane_soft_target_fn is not None:
            if _verbose:
                print('[PlaneAuxHead._predict] calling zeroplane_soft_target_fn '
                      '(image shape={})'.format(
                      tuple(image.shape) if torch.is_tensor(image) else None), flush=True)
            soft_target = self.zeroplane_soft_target_fn(
                image=image,
                zeroplane_inputs=zeroplane_inputs,
                pred_logits=pred_logits,
            )
            if soft_target is None:
                if _verbose:
                    print('[PlaneAuxHead._predict] zeroplane_soft_target_fn returned None', flush=True)
                return None
            if torch.is_tensor(soft_target):
                if soft_target.ndim == 2:
                    soft_target = soft_target.unsqueeze(0).unsqueeze(0)
                elif soft_target.ndim == 3:
                    soft_target = soft_target.unsqueeze(0)
                soft_target = torch.clamp(soft_target.float(), 0.0, 1.0)
                if _verbose:
                    print('[PlaneAuxHead._predict] soft_target shape={} min={:.4f} max={:.4f}'.format(
                          tuple(soft_target.shape), soft_target.min().item(), soft_target.max().item()),
                          flush=True)
                return soft_target
            if _verbose:
                print('[PlaneAuxHead._predict] soft_target_fn returned non-tensor: {}'.format(type(soft_target)), flush=True)
            return None

        if self.zeroplane_model is None:
            if _verbose:
                print('[PlaneAuxHead._predict] no soft_target_fn and no zeroplane_model — returning None', flush=True)
            return None

        model_inputs = zeroplane_inputs
        if model_inputs is None and torch.is_tensor(image):
            model_inputs = [{"image": img} for img in image]

        if model_inputs is None:
            if _verbose:
                print('[PlaneAuxHead._predict] model_inputs is None — returning None', flush=True)
            return None

        if _verbose:
            print('[PlaneAuxHead._predict] calling zeroplane_model with {} inputs'.format(len(model_inputs)), flush=True)
        outputs = self.zeroplane_model(model_inputs)
        if torch.is_tensor(outputs):
            plane = outputs.float()
            if plane.ndim == 2:
                plane = plane.unsqueeze(0).unsqueeze(0)
            elif plane.ndim == 3:
                plane = plane.unsqueeze(0)
            plane = torch.clamp(plane, 0.0, 1.0)
            if _verbose:
                print('[PlaneAuxHead._predict] zeroplane_model output shape={}'.format(tuple(plane.shape)), flush=True)
            return plane
        if _verbose:
            print('[PlaneAuxHead._predict] zeroplane_model returned non-tensor: {}'.format(type(outputs)), flush=True)
        return None

    def forward(self, feat, image=None, zeroplane_inputs=None):
        self._fwd_call_count += 1
        _verbose = (self._fwd_call_count <= 2)
        if _verbose:
            print('[PlaneAuxHead.forward] call #{} feat.shape={}'.format(
                  self._fwd_call_count, tuple(feat.shape)), flush=True)
        pred_logits = self.pred_head(feat)
        if _verbose:
            print('[PlaneAuxHead.forward] pred_logits.shape={}'.format(tuple(pred_logits.shape)), flush=True)
        soft_target = None

        with torch.no_grad():
            if _verbose:
                print('[PlaneAuxHead.forward] entering _predict_zeroplane_soft_target...', flush=True)
            soft_target = self._predict_zeroplane_soft_target(
                image=image,
                zeroplane_inputs=zeroplane_inputs,
                pred_logits=pred_logits,
            )

            if soft_target is not None:
                soft_target = soft_target.to(pred_logits.device, dtype=pred_logits.dtype)
                if soft_target.shape[-2:] != pred_logits.shape[-2:]:
                    if _verbose:
                        print('[PlaneAuxHead.forward] resizing soft_target {} -> {}'.format(
                              tuple(soft_target.shape[-2:]), tuple(pred_logits.shape[-2:])), flush=True)
                    soft_target = F.interpolate(
                        soft_target,
                        size=pred_logits.shape[-2:],
                        mode='bilinear',
                        align_corners=True,
                    )
                soft_target = torch.clamp(soft_target, 0.0, 1.0)
                if _verbose:
                    print('[PlaneAuxHead.forward] final soft_target shape={} min={:.4f} max={:.4f}'.format(
                          tuple(soft_target.shape), soft_target.min().item(), soft_target.max().item()),
                          flush=True)
            else:
                if _verbose:
                    print('[PlaneAuxHead.forward] soft_target is None', flush=True)

        return pred_logits, soft_target

    def get_params(self):
        return self.pred_head.get_params()

class ContextPath(nn.Module):
    """
    Context Path — Paper §3.2 / Fig. 4.

    WHAT IT DOES IN PLAIN ENGLISH:
    To correctly label a pixel (e.g. "this pixel is part of a car") the network
    needs context — it needs to "look around" at the broader scene.  The Context
    Path is the part of the network responsible for gathering that big-picture
    understanding.  It does this in three steps:

      Step 1 — Global average pool on the coarsest feature map (1/32 scale):
               Squashes the entire image into a single vector.  This gives the
               model a rough idea of the whole scene ("there is sky, road, cars").

      Step 2 — ARM at 1/32 scale:
               Refines the very coarse features, guided by the global context
               from Step 1 which is added in before the ARM.

      Step 3 — ARM at 1/16 scale:
               Refines the 1/16 features, guided by the already-refined 1/32
               features passed down from Step 2.

    The final output (feat_cp8, 1/8 scale) is handed to the Feature Fusion Module
    where it is combined with the fine spatial detail from the backbone.
    """
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        # Multi-scale feature maps from the STDC backbone (Paper §3.1 / Fig. 3)
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        
        # --- Global context embedding (Paper §3.2) ---
        # Global average pool on the coarsest feature map gives a context vector
        # that encodes full-image semantic information (1×1 spatial resolution).
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)          # project to 128 channels
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')  # broadcast back to 1/32

        # --- ARM at 1/32 scale + top-down fusion (Paper §3.2) ---
        feat32_arm = self.arm32(feat32)   # channel-attention refinement of 1/32 features
        feat32_sum = feat32_arm + avg_up  # fuse with global context
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')  # upsample to 1/16
        feat32_up = self.conv_head32(feat32_up)

        # --- ARM at 1/16 scale + top-down fusion (Paper §3.2) ---
        feat16_arm = self.arm16(feat16)   # channel-attention refinement of 1/16 features
        feat16_sum = feat16_arm + feat32_up  # fuse with refined 1/32 context
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')  # upsample to 1/8
        feat16_up = self.conv_head16(feat16_up)  # = feat_cp8 fed into FFM
        
        # feat_cp16 (feat32_up) → auxiliary seg head for deep supervision (Paper §3.2)
        # feat_cp8  (feat16_up) → primary input to Feature Fusion Module (Paper §3.2)
        # feat32 (raw 1/32 backbone stage) is also returned for optional taps
        # such as plane_aux_tap='res32'.
        return feat2, feat4, feat8, feat16, feat16_up, feat32_up, feat32 # x8, x16, x32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM) — Paper §3.2 / Fig. 5.

    WHAT IT DOES IN PLAIN ENGLISH:
    At this point the network has two streams of information:
      - "fsp" (spatial path): the raw 1/8-scale backbone output.
        Rich in fine spatial detail (edges, textures) but lacks broad context.
      - "fcp" (context path): the refined output from the Context Path (also 1/8).
        Rich in semantic context (what objects are present) but less spatially sharp.

    The FFM merges them smartly:
      1. Concatenate both along the channel dimension.
      2. Apply a conv to reduce the combined channels to a single feature map.
      3. Use SE-style (Squeeze-and-Excitation) attention:
           • Squeeze: global average pool → one number per channel.
           • Excite:  small network learns which channels to amplify or suppress.
      4. Multiply the attention weights back in and ADD a residual connection.

    The result is a single feature map that combines the best of both streams.
    """
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        # Concatenate spatial-path (shallow backbone) and context-path features (Paper §3.2)
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)          # unify channel dimension
        # SE-style channel attention: squeeze via global avg-pool, then excite
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)          # compress: out -> out//4
        atten = self.relu(atten)
        atten = self.conv2(atten)          # restore: out//4 -> out
        atten = self.sigmoid(atten)        # per-channel weights in [0,1]
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat       # residual connection preserves base features
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class  BiSeNet(nn.Module):
    """
    STDC-Seg: the complete segmentation network (Paper §3 / Fig. 4).

    OVERVIEW:
    The paper's main contribution is removing BiSeNet's expensive Detail Branch
    at inference while still keeping sharp, edge-aware predictions.
    The trick: teach the network about edges DURING training only, via the
    boundary prediction heads, then throw those heads away at inference.

    The network has five parts:

      1. ContextPath  — STDC backbone + ARM top-down path (Paper §3.2)
                         Extracts both fine spatial detail AND broad scene context.

      2. FeatureFusionModule (FFM)  — smartly merges the two streams (Paper §3.2)
                                       outputs a single 1/8-scale feature map.

      3. conv_out (main head)  — the actual segmentation output used at inference.
                                  Takes the FFM output and predicts a class label
                                  for every pixel, then upsamples to full resolution.

      4. conv_out16 / conv_out32 (auxiliary heads)  — additional segmentation
                                  outputs used ONLY during training to help the
                                  network learn better intermediate features
                                  ("deep supervision", Paper §3.2).

      5. conv_out_sp2/4/8/16 (boundary heads)  — predict raw boundary edges
                                  from the shallow backbone features.  Used ONLY
                                  during training in the Detail Aggregation Loss.
                                  At inference these are simply not called, so
                                  there is ZERO extra runtime cost (Paper §3.3).
    """
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, use_plane_aux=False, plane_aux_tap='fuse', plane_aux_mid=64, zeroplane_model=None, zeroplane_soft_target_fn=None, plane_aux_soft_target_only_debug=False, *args, **kwargs):
        super(BiSeNet, self).__init__()
        
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        self.use_plane_aux = use_plane_aux
        self.plane_aux_tap = plane_aux_tap
        self.plane_aux_soft_target_only_debug = plane_aux_soft_target_only_debug
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)
            
        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 256)
        
        # --- Primary + auxiliary segmentation heads (Paper §3.2, deep supervision) ---
        # conv_out   : main head applied to FFM output (1/8 fused features)
        # conv_out16 : auxiliary head on feat_cp8  (Context Path 1/8 output)
        # conv_out32 : auxiliary head on feat_cp16 (Context Path 1/16 output)
        # All three are upsampled to full resolution and supervised with cross-entropy.
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        # --- Detail boundary prediction heads (Paper §3.3, training-only) ---
        # These lightweight heads predict a binary boundary map from shallow backbone
        # feature maps (1/2 to 1/16 resolution).  They are supervised by the
        # DetailAggregateLoss (BCE + Dice on Laplacian-derived GT boundaries).
        # They replace the full Detail Branch of BiSeNet at inference time —
        # only the STDC backbone stages producing feat_res2/4/8/16 are needed.
        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)

        if self.use_plane_aux:
            plane_inplanes_map = {
                'fuse': 256,
                'cp8': conv_out_inplanes,
                'cp16': conv_out_inplanes,
                'res8': sp8_inplanes,
                'res16': sp16_inplanes,
                'res32': 1024,
            }
            if self.plane_aux_tap not in plane_inplanes_map:
                raise ValueError('Unsupported plane_aux_tap {}. Choose from {}'.format(
                    self.plane_aux_tap, list(plane_inplanes_map.keys())
                ))

            self.plane_aux_head = PlaneAuxHead(
                plane_inplanes_map[self.plane_aux_tap],
                mid_chan=plane_aux_mid,
                n_classes=1,  # binary: plane vs non-plane
                zeroplane_model=zeroplane_model,
                zeroplane_soft_target_fn=zeroplane_soft_target_fn,
            )

        self.init_weight()

    def forward(self, x, zeroplane_inputs=None):
        H, W = x.size()[2:]
        
        # ContextPath returns STDC backbone stages (feat_res*) and refined context (feat_cp*)
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16, feat_res32 = self.cp(x)

        # --- Detail boundary predictions (Paper §3.3, Detail Aggregation Learning) ---
        # These are passed to DetailAggregateLoss during training only.
        # feat_res2/4/8/16 are the raw STDC backbone outputs at 1/2 to 1/16 resolution.
        feat_out_sp2 = self.conv_out_sp2(feat_res2)

        feat_out_sp4 = self.conv_out_sp4(feat_res4)
  
        feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        # --- Feature Fusion Module: spatial (1/8) + context (1/8) -> fused (Paper §3.2) ---
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        # --- Segmentation heads (Paper §3.2 deep supervision) ---
        feat_out = self.conv_out(feat_fuse)       # primary head on fused features
        feat_out16 = self.conv_out16(feat_cp8)    # auxiliary head at 1/8 context resolution
        feat_out32 = self.conv_out32(feat_cp16)   # auxiliary head at 1/16 context resolution

        # Upsample all segmentation outputs to full input resolution
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        plane_aux_logits = None
        plane_aux_soft_target = None
        if self.use_plane_aux:
            if self.plane_aux_tap == 'fuse':
                plane_feat = feat_fuse
            elif self.plane_aux_tap == 'cp8':
                plane_feat = feat_cp8
            elif self.plane_aux_tap == 'cp16':
                plane_feat = feat_cp16
            elif self.plane_aux_tap == 'res8':
                plane_feat = feat_res8
            elif self.plane_aux_tap == 'res32':
                plane_feat = feat_res32
            else:
                plane_feat = feat_res16
            if self.plane_aux_head._fwd_call_count < 2:
                print('[BiSeNet.forward] plane_aux_tap={} plane_feat.shape={}'.format(
                      self.plane_aux_tap, tuple(plane_feat.shape)), flush=True)

            plane_aux_logits, plane_aux_soft_target = self.plane_aux_head(
                plane_feat,
                image=x,
                zeroplane_inputs=zeroplane_inputs,
            )
            if self.plane_aux_head._fwd_call_count <= 2:
                print('[BiSeNet.forward] after plane_aux_head: logits={}, soft_target={}'.format(
                      tuple(plane_aux_logits.shape) if plane_aux_logits is not None else None,
                      tuple(plane_aux_soft_target.shape) if plane_aux_soft_target is not None else None),
                      flush=True)
            if self.plane_aux_soft_target_only_debug:
                plane_aux_logits = None
            else:
                plane_aux_logits = F.interpolate(plane_aux_logits, (H, W), mode='bilinear', align_corners=True)
            if plane_aux_soft_target is not None and plane_aux_soft_target.shape[-2:] != (H, W):
                plane_aux_soft_target = F.interpolate(
                    plane_aux_soft_target,
                    (H, W),
                    mode='bilinear',
                    align_corners=True,
                )
            if self.plane_aux_soft_target_only_debug:
                if plane_aux_soft_target is None:
                    print('[plane_aux_soft_target] None')
                else:
                    flat = plane_aux_soft_target.detach().reshape(-1)
                    sample = flat[:8].cpu().tolist()
                    print(
                        '[plane_aux_soft_target] '
                        'shape={}, dtype={}, device={}, min={:.6f}, max={:.6f}, mean={:.6f}, sample(first_8)={}'.format(
                            tuple(plane_aux_soft_target.shape),
                            plane_aux_soft_target.dtype,
                            plane_aux_soft_target.device,
                            plane_aux_soft_target.min().item(),
                            plane_aux_soft_target.max().item(),
                            plane_aux_soft_target.mean().item(),
                            sample,
                        )
                    )


        # Return segmentation outputs + boundary outputs selected by training flags.
        # At inference only feat_out (the primary head) is used; boundary heads are dropped.
        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            if self.use_plane_aux:
                return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8, plane_aux_logits, plane_aux_soft_target
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
        
        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            if self.use_plane_aux:
                return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8, plane_aux_logits, plane_aux_soft_target
            return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            if self.use_plane_aux:
                return feat_out, feat_out16, feat_out32, feat_out_sp8, plane_aux_logits, plane_aux_soft_target
            return feat_out, feat_out16, feat_out32, feat_out_sp8
        
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            if self.use_plane_aux:
                return feat_out, feat_out16, feat_out32, plane_aux_logits, plane_aux_soft_target
            return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput, PlaneAuxHead)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    
    net = BiSeNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

    
