import torch
import torch.nn as nn
from torch.nn import init
import math



class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    """
    STDC Module — additive variant ("STDC-Add", Paper §3.1 / Fig. 2b).

    This is an alternative version of the STDC building block kept for
    comparison in the paper's ablation study (Table 3).

    How it works:
      - Each sub-block processes the features and halves the number of channels
        (e.g. 256 → 128 → 64 → 32 → 32 …)
      - All sub-block outputs are glued together (concatenated) along the
        channel dimension
      - Then the original input is ADDED back in (like a residual/skip connection
        in a standard ResNet block) — that's the "Add" in the name.

    The paper shows that CatBottleneck (no addition, pure concat) performs
    better — see Table 3.
    """
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            # Strided STDC block (Paper Fig. 2c): depthwise 3×3 conv handles the
            # spatial downsampling; the input skip uses a depthwise conv + 1×1 proj
            # so dimensions match for the final element-wise addition.
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        # Build the sequence of sub-blocks with geometrically decreasing channel widths:
        # block_0: in -> out//2  (1×1, channel reduction)
        # block_1: out//2 -> out//4  (3×3, spatial processing)
        # block_2: out//4 -> out//8  ...
        # block_N-1: out//2^(N-1) -> out//2^(N-1)  (last block keeps width, Paper Eq. 1)
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        # Concatenate all sub-block outputs then add the skip — STDC-Add variant
        return torch.cat(out_list, dim=1) + x



class CatBottleneck(nn.Module):
    """
    STDC Module — the core building block of the STDC backbone (Paper §3.1 / Fig. 2a).
    "STDC" stands for Short-Term Dense Concatenate.

    THE KEY IDEA (in plain English):
    Traditional networks like ResNet apply the same operation at each layer and
    then add the input back.  The STDC block instead processes features in a
    chain of steps, each halving the channel count, and then stacks ALL those
    intermediate outputs together:

      Input → [Step 1: 1×1 conv, out//2 channels]
                    ↓
               [Step 2: 3×3 conv, out//4 channels]
                    ↓
               [Step 3: 3×3 conv, out//8 channels]
                    ↓  ...
               [Step N: 3×3 conv, same channels as step N-1]

      Output = stack(Step1, Step2, Step3, ..., StepN) along channel dimension
               → total channels = out  (Paper Eq. 1)

    Why is this good?
      - Early steps capture fine-grained detail (small receptive field).
      - Later steps capture broader context (large receptive field).
      - By stacking them all we get multi-scale info in one block WITHOUT
        needing two separate network branches like the original BiSeNet.
      - The channel halving means later steps are cheap to compute, so the
        whole block is fast despite capturing multiple scales.
    """
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            # Strided STDC block (Paper Fig. 2c): depthwise 3×3 conv on B_1 handles
            # downsampling; the first-block output is also average-pooled to align
            # spatial dimensions for the final concatenation.
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            # AvgPool skips the first output so all pieces share the downsampled resolution
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        # Geometrically decreasing channel widths per sub-block (Paper Eq. 1)
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        # B_1: 1×1 channel reduction (largest receptive-field contribution)
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    # Apply strided depthwise conv before feeding into next sub-block
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                # Each subsequent sub-block operates on the previous sub-block's output,
                # forming the sequential "short-term dense" feature hierarchy (Paper §3.1)
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            # Align B_1 spatial size with the rest before concatenation
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        # Final concatenation produces multi-scale features in a single tensor (Paper Eq. 1)
        out = torch.cat(out_list, dim=1)
        return out

# ---------------------------------------------------------------------------
# STDC2 Backbone  (Paper §3.1 / Table 1 — "STDC2-Seg")
# ---------------------------------------------------------------------------
# This is the LARGER and more accurate of the two STDC backbone options.
# It has more STDC blocks per stage (layers=[4,5,3]) compared to STDC1.
#
# Think of the backbone as a funnel:
#   - Input image (e.g. 1024×512) enters at full resolution.
#   - Each stage halves the spatial size while increasing the number of feature channels.
#   - The five outputs (feat2 … feat32) are like snapshots at different zoom levels:
#       feat2  (1/2  resolution) → very fine detail → used for edge/boundary detection
#       feat4  (1/4  resolution) → fine detail       → used for edge/boundary detection
#       feat8  (1/8  resolution) → mid-level detail  → main input to Feature Fusion Module
#       feat16 (1/16 resolution) → context            → Context Path ARM
#       feat32 (1/32 resolution) → coarse context     → Context Path ARM + global pool
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4,5,3], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        # Stage splits expose multi-scale feature maps for the segmentation head.
        # feat2 (1/2) and feat4 (1/4) → Detail boundary heads (Paper §3.3)
        # feat8 (1/8)                  → Spatial path input to FFM  (Paper §3.2)
        # feat16 (1/16) & feat32 (1/32) → Context Path ARM inputs    (Paper §3.2)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        # Stem: two strided 3×3 convs downsample to 1/4 resolution
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        # Returns 5 feature maps at 1/2, 1/4, 1/8, 1/16, 1/32 of input resolution
        # (Paper Fig. 3 — STDC2 backbone output stages)
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
           feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

# ---------------------------------------------------------------------------
# STDC1 Backbone  (Paper §3.1 / Table 1 — "STDC1-Seg")
# ---------------------------------------------------------------------------
# This is the SMALLER and faster of the two STDC backbones.
# It has fewer blocks per stage (layers=[2,2,2]) so it runs faster but with
# slightly lower accuracy.  The paper reports it achieves state-of-the-art
# speed on Cityscapes (Table 4).  Architecture is identical to STDCNet1446
# and exposes the same five multi-scale outputs (feat2 … feat32).
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2,2,2], block_num=4, type="cat", num_classes=1000, dropout=0.20, pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base*16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        # Stage splits — same semantics as STDCNet1446 but shallower per stage
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
           feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
    model.eval()
    x = torch.randn(1,3,224,224)
    y = model(x)
    torch.save(model.state_dict(), 'cat.pth')
    print(y.size())
