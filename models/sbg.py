import torch
import torch.nn as nn
import torch.nn.functional as F


class SBG(nn.Module):
    """Semantic Boundary Gate, with per-branch flags for stepwise ablation.

    use_variance : enable Branch 1 (channel-variance gate). If False, R=U=1 (no gating).
    use_semantic : enable Branch 3 (semantic edge from logits). If False, only appearance.
    Always active: Branch 2 (appearance edge) — it's the baseline arm.
    """

    def __init__(self, feat_chan=256, n_classes=19,
                 use_variance=True, use_semantic=True, detach_logits=True):
        super().__init__()  # zero-arg form: rename-proof
        self.use_variance = use_variance
        self.use_semantic = use_semantic
        self.detach_logits = detach_logits

        # Fixed Gaussian blur for the "blur and subtract" edge trick (not learned).
        g = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
        g = g.reshape(1, 1, 3, 3)
        self.register_buffer('gauss1', g.clone())
        self.register_buffer('gaussK', g.repeat(n_classes, 1, 1, 1))

        # Learnable pieces (tiny).
        self.conv_det = nn.Conv2d(1, 1, 3, padding=1, bias=True)  # appearance edge
        if use_semantic:
            self.conv_sem = nn.Conv2d(n_classes, 1, 1, bias=True)  # semantic edge

        # Fuse: 1 input channel if appearance-only, 2 if semantic is on.
        fuse_in = 2 if use_semantic else 1
        self.conv_fuse = nn.Conv2d(fuse_in, 1, 1, bias=True)

        nn.init.constant_(self.conv_fuse.bias, -2.0)  # start gate near "do little"

    def forward(self, F_d, L_c=None):
        # --- Branch 1: channel variance -> confusion weight U (and R = 1 - U) ---
        if self.use_variance:
            F_bar = F_d.mean(dim=1, keepdim=True)
            var = F_d.var(dim=1, keepdim=True, unbiased=False)
            mu = var.mean(dim=(2, 3), keepdim=True)
            sd = var.std(dim=(2, 3), keepdim=True, unbiased=False)
            z = (var - mu) / (sd + 1e-6)
            U = torch.sigmoid(z)
            R = 1.0 - U
        else:
            F_bar = F_d.mean(dim=1, keepdim=True)
            U = torch.ones_like(F_bar)  # no gating: everything trusted equally
            R = torch.ones_like(F_bar)

        # --- Branch 2: appearance edge, gated by reliability R ---
        F_hf = F_bar - F.conv2d(F_bar, self.gauss1, padding=1)
        A_det = torch.sigmoid(self.conv_det(F_hf.abs()))
        gated = [A_det * R]

        # --- Branch 3: semantic edge, gated by confusion U (if enabled) ---
        if self.use_semantic:
            assert L_c is not None, "semantic branch on but no logits passed"
            if self.detach_logits:
                L_c = L_c.detach()
            K = L_c.shape[1]
            L_hf = L_c - F.conv2d(L_c, self.gaussK, padding=1, groups=K)
            A_sem = torch.sigmoid(self.conv_sem(L_hf.abs()))
            gated = [A_sem * U, A_det * R]

        # --- Fuse in LOGIT space (never sigmoid before the loss) ---
        A_logit = self.conv_fuse(torch.cat(gated, dim=1))
        A = torch.sigmoid(A_logit)
        return F_d * (1.0 + A), A_logit

    def get_params(self):
        wd, nowd = [], []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                wd.append(m.weight)
                if m.bias is not None:
                    nowd.append(m.bias)
        return wd, nowd
