import torch
import torch.nn.functional as F

from models.sbg import SBG

torch.manual_seed(0)

B, C, H, W = 2, 256, 96, 192
N_CLASSES = 19

CONFIGS = [
    ("Arm B: appearance only",        dict(use_variance=False, use_semantic=False)),
    ("Arm C: + variance",             dict(use_variance=True,  use_semantic=False)),
    ("Arm D: full SBG (+ semantic)",  dict(use_variance=True,  use_semantic=True)),
]


def run_config(name, kwargs):
    print(f"\n=== {name}  ({kwargs}) ===")

    sbg = SBG(feat_chan=C, n_classes=N_CLASSES, **kwargs)

    F_d = torch.randn(B, C, H, W, requires_grad=True)
    L_c = torch.randn(B, N_CLASSES, H, W) if kwargs["use_semantic"] else None

    print(f"F_d shape: {tuple(F_d.shape)}")
    if L_c is not None:
        print(f"L_c shape: {tuple(L_c.shape)}")

    refined, A_logit = sbg(F_d, L_c)

    assert refined.shape == F_d.shape, \
        f"refined.shape {tuple(refined.shape)} != F_d.shape {tuple(F_d.shape)}"
    assert A_logit.shape == (B, 1, H, W), \
        f"A_logit.shape {tuple(A_logit.shape)} != {(B, 1, H, W)}"

    lo, hi = A_logit.min().item(), A_logit.max().item()
    print(f"refined.shape: {tuple(refined.shape)}")
    print(f"A_logit.shape: {tuple(A_logit.shape)}")
    print(f"A_logit min/max: {lo:.4f} / {hi:.4f}")
    assert not (0.0 <= lo and hi <= 1.0), (
        "A_logit is confined to [0,1] -- looks double-sigmoided "
        "(fuse output should be raw logits, not already squashed)"
    )

    target = torch.randint(0, 2, A_logit.shape).float()
    loss = F.binary_cross_entropy_with_logits(A_logit, target)
    loss.backward()

    assert F_d.grad is not None, "F_d.grad is None -- no gradient reached F_d"
    grad_abs_sum = F_d.grad.abs().sum().item()
    print(f"loss: {loss.item():.4f}  |  F_d.grad.abs().sum(): {grad_abs_sum:.4f}")
    assert grad_abs_sum > 0, "F_d.grad is all zero -- gradient did not flow to F_d"

    print(f"{name}: PASSED")


if __name__ == "__main__":
    for name, kwargs in CONFIGS:
        run_config(name, kwargs)
    print("\nAll SBG configs passed.")
