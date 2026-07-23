import torch

# 4 channels, width 6 (H=1). Rows = channels, columns = spatial positions.
channels = [
    [6, 2, 2, 8, 8, 8],
    [6, 3, 3, 8, 8, 8],
    [6, 1, 1, 8, 8, 8],
    [6, 2, 2, 8, 8, 8],
]
F_d = torch.tensor(channels, dtype=torch.float32).unsqueeze(0).unsqueeze(2)  # [1, 4, 1, 6]

# Exact Branch-1 formula from models/sbg.py:41-46
F_bar = F_d.mean(dim=1, keepdim=True)
var = F_d.var(dim=1, keepdim=True, unbiased=False)
mu = var.mean(dim=(2, 3), keepdim=True)
sd = var.std(dim=(2, 3), keepdim=True, unbiased=False)
z = (var - mu) / (sd + 1e-6)
U = torch.sigmoid(z)

pos = list(range(6))
var_row = var.view(-1).tolist()
z_row = z.view(-1).tolist()
U_row = U.view(-1).tolist()

print("position:  ", pos)
print("var (unbiased=False, /N):", [round(v, 4) for v in var_row])
print("z:         ", [round(v, 4) for v in z_row])
print("U=sigmoid(z):", [round(v, 4) for v in U_row])

print(f"\nmu (mean of var over space): {mu.item():.4f}")
print(f"sd (std of var over space):  {sd.item():.4f}")

agree_positions = [i for i, v in enumerate(var_row) if abs(v) < 1e-6]
disagree_positions = [i for i, v in enumerate(var_row) if abs(v) >= 1e-6]
print(f"\nagree positions {agree_positions}: U = {[round(U_row[i], 4) for i in agree_positions]}")
print(f"disagree positions {disagree_positions}: U = {[round(U_row[i], 4) for i in disagree_positions]}")
assert min(U_row[i] for i in disagree_positions) > max(U_row[i] for i in agree_positions), \
    "U is not elevated at disagreement positions relative to agreement positions"
print("\nConfirmed: U is elevated exactly at the positions where channels disagree.")
