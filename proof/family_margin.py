"""The P_90 margin vanishes identically on the 120-degree symmetric family s_{k+2} = w s_k, and grows
linearly under generic class-preserving perturbations."""
import numpy as np
from p90core import *
rng = np.random.default_rng(0)
# 1) margin on the 120-degree symmetric family
vals = []
for _ in range(4000):
    r = rng.uniform(0.2, 1.6); al = rng.uniform(0.01, 2 * np.pi / 3 - 0.01)
    s = sym_family(r, al)
    if not is_convex(s):
        continue
    vals.append((margin(s), r, al))
vals = np.array(vals)
print("family members tested:", len(vals), " margin min/max: %.3e %.3e" % (vals[:, 0].min(), vals[:, 0].max()))
# 2) perturb family members along random class-preserving directions
def rand_dir():
    d = rng.normal(size=4) + 1j * rng.normal(size=4)
    return np.array([d[0], d[1], d[2], d[3], -d[0] - d[2], -d[1] - d[3]])
worst = []
for trial in range(300):
    while True:
        r = rng.uniform(0.3, 1.5); al = rng.uniform(0.05, 2 * np.pi / 3 - 0.05)
        s0 = sym_family(r, al)
        if is_convex(s0) and convexity(s0)[0] > 1e-3:
            break
    d = rand_dir(); d /= np.linalg.norm(d)
    row = []
    for e in (1e-1, 1e-2, 1e-3, 1e-4):
        s = s0 + e * d
        row.append(margin(s) if is_convex(s) else np.nan)
    worst.append((r, al, *row))
worst = np.array(worst)
print("perturbed: min margin per eps:", np.nanmin(worst[:, 2:], axis=0))
print("ratio margin/eps (min over trials):", np.nanmin(worst[:, 2:] / np.array([1e-1, 1e-2, 1e-3, 1e-4]), axis=0))
neg = worst[np.nanmin(worst[:, 2:], axis=1) < -1e-12]
print("trials with negative margin:", len(neg))
for row in neg[:10]:
    print("  r=%.3f alpha=%.1fdeg" % (row[0], np.degrees(row[1])), row[2:])
