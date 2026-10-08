"""Float branch-and-bound search for Proposition P_90; writes the cover (boxes + certificate codes).
Usage: python3 run_search.py [OUT.npz]      (default OUT = cover.npz; takes a few seconds)

The search itself is not part of the proof: its output is re-checked box by box with ball
arithmetic by verify_arb.py (and independently by independent_verify.py)."""
import sys
import time
import numpy as np
from bnb import *

out_name = sys.argv[1] if len(sys.argv) > 1 else "cover.npz"
lo0 = np.array([-1, 0, -1, 0, -1, -1.0]); hi0 = np.array([1, 1, 0, 0.875, 1, 1.0])
t0 = time.time()
leaves, failed, queue, infeasible = run(lo0, hi0, max_boxes=10**8, min_width=1e-6, verbose=False)
print("time %.1fs, failed %d, queue %d" % (time.time() - t0, sum(len(f) for f in failed), sum(len(q[0]) for q in queue)))
out = {}
for c, v in leaves.items():
    arr = np.concatenate(v) if v else np.zeros((0, 14))
    out["cert%d" % c] = arr
    print(CERT_NAMES[c], len(arr))
inf = np.concatenate(infeasible)
out["infeasible"] = inf
print("infeasible", len(inf))
np.savez_compressed(out_name, lo0=lo0, hi0=hi0, **out)
print("written", out_name)
# volume check: sum of box volumes (certified + infeasible) vs initial box
vol = 0.0
for c in CERT_NAMES:
    a = out["cert%d" % c]
    vol += np.prod(2 * a[:, 6:12], 1).sum()
vol += np.prod(2 * inf[:, 6:12], 1).sum()
print("volume covered %.15f  initial %.15f" % (vol, np.prod(hi0 - lo0)))
