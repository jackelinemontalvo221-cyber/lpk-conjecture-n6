"""Search for points where every certificate type (potentials A, B, C, Lemma T, witnesses) has margin <= 0.
Usage: python3 trouble_search.py SEED NRUNS TOL   (found none: best value about 0.03)"""
import numpy as np, cma, sys
from certs import *
TOL = float(sys.argv[3])
def unpack(x):
    return np.array([1, x[0] + 1j * x[1], x[2] + 1j * x[3], x[4] + 1j * x[5], -1 - x[2] - 1j * x[3], -x[0] - x[4] - 1j * (x[1] + x[5])])
def obj(x):
    s = unpack(x)
    pen = max(0, np.max(abs(s) ** 2) - 1)
    m, w = convexity(s)
    ang = np.angle(np.roll(s, -1) / s)
    pen += max(0, -m) + max(0, -ang.min()) + abs(w - 1)
    if pen > 0:
        return 1 + pen
    return best_cert(s, TOL)[0]
seed = int(sys.argv[1]); rng = np.random.default_rng(seed)
from p90core import gaussian_class
S = gaussian_class(5000, rng)
S = np.array([np.roll(z, -np.argmax(np.abs(z))) for z in S]); S = S / S[:, :1]
out = []
for trial in range(int(sys.argv[2])):
    s0 = S[rng.integers(len(S))]
    x0 = np.r_[s0[1].real, s0[1].imag, s0[2].real, s0[2].imag, s0[3].real, s0[3].imag]
    es = cma.CMAEvolutionStrategy(x0, 0.15, {"verbose": -9, "maxfevals": 2500, "seed": seed * 1000 + trial + 1})
    es.optimize(obj)
    s = unpack(es.result.xbest)
    out.append((es.result.fbest, s))
out.sort(key=lambda r: r[0])
for f, s in out[:10]:
    c = best_cert(s, TOL)[1]
    print("best=%.3e " % f + " ".join("%s=%.2e" % kv for kv in c.items()), np.round(s, 3))
