"""Consistency check of the cover against random feasible hexagons (float, point-wise)."""
import numpy as np
from bnb import CERT_NAMES, s_from_x
from certs import potentials, cond, cross_margin
from p90core import is_convex, qmin, S3
from p90core import gaussian_class
d = np.load("cover.npz")
boxes = []; kinds = []
for c in CERT_NAMES:
    a = d["cert%d" % c]; boxes.append(a); kinds += [c] * len(a)
inf = d["infeasible"]
allb = np.concatenate([np.c_[b[:, :12], b[:, 12:14]] for b in boxes] + [np.c_[inf[:, :12], inf[:, 12:13], np.zeros(len(inf))]])
kinds = np.array(kinds + [-1] * len(inf))
LO = allb[:, :6] - allb[:, 6:12]; HI = allb[:, :6] + allb[:, 6:12]
rng = np.random.default_rng(5)
# feasible normalised samples: gaussian class, relabel so that the max-modulus vertex is s1, scale/rotate to 1
S = gaussian_class(20000, rng)
X = []
for s in S:
    k = np.argmax(np.abs(s)); s = np.roll(s, -k); s = s / s[0]
    X.append([s[1].real, s[1].imag, s[2].real, s[2].imag, s[3].real, s[3].imag])
X = np.array(X)
X = np.concatenate([X, X[:0]])
bad = 0; checked = 0
for x in X:
    inside = np.where(((LO <= x) & (x <= HI)).all(1))[0]
    if len(inside) == 0:
        print("NOT COVERED", x); bad += 1; continue
    for i in inside:
        kind = kinds[i]; s = s_from_x(x[None])[0]
        if not is_convex(s): continue
        checked += 1
        if kind == -1:
            print("feasible point in infeasible box", x, allb[i, 12]); bad += 1
        elif kind == 0:
            k = int(allb[i, 12]); ok = qmin(s, s[k]) >= 0
            if not ok: print("W-vertex claim fails", x); bad += 1
        elif kind == 6:
            k = int(allb[i, 12]); t = allb[i, 13]; xx = (1 - t) * s[k] + t * s[(k + 1) % 6]
            if qmin(s, xx) < 0: print("W-edge claim fails", x); bad += 1
        else:
            PA, PB, PC = potentials(s)
            if kind == 1:
                k = np.argmax(PA); ok = cond(s, k, 3) >= 0
            elif kind == 2:
                k = np.argmax(PB); ok = (k % 2 == 1) or cond(s, k, 1) >= 0
            elif kind == 3:
                k = np.argmax(PC); ok = (k % 2 == 0) or cond(s, k, 1) >= 0
            elif kind in (4, 5):
                T = [0, 2, 4] if kind == 4 else [1, 3, 5]
                k = T[np.argmax(np.abs(s[T]))]; ok = cross_margin(s, k) >= 0
            if not ok: print("cert", CERT_NAMES[kind], "claim fails at", x); bad += 1
            # and the final conclusion: vertex k must be a full witness
            if qmin(s, s[k]) < -1e-12: print("LEMMA FAILS", CERT_NAMES[kind], x); bad += 1
print("samples %d, (point,box) checks %d, problems %d" % (len(X), checked, bad))
