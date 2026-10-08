"""End-to-end sanity checks:
 (1) for v in strictly convex position (sum 0), K_2(v) = conv{v_j + v_{j+1}} has exactly these 6 vertices in ccw order;
 (2) random doubly stochastic 6x6 matrices: whenever an eigenvector (left or right) of a non-real eigenvalue is in
     convex position, the eigenvalue lies in Pi_6."""
import numpy as np
from scipy.spatial import ConvexHull
rng = np.random.default_rng(3)

def convex_order(v):
    """return ccw order if the points are in strictly convex position, else None"""
    try:
        hull = ConvexHull(np.c_[v.real, v.imag])
    except Exception:
        return None
    if len(hull.vertices) != len(v):
        return None
    return list(hull.vertices)          # scipy returns ccw order in 2-D

bad1 = 0; n1 = 0
for _ in range(20000):
    v = rng.normal(size=6) + 1j * rng.normal(size=6); v -= v.mean()
    o = convex_order(v)
    if o is None: continue
    n1 += 1
    w = v[o]
    s = w + np.roll(w, -1)
    pairs = np.array([w[i] + w[j] for i in range(6) for j in range(i + 1, 6)])
    h = ConvexHull(np.c_[pairs.real, pairs.imag])
    hv = pairs[h.vertices]
    if len(hv) != 6 or not all(np.min(np.abs(hv - z)) < 1e-12 for z in s):
        bad1 += 1
print("(1) convex-position samples %d, K_2 vertex structure violations %d" % (n1, bad1))

def in_Pi6(lam, tol=1e-12):
    th = np.pi * (2 * np.arange(6) + 1) / 6
    return np.max(np.real(lam * np.exp(-1j * th))) / np.cos(np.pi / 6) <= 1 + tol

def random_ds():
    k = rng.integers(2, 9)
    w = rng.dirichlet(np.ones(k))
    A = sum(wi * np.eye(6)[rng.permutation(6)] for wi in w)
    return A

hits = 0; viol = 0
for _ in range(200000):
    A = random_ds()
    for M in (A, A.T):
        ev, V = np.linalg.eig(M)
        for i in range(6):
            if abs(ev[i].imag) < 1e-9: continue
            if convex_order(V[:, i]) is None: continue
            hits += 1
            if not in_Pi6(ev[i]):
                viol += 1; print("violation", ev[i])
print("(2) eigenvalues with convex-position eigenvector: %d, outside Pi_6: %d" % (hits, viol))
