"""Independent numerical test of the analytic lemmas (automatic conditions, equal areas,
Lemma T, Lemmas A, B, C) on random hexagons of the class, with random cyclic relabelling and rotation.
Written from the lemma statements (1-based in the statement; here 0-based index i = k-1)."""
import numpy as np, time
rng = np.random.default_rng(271828)
S3 = np.sqrt(3.0)
W = np.exp(2j * np.pi / 3)
cr = lambda a, b: (np.conj(a) * b).imag

def valid(s):
    e = np.roll(s, -1, 1) - s
    ok = (cr(e, np.roll(e, -1, 1)) > 0).all(1) & (cr(s, np.roll(s, -1, 1)) > 0).all(1)
    ok &= np.abs(np.angle(np.roll(s, -1, 1) / s).sum(1) - 2 * np.pi) < 1e-9
    return ok

def make(z):
    return np.c_[z[:, 0], z[:, 1], z[:, 2], z[:, 3], -z[:, 0] - z[:, 2], -z[:, 1] - z[:, 3]]

def gen_gauss(n):
    out = []
    while sum(len(o) for o in out) < n:
        N = 400000
        z = (rng.normal(size=(N, 4)) + 1j * rng.normal(size=(N, 4))) * 10 ** rng.uniform(-2.5, 0, size=(N, 4))
        s = make(z); out.append(s[valid(s)])
    return np.concatenate(out)[:n]

def gen_near(base, n, smin=-9, smax=-0.5):
    out = []
    while sum(len(o) for o in out) < n:
        N = 200000
        d = make(rng.normal(size=(N, 4)) + 1j * rng.normal(size=(N, 4)))
        s = base[None] + (10 ** rng.uniform(smin, smax, N))[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
        out.append(s[valid(s)])
    return np.concatenate(out)[:n]

def gen_cutcorner(n):
    """triangle (1,w,w^2) with all corners cut (two vertices near each corner) + generic perturbation"""
    out = []
    A = np.array([1, W, W * W])
    while sum(len(o) for o in out) < n:
        N = 200000
        al = 10 ** rng.uniform(-8, -0.7, N)[:, None]; be = al * 10 ** rng.uniform(-1, 1, N)[:, None]
        s = np.c_[A[0] + al[:, 0] * (A[2] - A[0]), A[0] + be[:, 0] * (A[1] - A[0]),
                  A[1] + al[:, 0] * (A[0] - A[1]), A[1] + be[:, 0] * (A[2] - A[1]),
                  A[2] + al[:, 0] * (A[1] - A[2]), A[2] + be[:, 0] * (A[0] - A[2])]
        d = make(rng.normal(size=(N, 4)) + 1j * rng.normal(size=(N, 4)))
        s = s + (al * 10 ** rng.uniform(-6, 0, N)[:, None]) * d / np.linalg.norm(d, axis=1)[:, None]
        out.append(s[valid(s)])
    return np.concatenate(out)[:n]

def gen_uniform_normalised(n):
    lo0 = np.array([-1, 0, -1, 0, -1, -1.0]); hi0 = np.array([1, 1, 0, 0.875, 1, 1.0])
    out = []
    while sum(len(o) for o in out) < n:
        X = lo0 + (hi0 - lo0) * rng.random((400000, 6))
        s = np.c_[np.ones(len(X)), X[:, 0] + 1j * X[:, 1], X[:, 2] + 1j * X[:, 3], X[:, 4] + 1j * X[:, 5],
                  -1 - X[:, 2] - 1j * X[:, 3], -X[:, 0] - X[:, 4] - 1j * (X[:, 1] + X[:, 5])]
        out.append(s[valid(s) & (np.abs(s) <= 1).all(1)])
    return np.concatenate(out)[:n]

t0 = time.time()
sets = {
    'gauss-multiscale': gen_gauss(100000),
    'uniform-normalised': gen_uniform_normalised(50000),
    'near-regular': gen_near(np.exp(1j * np.pi / 3 * np.arange(6)), 30000),
    'near (1,1,w,w,w2,w2)': gen_near(np.array([1, 1, W, W, W * W, W * W]), 20000),
    'near (1,w,w,w2,w2,1)': gen_near(np.array([1, W, W, W * W, W * W, 1]), 20000),
    'cut-corner triangles': gen_cutcorner(20000),
}
print("generated %s in %.1fs" % ({k: len(v) for k, v in sets.items()}, time.time() - t0), flush=True)

cB = np.array([0, 0, 1, 1, 2, 2]) * np.pi / 3          # 1-based k = 1..6
cC = np.array([0, 1, 1, 2, 2, 3]) * np.pi / 3
def cext(c, i):                                        # 0-based i, any integer; c_{i+6} = c_i + pi
    return c[i % 6] + np.pi * (i // 6)

TOL = 1e-12
grand = 0; total = 0
for name, s in sets.items():
    N = len(s)
    # random relabelling (cyclic shift) and rotation: lemma statements must not depend on them
    r = rng.integers(0, 6, N)
    s = np.take_along_axis(s, (r[:, None] + np.arange(6)[None]) % 6, 1) * np.exp(1j * rng.uniform(0, 2 * np.pi, N))[:, None]
    assert np.allclose(s[:, 0] + s[:, 2] + s[:, 4], 0, atol=1e-12 * np.abs(s).max()) and \
        np.allclose(s[:, 1] + s[:, 3] + s[:, 5], 0, atol=1e-12 * np.abs(s).max())
    I = np.arange(N)
    def ImR(k, d):                                    # Im(s_{k+d}/s_k), k array (0-based)
        return (s[I, (k + d) % 6] / s[I, k]).imag
    a = np.angle(np.roll(s, -1, 1) / s)               # a_i = arg(s_{i+1}/s_i) in (0, pi)
    th = np.angle(s[:, :1]) + np.c_[np.zeros(N), np.cumsum(a, 1)[:, :5]]
    lg = 0.5 * S3 * np.log(np.abs(s))
    viol = {}
    mins = {}
    def rec(key, slack):
        viol[key] = viol.get(key, 0) + int((slack < -TOL).sum())
        mins[key] = min(mins.get(key, np.inf), float(slack.min()))
    # automatic conditions, all k
    for k in range(6):
        kk = np.full(N, k)
        rec('auto j=k-1 (Im<0)', -ImR(kk, 5)); rec('auto j=k-2 (Im<0)', -ImR(kk, 4))
        # equal areas: cross(s_k, s_{k+2}) = cross(s_{k-2}, s_k)
        rec('equal areas |diff|', -np.abs(cr(s[:, k], s[:, (k + 2) % 6]) - cr(s[:, (k - 2) % 6], s[:, k])) /
            np.abs(s).max(1) ** 2 + 1e-13)
    # Lemma T: max-modulus vertex of each triangle, own-triangle conditions
    for T in ((0, 2, 4), (1, 3, 5)):
        T = np.array(T); k = T[np.argmax(np.abs(s[:, T]), 1)]
        rec('Lemma T (j=k+2)', S3 / 2 - ImR(k, 2)); rec('Lemma T (j=k+4)', S3 / 2 - ImR(k, 4))
    # Lemma A
    PA = lg + th / 2 - np.arange(6) * np.pi / 6
    # periodicity check Psi_{k+6} = Psi_k with theta_{k+6} = theta_k + 2pi, const -(k+6)pi/6
    k = np.argmax(PA, 1)
    rec('A: j=k+1', S3 / 2 - ImR(k, 1)); rec('A: j=k+2', S3 / 2 - ImR(k, 2))
    full_A = np.minimum.reduce([S3 / 2 - ImR(k, d) for d in (1, 2, 4, 5)])
    # Lemmas B and C: generic statement (all j in k+1..k+3 with c_j - c_k >= pi/3), then witness claim
    for lab, c, Tw in (('B', cB, 1), ('C', cC, 0)):
        P = lg - th / 2 + c
        k = np.argmax(P, 1)
        for dd in (1, 2, 3):
            off = cext(c, k + dd) - cext(c, k)
            sel = off >= np.pi / 3 - 1e-12
            rec('%s: j=k+%d where c_j-c_k>=pi/3' % (lab, dd), np.where(sel, S3 / 2 - ImR(k, dd), np.inf))
        # claimed witness: k in T2 (0-based odd) for B, k in T1 (0-based even) for C
        isw = (k % 2 == Tw)
        wit = np.minimum.reduce([S3 / 2 - ImR(k, d) for d in range(1, 6)])
        rec('%s: argmax in T%d is full witness' % (lab, 2 if lab == 'B' else 1), np.where(isw, wit, np.inf))
        oth = np.minimum.reduce([S3 / 2 - ImR(k, d) for d in (2, 3, 4, 5)])
        rec('%s: argmax in other triangle: all but j=k+1' % lab, np.where(~isw, oth, np.inf))
        # offsets table sanity: which d have offset < pi/3
    nv = sum(viol.values()); grand += nv; total += N
    print("%-22s N=%6d  violations=%d" % (name, N, nv))
    print("    min slack: " + ", ".join("%s: %.2e" % (kk_, v) for kk_, v in mins.items() if np.isfinite(v)), flush=True)

# offset table (units of pi/3) for the record
for lab, c in (('B', cB), ('C', cC)):
    print(lab, "offsets c_{k+d}-c_k (units pi/3), rows k=1..6 (1-based), cols d=1,2,3:",
          [[round((cext(c, k + d) - cext(c, k)) / (np.pi / 3), 6) for d in (1, 2, 3)] for k in range(6)])
print("TOTAL hexagons %d, total violations %d, time %.1fs" % (total, grand, time.time() - t0))
