"""Rigorous re-verification of the cover produced by bnb.py, using ball arithmetic (python-flint / Arb).

Every box [m - h, m + h] of cover.npz (m, h are exact binary floating-point numbers) is re-checked:
  * infeasible boxes: the recorded domain constraint is violated on the whole box;
  * certified boxes: the recorded certificate (W-vertex, W-edge, A, B, C, L1, L2) holds on the whole box.
All polynomial bounds use the centred form with exact coefficients (sqrt(3) as an Arb ball);
log and atan are evaluated in Arb.  A box passes only if every inequality is certain.
The coefficient matrices are computed in float64; every one of them is compared entry by entry with the same
matrix computed in exact rational arithmetic (fractions.Fraction) before it is used.
Also checks that the boxes tile the initial box (replay of the bisection tree and exact dyadic volume)."""
import sys, time
import numpy as np
from fractions import Fraction
from multiprocessing import Pool
from flint import arb, ctx

ctx.prec = 96
SQ3 = arb(3).sqrt()
PI = arb.pi()
ZERO = arb(0)

# ------------------------------------------------------------------ exact quadratic forms G = G0 + sqrt3*G1
def affine_maps():
    M = np.zeros((6, 2, 7))
    M[0, 0, 6] = 1.0
    M[1, 0, 0] = 1; M[1, 1, 1] = 1
    M[2, 0, 2] = 1; M[2, 1, 3] = 1
    M[3, 0, 4] = 1; M[3, 1, 5] = 1
    M[4] = -M[0] - M[2]
    M[5] = -M[1] - M[3]
    return M


MAPS = affine_maps()
J = np.array([[0.0, 1.0], [-1.0, 0.0]])
sym = lambda A: 0.5 * (A + A.T)
Gn2 = lambda Ma: Ma.T @ Ma
Gdot = lambda Ma, Mb: sym(Ma.T @ Mb)
Gcr = lambda Ma, Mb: sym(Ma.T @ J @ Mb)


# exact rational copies, used only to certify that the float64 coefficient matrices are exact
def _frac(A):
    return np.array([Fraction(float(x)) for x in np.ravel(A)], dtype=object).reshape(np.shape(A))


MAPS_Q = [_frac(M) for M in MAPS]
J_Q = _frac(J)
symq = lambda A: (A + A.T) / 2
Gn2q = lambda Ma: Ma.T.dot(Ma)
Gdotq = lambda Ma, Mb: symq(Ma.T.dot(Mb))
Gcrq = lambda Ma, Mb: symq(Ma.T.dot(J_Q).dot(Mb))


def same(Gf, Gq):
    """True iff the float matrix Gf equals the rational matrix Gq entry by entry"""
    return np.shape(Gf) == np.shape(Gq) and all(Fraction(float(a)) == b for a, b in zip(np.ravel(Gf), np.ravel(Gq)))


def poly(G0, G1=None, Q0=None, Q1=None):
    """sparse exact representation: list of (i, j, arb coefficient) of y^T (G0 + sqrt3*G1) y.
    Q0, Q1: the same matrices in exact rational arithmetic; the float matrices must agree with them exactly."""
    if G1 is None:
        G1 = np.zeros_like(G0)
    assert Q0 is not None and same(G0, Q0), "coefficient matrix not exact"
    if Q1 is not None:
        assert same(G1, Q1), "coefficient matrix not exact"
    else:
        assert not np.any(G1), "coefficient matrix not exact"
    out = []
    for i in range(7):
        for j in range(7):
            if G0[i, j] != 0 or G1[i, j] != 0:
                out.append((i, j, arb(float(G0[i, j])) + SQ3 * arb(float(G1[i, j]))))
    return out


def Pq(Mx, Mj, Mxq, Mjq):
    # q_j(x) = (sqrt3/2)|x|^2 - cross(x, s_j);  Mxq, Mjq: exact rational copies of Mx, Mj
    return poly(-Gcr(Mx, Mj), 0.5 * Gn2(Mx), -Gcrq(Mxq, Mjq), Gn2q(Mxq) / 2)


Mq = MAPS_Q
P_N2 = [poly(Gn2(MAPS[k]), None, Gn2q(Mq[k])) for k in range(6)]
P_CR = [poly(Gcr(MAPS[k], MAPS[(k + 1) % 6]), None, Gcrq(Mq[k], Mq[(k + 1) % 6])) for k in range(6)]
P_DT = [poly(Gdot(MAPS[k], MAPS[(k + 1) % 6]), None, Gdotq(Mq[k], Mq[(k + 1) % 6])) for k in range(6)]
P_TU = [poly(Gcr(MAPS[(k + 1) % 6] - MAPS[k], MAPS[(k + 2) % 6] - MAPS[(k + 1) % 6]), None,
             Gcrq(Mq[(k + 1) % 6] - Mq[k], Mq[(k + 2) % 6] - Mq[(k + 1) % 6])) for k in range(6)]
EV_LIST = [(k, j) for k in range(6) for j in range(6) if j not in (k, (k + 1) % 6)]
P_EV = [poly(Gcr(MAPS[(k + 1) % 6] - MAPS[k], MAPS[j] - MAPS[k]), None,
             Gcrq(Mq[(k + 1) % 6] - Mq[k], Mq[j] - Mq[k])) for k, j in EV_LIST]
P_QV = [[Pq(MAPS[k], MAPS[j], Mq[k], Mq[j]) for j in range(6)] for k in range(6)]
_PQ_EDGE = {}          # cache of edge-witness polynomials, keyed by (k, t)

# exactness of the float matrices used above (all entries are small dyadic rationals)
for M in MAPS:
    assert np.all(M * 4 == np.round(M * 4))


def bound(P, m, h):
    """rigorous [lo, hi] (exact arb endpoints) of y^T G y over the box, y = (x, 1)."""
    y = m + [arb(1)]
    Gy = [ZERO] * 7
    diag = {}
    absoff = ZERO
    for (i, j, g) in P:
        Gy[i] = Gy[i] + g * y[j]
        if i < 6 and j < 6:
            if i == j:
                diag[i] = g
            else:
                absoff = absoff + g.abs_upper() * h[i] * h[j]
    p = ZERO
    for i in range(7):
        p = p + y[i] * Gy[i]
    lin = ZERO
    for i in range(6):
        lin = lin + 2 * Gy[i].abs_upper() * h[i]
    dlo = ZERO; dhi = ZERO
    for i, g in diag.items():
        gl, gu = g.lower(), g.upper()
        if gl < 0:
            dlo = dlo + gl * h[i] * h[i]
        if gu > 0:
            dhi = dhi + gu * h[i] * h[i]
    lo = (p - lin + dlo - absoff).lower()
    hi = (p + lin + dhi + absoff).upper()
    return lo, hi


def iangle(c, d):
    """interval of the angle atan2(c, d) in [0, pi) for points of the domain; None if undecidable"""
    clo, chi = c; dlo, dhi = d
    if clo > 0:
        r_lo = (dlo / chi) if dlo >= 0 else (dlo / clo)
        r_hi = (dhi / clo) if dhi >= 0 else (dhi / chi)
        return ((PI / 2 - r_hi.atan()).lower(), (PI / 2 - r_lo.atan()).upper())
    if dlo > 0:
        q_lo = (clo / dhi) if clo >= 0 else (clo / dlo)
        q_hi = (chi / dlo) if chi >= 0 else (chi / dhi)
        lo = q_lo.atan().lower()
        if lo < 0:
            lo = ZERO
        return (lo, q_hi.atan().upper())
    return None


CA = [-(k * PI) / 6 for k in range(6)]
CB = [(c * PI) / 3 for c in (0, 0, 1, 1, 2, 2)]
CC = [(c * PI) / 3 for c in (0, 1, 1, 2, 2, 3)]
T1, T2 = (0, 2, 4), (1, 3, 5)


def potentials_data(m, h):
    n2 = [bound(P_N2[k], m, h) for k in range(6)]
    if any(not (n2[k][1] > 0) for k in range(6)):
        return None
    # lower bound None means -infinity (|s_k| may be arbitrarily small on the box)
    lg = [(((SQ3 / 4) * n2[k][0].log()) if n2[k][0] > 0 else None, (SQ3 / 4) * n2[k][1].log()) for k in range(6)]
    lg[0] = (ZERO, ZERO)
    ang = []
    for k in range(5):
        a = iangle(bound(P_CR[k], m, h), bound(P_DT[k], m, h))
        if a is None:
            return None
        ang.append(a)
    return lg, ang


def candidates(lg, ang, sign, C):
    cum_lo = [ZERO]; cum_hi = [ZERO]
    for a in ang:
        cum_lo.append(cum_lo[-1] + a[0]); cum_hi.append(cum_hi[-1] + a[1])
    cand = []
    for k in range(6):
        ok = True
        for j in range(6):
            if j == k:
                continue
            if k > j:
                dlo, dhi = cum_lo[k] - cum_lo[j], cum_hi[k] - cum_hi[j]
            else:
                dlo, dhi = -(cum_hi[j] - cum_hi[k]), -(cum_lo[j] - cum_lo[k])
            t_hi = dhi / 2 if sign > 0 else -dlo / 2
            if lg[j][0] is None:          # Psi_j unbounded below: j cannot exclude k
                continue
            U = lg[k][1] - lg[j][0] + t_hi + (C[k] - C[j])
            if U < 0:            # certainly Psi_k < Psi_j on the box
                ok = False
                break
        if ok:
            cand.append(k)
    return cand


def qlo(k, j, m, h):
    return bound(P_QV[k][j], m, h)[0]


def check_box(row, kind):
    m = [arb(float(v)) for v in row[:6]]
    h = [arb(float(v)) for v in row[6:12]]
    if kind == "inf":
        rid = int(row[12])
        if rid < 6:
            return bound(P_N2[rid], m, h)[0] > 1
        if 10 <= rid < 16:
            return bound(P_TU[rid - 10], m, h)[1] < 0
        if 20 <= rid < 26:
            return bound(P_CR[rid - 20], m, h)[1] < 0
        if 100 <= rid < 100 + len(P_EV):
            return bound(P_EV[rid - 100], m, h)[1] < 0
        if rid == 200:
            tot = ZERO
            for k in range(6):
                a = iangle(bound(P_CR[k], m, h), bound(P_DT[k], m, h))
                if a is None:
                    return False
                tot = tot + a[0]
            return tot > 2 * PI
        return False
    code = kind
    if code == 0:                                             # vertex witness s_k
        k = int(row[12])
        assert 0 <= k <= 5
        return all(qlo(k, j, m, h) >= 0 for j in range(6) if j != k)
    if code == 6:                                             # edge witness (1-t)s_k + t s_{k+1}
        k = int(row[12]); t = float(row[13])
        assert 0 <= k <= 5
        assert t * 4096 == round(t * 4096) and 0 <= t <= 1
        if (k, t) not in _PQ_EDGE:
            Mx = (1 - t) * MAPS[k] + t * MAPS[(k + 1) % 6]
            tq = Fraction(t)
            Mxq = (1 - tq) * Mq[k] + tq * Mq[(k + 1) % 6]
            assert same(Mx, Mxq), "edge point not exact"
            _PQ_EDGE[(k, t)] = [Pq(Mx, MAPS[j], Mxq, Mq[j]) for j in range(6)]
        return all(bound(P, m, h)[0] >= 0 for P in _PQ_EDGE[(k, t)])
    if code in (1, 2, 3):
        pdata = potentials_data(m, h)
        if pdata is None:
            return False
        lg, ang = pdata
        # an empty candidate set cannot occur for a box containing a domain point (the argmax exists);
        # we treat it as a failure to stay on the safe side
        if code == 1:
            cand = candidates(lg, ang, +1, CA)
            return len(cand) > 0 and all(qlo(k, (k + 3) % 6, m, h) >= 0 for k in cand)
        if code == 2:
            cand = candidates(lg, ang, -1, CB)
            return len(cand) > 0 and all(qlo(k, (k + 1) % 6, m, h) >= 0 for k in cand if k in T1)
        cand = candidates(lg, ang, -1, CC)
        return len(cand) > 0 and all(qlo(k, (k + 1) % 6, m, h) >= 0 for k in cand if k in T2)
    if code in (4, 5):
        T, O = (T1, T2) if code == 4 else (T2, T1)
        n2 = {k: bound(P_N2[k], m, h) for k in T}
        cand = [a for a in T if all(not (n2[a][1] < n2[b][0]) for b in T if b != a)]
        return len(cand) > 0 and all(qlo(a, j, m, h) >= 0 for a in cand for j in O)
    return False


def work(args):
    rows, kind = args
    bad = []
    for r in rows:
        if not check_box(r, kind):
            bad.append(r)
    return len(rows), bad


def self_test(n=300, seed=0):
    """random boxes and random points: values of all polynomials must lie in the rigorous bounds"""
    rng = np.random.default_rng(seed)
    allP = P_N2 + P_CR + P_DT + P_TU + P_EV + [p for row in P_QV for p in row]
    for _ in range(n):
        mm = rng.uniform(-1, 1, 6); hh = rng.uniform(1e-4, 0.3, 6)
        mm = np.round(mm * 2**20) / 2**20; hh = np.round(hh * 2**20) / 2**20
        m = [arb(float(v)) for v in mm]; h = [arb(float(v)) for v in hh]
        X = mm + hh * rng.choice([-1.0, 1.0, 0.3, -0.7], size=(8, 6))
        for P in allP:
            lo, hi = bound(P, m, h)
            for x in X:
                y = [arb(float(v)) for v in x] + [arb(1)]
                val = ZERO
                for (i, j, g) in P:
                    val = val + y[i] * g * y[j]
                # failure only if the value is certainly outside the computed enclosure
                assert not (val < lo) and not (val > hi), "enclosure self-test failed"
    return True


if __name__ == "__main__":
    t0 = time.time()
    print("enclosure self-test:", self_test(), flush=True)
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else "cover.npz")
    # the initial box is fixed here (it contains the normalised domain, see the write-up); the data file must agree
    lo0 = np.array([-1.0, 0.0, -1.0, 0.0, -1.0, -1.0]); hi0 = np.array([1.0, 1.0, 0.0, 0.875, 1.0, 1.0])
    assert np.array_equal(d["lo0"], lo0) and np.array_equal(d["hi0"], hi0), "initial box mismatch"
    import flint
    print("python-flint", flint.__version__, " precision", ctx.prec, "bits", flush=True)
    jobs = []
    allrows = []
    for code in range(7):
        a = d["cert%d" % code]
        allrows.append(a[:, :12])
        for s in range(0, len(a), 500):
            jobs.append((a[s:s + 500], code))
    inf = d["infeasible"]
    allrows.append(inf[:, :12])
    for s in range(0, len(inf), 500):
        jobs.append((inf[s:s + 500], "inf"))
    # exact tiling check: every box is a dyadic sub-box of the initial box obtained by bisection;
    # compare total volume exactly (Fractions) and check containment.
    R = np.concatenate(allrows)
    vol = Fraction(0)
    for r in R:
        v = Fraction(1)
        for i in range(6):
            v *= 2 * Fraction(float(r[6 + i]))
            assert Fraction(float(r[i])) - Fraction(float(r[6 + i])) >= Fraction(float(lo0[i]))
            assert Fraction(float(r[i])) + Fraction(float(r[6 + i])) <= Fraction(float(hi0[i]))
        vol += v
    V0 = Fraction(1)
    for i in range(6):
        V0 *= Fraction(float(hi0[i])) - Fraction(float(lo0[i]))
    print("boxes:", len(R), " exact volume equal to initial box:", vol == V0, flush=True)
    # tiling by replaying the bisection tree (same rule as bnb.run: split the first widest axis)
    leafset = {}
    for r in R:
        key = tuple(float(v) for v in r[:12])
        leafset[key] = leafset.get(key, 0) + 1
    assert max(leafset.values()) == 1, "duplicate boxes"
    stack = [(tuple(float(v) for v in (lo0 + hi0) / 2), tuple(float(v) for v in (hi0 - lo0) / 2))]
    used = 0
    while stack:
        mc, hc = stack.pop()
        key = mc + hc
        if key in leafset:
            used += 1
            continue
        ax = int(np.argmax(np.array(hc)))
        hn = list(hc); hn[ax] = hc[ax] / 2
        off = hn[ax]
        m1 = list(mc); m1[ax] = mc[ax] - off
        m2 = list(mc); m2[ax] = mc[ax] + off
        assert hn[ax] > 1e-7, "replay went too deep: boxes do not tile"
        stack.append((tuple(m1), tuple(hn))); stack.append((tuple(m2), tuple(hn)))
    print("tiling replay: leaves used %d of %d -> %s" % (used, len(leafset), used == len(leafset)), flush=True)
    with Pool(2) as pool:
        total = 0; bad = []
        for n, b in pool.imap_unordered(work, jobs):
            total += n; bad += b
    print("verified boxes: %d, failures: %d, time %.0fs" % (total, len(bad), time.time() - t0))
    if bad:
        np.save("verify_failures.npy", np.array(bad))
