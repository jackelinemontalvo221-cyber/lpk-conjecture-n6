"""Independent re-verification of the cover, written separately from verify_arb.py / bnb.py (imports neither).
First-order affine arithmetic with an interval remainder, using Arb balls (no centred form, no Gram matrices),
own angle enclosure, own potentials written from the lemma statements, own candidate logic.
For each box of the partition: accept if ANY infeasibility reason holds, if the recorded certificate holds,
or if ANY certificate type other than the edge witness holds on the box;
otherwise bisect (widest axis) up to MAXD levels and require every sub-box to be accepted.
Also re-does the tiling check independently (exact rational box arithmetic, sorted sweep-free method:
replay of bisection with the leaf dictionary; and an exact volume sum)."""
import sys, time
import numpy as np
from fractions import Fraction
from multiprocessing import Pool
from flint import arb, ctx

ctx.prec = 80
PI = arb.pi()
MAXD = int(sys.argv[2]) if len(sys.argv) > 2 else 10

def lt(a, b): return a < b
def amin(*xs):
    m = xs[0]
    for x in xs[1:]:
        if x < m: m = x
    return m
def amax(*xs):
    m = xs[0]
    for x in xs[1:]:
        if x > m: m = x
    return m

class I:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi): s.lo = lo; s.hi = hi
    @staticmethod
    def c(x):
        x = arb(x); return I(x.lower(), x.upper())
    def __add__(a, b):
        if not isinstance(b, I): b = I.c(b)
        return I((a.lo + b.lo).lower(), (a.hi + b.hi).upper())
    __radd__ = __add__
    def __neg__(a): return I(-a.hi, -a.lo)
    def __sub__(a, b):
        if not isinstance(b, I): b = I.c(b)
        return I((a.lo - b.hi).lower(), (a.hi - b.lo).upper())
    def __rsub__(a, b): return (-a) + b
    def __mul__(a, b):
        if not isinstance(b, I): b = I.c(b)
        ps = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return I(amin(*[p.lower() for p in ps]), amax(*[p.upper() for p in ps]))
    __rmul__ = __mul__
    def sq(a):
        if a.lo >= 0: return I((a.lo * a.lo).lower(), (a.hi * a.hi).upper())
        if a.hi <= 0: return I((a.hi * a.hi).lower(), (a.lo * a.lo).upper())
        m = amax(-a.lo, a.hi); return I(arb(0), (m * m).upper())


H = None          # half-widths of the current box (exact arbs), set by verts()
class AF:
    """affine form: c + sum_i l_i d_i + [rlo, rhi],  |d_i| <= H[i]  (c, l_i balls; rlo, rhi exact)"""
    __slots__ = ('c', 'l', 'r')
    def __init__(s, c, l, r): s.c = c; s.l = l; s.r = r
    @staticmethod
    def const(v): return AF(arb(v), [arb(0)] * 6, I(arb(0), arb(0)))
    def lin(s):
        t = arb(0)
        for li, hi_ in zip(s.l, H): t = t + li.abs_upper() * hi_
        return t.upper()
    def __add__(a, b):
        if not isinstance(b, AF): b = AF.const(b)
        return AF(a.c + b.c, [x + y for x, y in zip(a.l, b.l)], a.r + b.r)
    __radd__ = __add__
    def __neg__(a): return AF(-a.c, [-x for x in a.l], -a.r)
    def __sub__(a, b):
        if not isinstance(b, AF): b = AF.const(b)
        return a + (-b)
    def __rsub__(a, b): return (-a) + b
    def scale(a, k):            # k: arb ball scalar
        return AF(a.c * k, [x * k for x in a.l], a.r * I.c(k))
    def __mul__(a, b):
        if not isinstance(b, AF): return a.scale(arb(b))
        A = a.lin(); B = b.lin()
        if a is b:
            quad = I(arb(0), (A * A).upper())
        else:
            quad = I(-A, A) * I(-B, B)
        rem = quad + I.c(a.c) * b.r + I.c(b.c) * a.r + I(-A, A) * b.r + I(-B, B) * a.r + a.r * b.r
        return AF(a.c * b.c, [a.c * y + b.c * x for x, y in zip(a.l, b.l)], rem)
    __rmul__ = __mul__
    def sq(a): return a * a
    def rng(a):
        L = a.lin()
        return I((a.c - L + a.r.lo).lower(), (a.c + L + a.r.hi).upper())

SQ3H = I.c(arb(3).sqrt() / 2)           # sqrt(3)/2 as an interval
SQ3Q = I.c(arb(3).sqrt() / 4)

def n2(z): return (z[0].sq() + z[1].sq()).rng()
def cross(a, b): return (a[0] * b[1] - a[1] * b[0]).rng()
def dot(a, b): return (a[0] * b[0] + a[1] * b[1]).rng()
def sub(a, b): return (a[0] - b[0], a[1] - b[1])
SQ3H_B = arb(3).sqrt() / 2
def q(x, sj): return ((x[0].sq() + x[1].sq()).scale(SQ3H_B) - (x[0] * sj[1] - x[1] * sj[0])).rng()

def verts(lo, hi):
    global H
    m = [(arb(l) + arb(h)) / 2 for l, h in zip(lo, hi)]       # exact (dyadic, prec 80)
    H = [((arb(h) - arb(l)) / 2) for l, h in zip(lo, hi)]
    assert all(x.rad() == 0 for x in m + H)
    X = [AF(m[i], [arb(1) if j == i else arb(0) for j in range(6)], I(arb(0), arb(0))) for i in range(6)]
    one = AF.const(1); zero = AF.const(0)
    s = [(one, zero), (X[0], X[1]), (X[2], X[3]), (X[4], X[5])]
    s.append((-1 - X[2], -X[3]))
    s.append((-X[0] - X[4], -X[1] - X[5]))
    return s

def angle(c, d):
    """enclosure of atan2(c, d) for points with c > 0 (domain); None if undecidable"""
    if c.lo > 0:
        rs = [d.lo / c.lo, d.lo / c.hi, d.hi / c.lo, d.hi / c.hi]
        rlo = amin(*[r.lower() for r in rs]); rhi = amax(*[r.upper() for r in rs])
        return ((PI / 2 - rhi.atan()).lower(), (PI / 2 - rlo.atan()).upper())
    if d.lo > 0:
        qs = [c.lo / d.lo, c.lo / d.hi, c.hi / d.lo, c.hi / d.hi]
        qlo = amin(*[r.lower() for r in qs]); qhi = amax(*[r.upper() for r in qs])
        lo = qlo.atan().lower()
        return (lo if lo > 0 else arb(0), qhi.atan().upper())
    return None

# constants written from the lemma statements (1-based k = 1..6  ->  list index k-1)
CA = [-(k - 1) * PI / 6 for k in range(1, 7)]
CB = [c * PI / 3 for c in (0, 0, 1, 1, 2, 2)]
CC = [c * PI / 3 for c in (0, 1, 1, 2, 2, 3)]
T1 = (0, 2, 4); T2 = (1, 3, 5)                 # 1-based {1,3,5}, {2,4,6}

def infeasible(s):
    for k in range(1, 6):
        if n2(s[k]).lo > 1: return True
    for k in range(6):
        if cross(s[k], s[(k + 1) % 6]).hi < 0: return True
        e1 = sub(s[(k + 1) % 6], s[k]); e2 = sub(s[(k + 2) % 6], s[(k + 1) % 6])
        if cross(e1, e2).hi < 0: return True
        for j in range(6):
            if j not in (k, (k + 1) % 6) and cross(e1, sub(s[j], s[k])).hi < 0: return True
    tot = arb(0)
    for k in range(6):
        a = angle(cross(s[k], s[(k + 1) % 6]), dot(s[k], s[(k + 1) % 6]))
        if a is None: return False
        tot = tot + a[0]
    return tot > 2 * PI

def cands(s, which):
    N2 = [n2(z) for z in s]
    lg = []
    for k in range(6):
        if k == 0: lg.append((arb(0), arb(0))); continue
        lo = (arb(3).sqrt() / 4 * N2[k].lo.log()).lower() if N2[k].lo > 0 else None   # None = -infinity
        hi = (arb(3).sqrt() / 4 * N2[k].hi.log()).upper() if N2[k].hi > 0 else None
        if hi is None: return None
        lg.append((lo, hi))
    ang = []
    for k in range(5):
        a = angle(cross(s[k], s[k + 1]), dot(s[k], s[k + 1]))
        if a is None: return None
        ang.append(a)
    sign, C = {'A': (+1, CA), 'B': (-1, CB), 'C': (-1, CC)}[which]
    out = []
    for k in range(6):
        keep = True
        for j in range(6):
            if j == k or lg[j][0] is None: continue
            if k > j:
                Dlo = sum((ang[i][0] for i in range(j, k)), arb(0)); Dhi = sum((ang[i][1] for i in range(j, k)), arb(0))
                tlo, thi = Dlo, Dhi                    # theta_k - theta_j in [tlo, thi]
            else:
                Dlo = sum((ang[i][0] for i in range(k, j)), arb(0)); Dhi = sum((ang[i][1] for i in range(k, j)), arb(0))
                tlo, thi = -Dhi, -Dlo
            up = lg[k][1] - lg[j][0] + (thi / 2 if sign > 0 else -tlo / 2) + (C[k] - C[j])
            if up < 0:
                keep = False; break
        if keep: out.append(k)
    return out

def certified(s, kind, row):
    if kind == 0 or kind == 'any':
        ks = [int(row[12])] if kind == 0 else range(6)
        for k in ks:
            if all(q(s[k], s[j]).lo >= 0 for j in range(6) if j != k): return True
        if kind == 0: return False
    if kind == 6:
        k = int(row[12]); t = Fraction(float(row[13]))
        assert 0 <= t <= 1 and t.denominator <= 4096
        ta = arb(float(t)); ua = arb(float(1 - t))
        x = (s[k][0].scale(ua) + s[(k + 1) % 6][0].scale(ta), s[k][1].scale(ua) + s[(k + 1) % 6][1].scale(ta))
        return all(q(x, s[j]).lo >= 0 for j in range(6))
    for code, which in ((1, 'A'), (2, 'B'), (3, 'C')):
        if kind == code or kind == 'any':
            cs = cands(s, which)
            if cs is not None:
                if which == 'A': ok = all(q(s[k], s[(k + 3) % 6]).lo >= 0 for k in cs)
                elif which == 'B': ok = all(q(s[k], s[(k + 1) % 6]).lo >= 0 for k in cs if k in T1)
                else: ok = all(q(s[k], s[(k + 1) % 6]).lo >= 0 for k in cs if k in T2)
                if ok: return True
            if kind == code: return False
    for code, T, O in ((4, T1, T2), (5, T2, T1)):
        if kind == code or kind == 'any':
            N2 = {a: n2(s[a]) for a in T}
            cs = [a for a in T if not any(N2[a].hi < N2[b].lo for b in T if b != a)]
            if all(q(s[a], s[j]).lo >= 0 for a in cs for j in O): return True
            if kind == code: return False
    return False

def accept(lo, hi, kind, row, depth, stats):
    s = verts(lo, hi)
    if infeasible(s):
        stats['inf@%d' % depth] = stats.get('inf@%d' % depth, 0) + 1; return True
    if kind != 'inf' and certified(s, kind, row):
        stats['rec@%d' % depth] = stats.get('rec@%d' % depth, 0) + 1; return True
    if certified(s, 'any', row):
        stats['any@%d' % depth] = stats.get('any@%d' % depth, 0) + 1; return True
    if depth >= MAXD: return False
    ax = int(np.argmax(hi - lo)); mid = 0.5 * (lo[ax] + hi[ax])
    assert lo[ax] < mid < hi[ax]
    h1 = hi.copy(); h1[ax] = mid; l2 = lo.copy(); l2[ax] = mid
    return accept(lo, h1, kind, row, depth + 1, stats) and accept(l2, hi, kind, row, depth + 1, stats)

def work(job):
    rows, kind = job
    stats = {}; bad = []
    for r in rows:
        m, h = r[:6].astype(float), r[6:12].astype(float)
        lo, hi = m - h, m + h
        # exactness of the float box endpoints (needed since endpoints are used as exact arbs)
        for i in range(6):
            assert Fraction(lo[i]) == Fraction(m[i]) - Fraction(h[i]) and Fraction(hi[i]) == Fraction(m[i]) + Fraction(h[i])
        if not accept(lo, hi, kind, r, 0, stats):
            bad.append(r)
    return stats, bad

if __name__ == "__main__":
    t0 = time.time()
    d = np.load(sys.argv[1])
    LO0 = np.array([-1, 0, -1, 0, -1, -1.0]); HI0 = np.array([1, 1, 0, 0.875, 1, 1.0])   # hard-coded claim
    assert np.array_equal(d['lo0'], LO0) and np.array_equal(d['hi0'], HI0), "initial box in file differs from claim"
    jobs = []; R = []
    for code in range(7):
        a = d['cert%d' % code]; R.append(a[:, :12])
        jobs += [(a[i:i + 200], code) for i in range(0, len(a), 200)]
    a = d['infeasible']; R.append(a[:, :12]); jobs += [(a[i:i + 200], 'inf') for i in range(0, len(a), 200)]
    R = np.concatenate(R)
    # independent tiling check: exact volume + pairwise-disjoint interiors via bisection replay from hard-coded box
    leaves = {}
    for r in R:
        key = tuple(Fraction(float(v)) for v in r[:12]); assert key not in leaves; leaves[key] = 1
    vol = sum(np.prod([2 * Fraction(float(x)) for x in r[6:12]]) for r in R)
    V0 = np.prod([Fraction(float(b)) - Fraction(float(a)) for a, b in zip(LO0, HI0)])
    stack = [tuple(Fraction(float(v)) for v in (LO0 + HI0) / 2) + tuple(Fraction(float(v)) for v in (HI0 - LO0) / 2)]
    used = 0
    while stack:
        node = stack.pop()
        if node in leaves: used += 1; continue
        mc, hc = list(node[:6]), list(node[6:])
        ax = max(range(6), key=lambda i: (hc[i], -i))
        assert hc[ax] > Fraction(1, 10 ** 7), "not a tiling"
        hn = hc.copy(); hn[ax] = hc[ax] / 2
        for sgn in (-1, 1):
            mn = mc.copy(); mn[ax] = mc[ax] + sgn * hn[ax]
            stack.append(tuple(mn) + tuple(hn))
    print("independent tiling: boxes %d, exact volume ok %s, replay used %d of %d" % (len(R), vol == V0, used, len(leaves)), flush=True)
    with Pool(2) as pool:
        tot = {}; bad = []
        for st, b in pool.imap_unordered(work, jobs):
            for k, v in st.items(): tot[k] = tot.get(k, 0) + v
            bad += b
    print("independent verification: unresolved boxes %d (MAXD=%d), time %.0fs" % (len(bad), MAXD, time.time() - t0))
    print("accept stats (reason@depth):", dict(sorted(tot.items())))
    if bad: np.save('indep_unresolved.npy', np.array(bad))
