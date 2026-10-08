"""Branch-and-bound cover of the normalised domain for Proposition P_90 (search phase, float64).

Parameters x = (Re s2, Im s2, Re s3, Im s3, Re s4, Im s4);  s1 = 1, s5 = -1 - s3, s6 = -s2 - s4.
Domain D: |s_k| <= 1 (s1 is a vertex of maximal modulus), all turns >= 0, 0 inside (consecutive
cross products >= 0), total angle 2*pi.

Every quantity used is a quadratic polynomial in x, written as y^T G y with y = (x, 1) and a
symmetric 7x7 matrix G; on a box (centre m, half widths h) it is bounded by the centred form
    p(m) +- sum_i |g_i| h_i  + (second order term computed from G[:6,:6]).
Transcendental quantities (log |s_k|, angles) enter only through the three potentials.

Certificates for a box (any one suffices):
  W  : a witness point x on dH (vertex or edge point with fixed t) with q_j(x) > 0 for all j
  A  : every possible argmax k of PsiA has q_{k+3}(s_k) >= 0            (Lemma A)
  B  : every possible argmax k of PsiB lying in T1 has q_{k+1}(s_k) >= 0  (Lemma B)
  C  : every possible argmax k of PsiC lying in T2 has q_{k+1}(s_k) >= 0  (Lemma C)
  L1 : every possible max-modulus vertex k of T1 has q_j(s_k) >= 0 for j in T2  (Lemma T)
  L2 : same with T1, T2 exchanged
where q_j(x) = (sqrt3/2)|x|^2 - cross(x, s_j).
"""
import numpy as np

S3 = np.sqrt(3.0)
T1 = (0, 2, 4)
T2 = (1, 3, 5)
CA = -np.arange(6) * np.pi / 6
CB = np.array([0, 0, 1, 1, 2, 2]) * np.pi / 3
CC = np.array([0, 1, 1, 2, 2, 3]) * np.pi / 3


# ---------------------------------------------------------------- affine maps and quadratic forms
def affine_maps():
    """M[k] (2x7): (Re s_k, Im s_k) = M[k] @ (x, 1)."""
    M = np.zeros((6, 2, 7))
    M[0, 0, 6] = 1.0                                   # s1 = 1
    M[1, 0, 0] = 1; M[1, 1, 1] = 1                     # s2
    M[2, 0, 2] = 1; M[2, 1, 3] = 1                     # s3
    M[3, 0, 4] = 1; M[3, 1, 5] = 1                     # s4
    M[4] = -M[0] - M[2]                                # s5 = -s1 - s3
    M[5] = -M[1] - M[3]                                # s6 = -s2 - s4
    return M


MAPS = affine_maps()
J = np.array([[0.0, 1.0], [-1.0, 0.0]])


def sym(A):
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def G_norm2(Ma):
    return np.swapaxes(Ma, -1, -2) @ Ma


def G_dot(Ma, Mb):
    return sym(np.swapaxes(Ma, -1, -2) @ Mb)


def G_cross(Ma, Mb):
    """cross(a, b) = Re a Im b - Im a Re b = (Ma y)^T J (Mb y)"""
    return sym(np.swapaxes(Ma, -1, -2) @ J @ Mb)


def G_q(Mx, Mj):
    """q_j(x) = (sqrt3/2)|x|^2 - cross(x, s_j)"""
    return 0.5 * S3 * G_norm2(Mx) - G_cross(Mx, Mj)


# base polynomial families
G_N2 = np.array([G_norm2(MAPS[k]) for k in range(6)])                         # |s_k|^2
G_CR = np.array([G_cross(MAPS[k], MAPS[(k + 1) % 6]) for k in range(6)])     # cross(s_k, s_k+1)
G_DT = np.array([G_dot(MAPS[k], MAPS[(k + 1) % 6]) for k in range(6)])       # dot(s_k, s_k+1)
G_TURN = np.array([G_cross(MAPS[(k + 1) % 6] - MAPS[k], MAPS[(k + 2) % 6] - MAPS[(k + 1) % 6])
                   for k in range(6)])                                       # turn at s_{k+1}
G_EV = np.array([G_cross(MAPS[(k + 1) % 6] - MAPS[k], MAPS[j] - MAPS[k])
                 for k in range(6) for j in range(6) if j not in (k, (k + 1) % 6)])   # vertex j left of edge k
G_QV = np.zeros((6, 6, 7, 7))                                                 # q_j(s_k)
for k in range(6):
    for j in range(6):
        G_QV[k, j] = G_q(MAPS[k], MAPS[j])


# ---------------------------------------------------------------- centred-form bounds
def bound(G, m, h):
    """G: (P,7,7) shared, or (N,P,7,7) per box; m, h: (N,6).  Returns lo, hi of shape (N,P).
    p(m+d) = p(m) + 2 (G y)[:6].d + d^T G[:6,:6] d,  |d_i| <= h_i."""
    N = m.shape[0]
    y = np.concatenate([m, np.ones((N, 1))], axis=1)
    if G.ndim == 3:
        Gy = np.einsum('pij,nj->npi', G, y)
        Q = G[:, :6, :6]
        D = np.einsum('pii->pi', Q)
        absQ = np.abs(Q)
        full = np.einsum('pij,ni,nj->np', absQ, h, h)
        dabs = np.einsum('pi,ni->np', np.abs(D), h * h)
        dlo = np.einsum('pi,ni->np', np.minimum(D, 0), h * h)
        dhi = np.einsum('pi,ni->np', np.maximum(D, 0), h * h)
    else:
        Gy = np.einsum('npij,nj->npi', G, y)
        Q = G[:, :, :6, :6]
        D = np.einsum('npii->npi', Q)
        absQ = np.abs(Q)
        full = np.einsum('npij,ni,nj->np', absQ, h, h)
        dabs = np.einsum('npi,ni->np', np.abs(D), h * h)
        dlo = np.einsum('npi,ni->np', np.minimum(D, 0), h * h)
        dhi = np.einsum('npi,ni->np', np.maximum(D, 0), h * h)
    val = np.einsum('npi,ni->np', Gy, y)
    lin = 2.0 * np.einsum('npi,ni->np', np.abs(Gy[:, :, :6]), h)
    offd = full - dabs
    return val - lin + dlo - offd, val + lin + dhi + offd


# ---------------------------------------------------------------- interval helpers
def ilog(lo, hi):
    lo = np.maximum(lo, 1e-300)
    return np.log(lo), np.log(hi)


def iangle(c_lo, c_hi, d_lo, d_hi):
    """interval of atan2(c, d) for (c, d) in the box, assuming the angle lies in [0, pi)
    (c = cross >= 0 for points of the domain).  Returns lo, hi and ok-mask."""
    N = c_lo.shape
    lo = np.full(N, 0.0); hi = np.full(N, np.pi); ok = np.zeros(N, bool)
    # case 1: c > 0 : angle = pi/2 - atan(d/c), decreasing in d/c
    m1 = c_lo > 0
    r_lo = np.where(d_lo >= 0, d_lo / np.where(m1, c_hi, 1), d_lo / np.where(m1, c_lo, 1))
    r_hi = np.where(d_hi >= 0, d_hi / np.where(m1, c_lo, 1), d_hi / np.where(m1, c_hi, 1))
    lo = np.where(m1, np.pi / 2 - np.arctan(r_hi), lo)
    hi = np.where(m1, np.pi / 2 - np.arctan(r_lo), hi)
    ok |= m1
    # case 2: d > 0 (and c possibly straddling 0): angle = atan(c/d), increasing in c/d
    m2 = (~m1) & (d_lo > 0)
    q_lo = np.where(c_lo >= 0, c_lo / np.where(m2, d_hi, 1), c_lo / np.where(m2, d_lo, 1))
    q_hi = np.where(c_hi >= 0, c_hi / np.where(m2, d_lo, 1), c_hi / np.where(m2, d_hi, 1))
    lo = np.where(m2, np.maximum(np.arctan(q_lo), 0.0), lo)
    hi = np.where(m2, np.arctan(q_hi), hi)
    ok |= m2
    return lo, hi, ok


# ---------------------------------------------------------------- per-box evaluation
def evaluate(m, h, eps=1e-9, marg=1e-9, want_edge=True):
    """Classify boxes.  Returns (status, cert) arrays:
       status 0 = infeasible (discard), 1 = certified, 2 = undecided.
       cert: integer code of the certificate used (see CERT_NAMES)."""
    N = m.shape[0]
    status = np.full(N, 2, np.int8)
    cert = np.full(N, -1, np.int16)
    det_k = np.full(N, -1, np.int16)      # witness vertex / edge index, or infeasibility constraint id
    det_t = np.zeros(N)                   # edge parameter t

    n2_lo, n2_hi = bound(G_N2, m, h)
    cr_lo, cr_hi = bound(G_CR, m, h)
    dt_lo, dt_hi = bound(G_DT, m, h)
    tu_lo, tu_hi = bound(G_TURN, m, h)
    ev_lo, ev_hi = bound(G_EV, m, h)
    # infeasibility reasons: id 0-5 |s_k|^2>1 (k=id), 10-15 turn<0, 20-25 cross<0, 100+i EV_i<0, 200 winding
    viol = np.concatenate([n2_lo - (1 + eps), -eps - tu_hi, -eps - cr_hi, -eps - ev_hi], 1)
    ids = np.r_[np.arange(6), 10 + np.arange(6), 20 + np.arange(6), 100 + np.arange(G_EV.shape[0])]
    viol[:, 0] = -1.0                      # |s_1|^2 = 1 exactly, never a reason
    infeas = (viol > 0).any(1)
    det_k[infeas] = ids[np.argmax(viol[infeas], 1)]
    # angles
    a_lo, a_hi, a_ok = iangle(cr_lo, cr_hi, dt_lo, dt_hi)
    tot_lo = a_lo.sum(1)
    wind = a_ok.all(1) & (tot_lo > 2 * np.pi + 1e-9) & ~infeas     # winds twice: not convex
    det_k[wind] = 200
    infeas |= wind
    status[infeas] = 0
    live = ~infeas

    qv_lo, qv_hi = bound(G_QV.reshape(36, 7, 7), m, h)
    qv_lo = qv_lo.reshape(N, 6, 6)                                # [n, k, j] lower bound of q_j(s_k)

    # ---- W (vertex witnesses)
    offdiag = ~np.eye(6, dtype=bool)
    vmin = np.where(offdiag[None], qv_lo, np.inf).min(2)          # (N,6)
    wv = (vmin >= marg).any(1) & live
    status[wv] = 1; cert[wv] = 0; det_k[wv] = np.argmax(vmin, 1)[wv]

    # ---- potentials
    lg_lo, lg_hi = ilog(np.maximum(n2_lo, 1e-300), n2_hi)
    lg_lo, lg_hi = 0.25 * S3 * lg_lo, 0.25 * S3 * lg_hi
    lg_lo[:, 0] = 0.0; lg_hi[:, 0] = 0.0
    th_lo = np.concatenate([np.zeros((N, 1)), np.cumsum(a_lo, 1)[:, :5]], 1)
    th_hi = np.concatenate([np.zeros((N, 1)), np.cumsum(a_hi, 1)[:, :5]], 1)
    angok = a_ok[:, :5].all(1)

    def candidates(sign, C):
        # D[k, j] = Psi_k - Psi_j  bounds; Psi = lg + sign*theta/2 + C
        # theta_k - theta_j for k > j: sum a[j..k-1]; use cumulative sums of a (independent increments)
        cum_lo = np.concatenate([np.zeros((N, 1)), np.cumsum(a_lo, 1)], 1)
        cum_hi = np.concatenate([np.zeros((N, 1)), np.cumsum(a_hi, 1)], 1)
        cand = np.ones((N, 6), bool)
        for k in range(6):
            for j in range(6):
                if j == k:
                    continue
                if k > j:
                    dth_lo = cum_lo[:, k] - cum_lo[:, j]; dth_hi = cum_hi[:, k] - cum_hi[:, j]
                else:
                    dth_lo = -(cum_hi[:, j] - cum_hi[:, k]); dth_hi = -(cum_lo[:, j] - cum_lo[:, k])
                if sign > 0:
                    t_hi = dth_hi / 2
                else:
                    t_hi = -dth_lo / 2
                dhi = lg_hi[:, k] - lg_lo[:, j] + t_hi + (C[k] - C[j])
                cand[:, k] &= dhi >= -eps
        return cand

    und = live & (status == 2) & angok
    if und.any():
        candA = candidates(+1, CA)
        candB = candidates(-1, CB)
        candC = candidates(-1, CC)
        k3 = np.array([qv_lo[:, k, (k + 3) % 6] for k in range(6)]).T     # q_{k+3}(s_k)
        k1 = np.array([qv_lo[:, k, (k + 1) % 6] for k in range(6)]).T     # q_{k+1}(s_k)
        okA = np.where(candA, k3 >= marg, True).all(1)
        isT1 = np.array([1, 0, 1, 0, 1, 0], bool)
        okB = np.where(candB & isT1[None], k1 >= marg, True).all(1)
        okC = np.where(candC & ~isT1[None], k1 >= marg, True).all(1)
        for code, ok in ((1, okA), (2, okB), (3, okC)):
            sel = und & ok & (status == 2)
            status[sel] = 1; cert[sel] = code

    # ---- Lemma T certificates
    und = live & (status == 2)
    if und.any():
        for code, T, O in ((4, T1, T2), (5, T2, T1)):
            T = list(T); O = list(O)
            cand = np.ones((N, 3), bool)
            for a in range(3):
                for b in range(3):
                    if a != b:
                        cand[:, a] &= n2_hi[:, T[a]] - n2_lo[:, T[b]] >= 0   # crude but valid
            crossok = np.stack([qv_lo[:, T[a]][:, O].min(1) >= marg for a in range(3)], 1)
            ok = np.where(cand, crossok, True).all(1)
            sel = und & ok & (status == 2)
            status[sel] = 1; cert[sel] = code

    # ---- edge witnesses (best point at the centre)
    und = live & (status == 2)
    if want_edge and und.any():
        idx = np.where(und)[0]
        k_best, t_best = best_witness_at(m[idx])
        Mx = (1 - t_best)[:, None, None] * MAPS[k_best] + t_best[:, None, None] * MAPS[(k_best + 1) % 6]
        G = 0.5 * S3 * G_norm2(Mx)[:, None] - G_cross(Mx[:, None], MAPS[None])     # (n,6,7,7)
        lo, _ = bound(G, m[idx], h[idx])
        ok = (lo >= marg).all(1)
        status[idx[ok]] = 1; cert[idx[ok]] = 6; det_k[idx[ok]] = k_best[ok]; det_t[idx[ok]] = t_best[ok]
    return status, cert, det_k, det_t


CERT_NAMES = {0: "W-vertex", 1: "A", 2: "B", 3: "C", 4: "L1", 5: "L2", 6: "W-edge"}


def s_from_x(x):
    s1 = np.ones(len(x), complex)
    s2 = x[:, 0] + 1j * x[:, 1]; s3 = x[:, 2] + 1j * x[:, 3]; s4 = x[:, 4] + 1j * x[:, 5]
    return np.stack([s1, s2, s3, s4, -s1 - s3, -s2 - s4], 1)


def best_witness_at(xc):
    """For each centre configuration, the witness point (edge k from s_k to s_k+1, parameter t) maximising
    min_j q_j(x); candidates t in {0,1} and crossings of q_i, q_j (linear in t).  t rounded to 2^-12."""
    s = s_from_x(xc)                                    # (n,6)
    n = len(s)
    best = np.full(n, -np.inf); kb = np.zeros(n, int); tb = np.zeros(n)
    for k in range(6):
        a = s[:, k]; d = s[:, (k + 1) % 6] - a
        ts = [np.zeros(n), np.ones(n)]
        for i in range(6):
            for j in range(i + 1, 6):
                den = (np.conj(d) * (s[:, i] - s[:, j])).imag
                num = -(np.conj(a) * (s[:, i] - s[:, j])).imag
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = np.where(np.abs(den) > 1e-14, num / den, -1.0)
                ts.append(np.where((t > 0) & (t < 1), t, 0.0))
        T = np.stack(ts, 1)                             # (n, ncand)
        T = np.round(T * 4096) / 4096
        X = a[:, None] + T * d[:, None]
        q = 0.5 * S3 * np.abs(X)[..., None] ** 2 - (np.conj(X)[..., None] * s[:, None, :]).imag
        qm = q.min(2)
        i = np.argmax(qm, 1)
        v = qm[np.arange(n), i]
        upd = v > best
        best = np.where(upd, v, best); kb = np.where(upd, k, kb); tb = np.where(upd, T[np.arange(n), i], tb)
    return kb, tb


# ---------------------------------------------------------------- driver
def run(lo0, hi0, max_boxes=10 ** 9, batch=1 << 15, min_width=1e-5, verbose=True, log_every=20):
    import time
    queue = [(np.array([(lo0 + hi0) / 2]), np.array([(hi0 - lo0) / 2]))]
    leaves = {c: [] for c in CERT_NAMES}
    failed = []
    infeasible = []
    nproc = 0; ninf = 0; it = 0
    t0 = time.time()
    while queue:
        m = np.concatenate([q[0] for q in queue]); h = np.concatenate([q[1] for q in queue])
        queue = []
        for s0 in range(0, len(m), batch):
            mb, hb = m[s0:s0 + batch], h[s0:s0 + batch]
            st, ce, dk, dt = evaluate(mb, hb)
            nproc += len(mb); ninf += (st == 0).sum()
            sel = st == 0
            if sel.any():
                infeasible.append(np.concatenate([mb[sel], hb[sel], dk[sel][:, None]], 1))
            for c in CERT_NAMES:
                sel = (st == 1) & (ce == c)
                if sel.any():
                    leaves[c].append(np.concatenate([mb[sel], hb[sel], dk[sel][:, None], dt[sel][:, None]], 1))
            und = st == 2
            if und.any():
                mu, hu = mb[und], hb[und]
                small = hu.max(1) < min_width
                if small.any():
                    failed.append(np.concatenate([mu[small], hu[small]], 1))
                mu, hu = mu[~small], hu[~small]
                if len(mu):
                    ax = np.argmax(hu, 1)
                    hn = hu.copy(); hn[np.arange(len(hu)), ax] /= 2
                    off = np.zeros_like(hu); off[np.arange(len(hu)), ax] = hn[np.arange(len(hu)), ax]
                    queue.append((np.concatenate([mu - off, mu + off]), np.concatenate([hn, hn])))
            if nproc > max_boxes:
                break
        it += 1
        if verbose:
            nq = sum(len(q[0]) for q in queue)
            nl = {CERT_NAMES[c]: sum(len(a) for a in v) for c, v in leaves.items()}
            print("iter %d: processed %d, infeasible %d, queue %d, failed %d, leaves %s  (%.0fs)" %
                  (it, nproc, ninf, nq, sum(len(f) for f in failed), nl, time.time() - t0), flush=True)
        if nproc > max_boxes:
            break
    return leaves, failed, queue, infeasible
