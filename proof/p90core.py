"""Exact evaluation of the P_90 margin for two-triangle hexagons.

H = conv(s_1..s_6), ccw convex, s1+s3+s5 = 0 = s2+s4+s6.
For x != 0:  q_j(x) = (sqrt3/2)|x|^2 - cross(x, s_j),  cross(a,b) = Im(conj(a) b).
P_90 holds for H  <=>  exists x in dH with min_j q_j(x) >= 0.
On an edge x(t) = a + t d, all q_j(t) share the same convex quadratic part, so q_i - q_j is linear
and max_t min_j q_j(t) is attained at t in {0,1} or at a crossing t_ij.  Hence the margin
    Mq(H) = max_{x in dH} min_j q_j(x) / R^2 ,   R = max |s_j|
is computed exactly (up to floating point) from finitely many candidates."""
import numpy as np

S3 = np.sqrt(3.0)
W = np.exp(2j * np.pi / 3)


def cross(a, b):
    return (np.conj(a) * b).imag


def hexagon(p):
    """p: 8 reals -> s (6 complex)"""
    s1, s2, s3, s4 = p[0] + 1j * p[4], p[1] + 1j * p[5], p[2] + 1j * p[6], p[3] + 1j * p[7]
    return np.array([s1, s2, s3, s4, -s1 - s3, -s2 - s4])


def params(s):
    return np.r_[s[:4].real, s[:4].imag]


def convexity(s):
    """min over k of the normalised left-turn cross product, and the winding (should be 1).
    >= 0 and winding 1 <=> weakly convex ccw hexagon in this vertex order."""
    e = np.roll(s, -1) - s
    turns = cross(e, np.roll(e, -1))
    ang = np.angle(np.roll(s, -1) / s)
    wind = np.sum(ang) / (2 * np.pi)
    sc = np.max(np.abs(s)) ** 2
    return np.min(turns) / sc, wind


def is_convex(s, tol=0.0):
    m, w = convexity(s)
    return (m >= -tol) and abs(w - 1) < 1e-6 and np.all(np.angle(np.roll(s, -1) / s) > -1e-12)


def edge_candidates(s, k):
    a = s[k]
    d = s[(k + 1) % 6] - a
    ts = [0.0, 1.0]
    for i in range(6):
        for j in range(i + 1, 6):
            den = cross(d, s[i] - s[j])
            if abs(den) > 1e-15:
                t = -cross(a, s[i] - s[j]) / den
                if 0.0 < t < 1.0:
                    ts.append(t)
    return a, d, np.array(ts)


def qmin(s, x):
    """min_j q_j(x) for an array of points x"""
    x = np.asarray(x)
    return np.min(0.5 * S3 * np.abs(x)[..., None] ** 2 - cross(x[..., None], s), axis=-1)


def margin(s, return_witness=False):
    R2 = np.max(np.abs(s)) ** 2
    best = (-np.inf, None)
    for k in range(6):
        a, d, ts = edge_candidates(s, k)
        x = a + ts * d
        q = qmin(s, x)
        i = np.argmax(q)
        if q[i] > best[0]:
            best = (q[i], (k, ts[i]))
    if return_witness:
        return best[0] / R2, best[1]
    return best[0] / R2


def margin_vertex(s):
    """margin using vertex witnesses only"""
    return np.max(qmin(s, s)) / np.max(np.abs(s)) ** 2


def sym_family(r, alpha):
    """120-degree symmetric member: s1 = 1, s2 = r e^{i alpha}, s_{k+2} = w s_k"""
    s1 = 1.0 + 0j
    s2 = r * np.exp(1j * alpha)
    return np.array([s1, s2, W * s1, W * s2, W * W * s1, W * W * s2])


def gaussian_class(N, rng):
    """random strictly convex class hexagons (s_k = Gaussian, s5 = -s1-s3, s6 = -s2-s4, keep convex ccw ones)"""
    out = []
    while sum(len(o) for o in out) < N:
        z = rng.normal(size=(50000, 4)) + 1j * rng.normal(size=(50000, 4))
        s = np.c_[z[:, 0], z[:, 1], z[:, 2], z[:, 3], -z[:, 0] - z[:, 2], -z[:, 1] - z[:, 3]]
        e = np.roll(s, -1, axis=1) - s
        turns = (np.conj(e) * np.roll(e, -1, axis=1)).imag
        ang = np.angle(np.roll(s, -1, axis=1) / s)
        good = (turns.min(1) > 0) & (ang.min(1) > 0) & (np.abs(ang.sum(1) - 2 * np.pi) < 1e-9)
        out.append(s[good])
    return np.concatenate(out)[:N]
