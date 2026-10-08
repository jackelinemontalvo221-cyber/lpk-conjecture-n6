"""Point-wise certificate margins (robust versions with candidate tolerance TOL).
Potentials (theta = continuous increasing lift of arg s_k, s indices 0..5 = vertices 1..6):
  PsiA_k  = (sqrt3/2) log|s_k| + theta_k/2 - k*pi/6           (argmax: all conditions except opposite j=k+3)
  PsiB_k  = (sqrt3/2) log|s_k| - theta_k/2 + cB_k, cB = (0,0,1,1,2,2)*pi/3   (argmax T2 [odd index]: all; T1: needs cross j=k+1)
  PsiC_k  = (sqrt3/2) log|s_k| - theta_k/2 + cC_k, cC = (0,1,1,2,2,3)*pi/3   (argmax T1 [even index]: all; T2: needs cross)"""
import numpy as np
from p90core import *
T1, T2 = [0, 2, 4], [1, 3, 5]
CB = np.array([0, 0, 1, 1, 2, 2]) * np.pi / 3
CC = np.array([0, 1, 1, 2, 2, 3]) * np.pi / 3

def thetas(s):
    ang = np.angle(np.roll(s, -1) / s)
    return np.angle(s[0]) + np.r_[0, np.cumsum(ang)[:5]]

def potentials(s):
    th = thetas(s); lg = 0.5 * S3 * np.log(np.abs(s))
    return lg + th / 2 - np.arange(6) * np.pi / 6, lg - th / 2 + CB, lg - th / 2 + CC

def cond(s, k, d):
    """margin of condition Im(s_{k+d}/s_k) <= sqrt3/2, scaled: (sqrt3/2)|s_k|^2 - cross(s_k, s_{k+d})"""
    return 0.5 * S3 * abs(s[k]) ** 2 - cross(s[k], s[(k + d) % 6])

def cross_margin(s, k):
    other = T2 if k in T1 else T1
    return np.min(0.5 * S3 * abs(s[k]) ** 2 - cross(s[k], s[other]))

def all_certs(s, tol=1e-3):
    R2 = np.max(abs(s)) ** 2
    PA, PB, PC = potentials(s)
    out = {}
    m = PA.max(); out["psiA"] = min(cond(s, k, 3) for k in range(6) if PA[k] >= m - tol) / R2
    m = PB.max(); out["psiB"] = min((cond(s, k, 1) if k in T1 else np.inf) for k in range(6) if PB[k] >= m - tol) / R2
    m = PC.max(); out["psiC"] = min((cond(s, k, 1) if k in T2 else np.inf) for k in range(6) if PC[k] >= m - tol) / R2
    r2 = abs(s) ** 2
    for name, T in (("L1", T1), ("L2", T2)):
        mm = r2[T].max(); out[name] = min(cross_margin(s, k) for k in T if r2[k] >= mm - tol) / R2
    out["M"] = margin(s)
    return out

def best_cert(s, tol=1e-3):
    c = all_certs(s, tol)
    return max(c.values()), c
