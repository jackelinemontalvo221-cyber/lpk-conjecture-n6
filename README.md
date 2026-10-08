# Convexly independent eigenvectors of 6×6 doubly stochastic matrices

This repository contains a short write-up and the code and certificate data for a computer-assisted proof of the following theorem.

> **Theorem.** Let $A$ be a $6\times 6$ doubly stochastic matrix and let $\lambda$ be an eigenvalue of $A$. If $\lambda$ has a left or right eigenvector whose six entries are convexly independent (that is, they are the six vertices of a convex hexagon), then $\lambda\in\Pi_6$, the convex hull of the sixth roots of unity.

This is the case $n=6$ of Conjecture 2 of Levick, Pereira and Kribs ([Proc. AMS 143 (2015)](https://www.ams.org/journals/proc/2015-143-05/S0002-9939-2014-12412-9/S0002-9939-2014-12412-9.pdf)), in the stronger form $\lambda\in\Pi_6$. The bound is sharp: the eigenvalues of $6\times 6$ doubly stochastic matrices that admit a convexly independent eigenvector fill exactly $\Pi_6$.

**Relation to other work.** Bouthat, Mashreghi and Morneau-Guérin have announced the inclusion $\lambda\in\Pi_n$ for every $n$ in two talks given in May 2026 (see also their 2025 poster; references in the write-up). As far as we know their paper is not yet available, and in the slides one step of their argument is labelled as a conjecture. The proof here follows a different route, does not depend on their argument, and is specific to $n=6$.

**Write-up:** [`paper/lpk_n6.pdf`](paper/lpk_n6.pdf) (LaTeX source in the same folder).

## Outline of the proof

Let $Av=\lambda v$ with $\operatorname{Im}\lambda>0$ and $v$ convexly independent. Then $\sum_i v_i=0$, and $\lambda K_m(v)\subseteq K_m(v)$ for every $m$, where $K_m(v)$ is the convex hull of all sums of $m$ entries of $v$ with distinct indices. The proof splits according to $\arg\lambda$.

| sector | argument |
|---|---|
| $0<\arg\lambda\le\pi/3$ | classical sector bound of Dmitriev–Dynkin for stochastic matrices |
| $2\pi/3\le\arg\lambda<\pi$ | $K_3(v)$ is a centrally symmetric hexagon, so $\lambda$ is an eigenvalue of a real $3\times 3$ matrix with column $\ell^1$-norms $\le 1$; such eigenvalues lie in the Karpelevič region $\Theta_6$, and the sector bound applied to $-\bar\lambda$ finishes |
| $\pi/3<\arg\lambda<2\pi/3$ | $H=K_2(v)$ is a hexagon with vertices $s_j=v_j+v_{j+1}$, made of two interleaved triangles with centroid $0$. **Proposition P90:** every such hexagon has a boundary point $x$ with $\operatorname{Im}(s_j/x)\le\sqrt3/2$ for all $j$. Since $\lambda x\in H$, this gives $\operatorname{Im}\lambda\le\sqrt3/2$ |

Proposition P90 has zero margin along a whole two-parameter family of hexagons (those invariant under rotation by $120^\circ$), so it cannot be checked by bounding its margin directly. The proof uses three exact "potential" lemmas, each of which turns a vertex into a witness provided one remaining inequality holds, and a branch-and-bound computation over a compact 6-dimensional parameter space. All 53,093 boxes of the resulting cover are re-verified in ball arithmetic (Arb, via python-flint).

## Reproducing the verification

Requirements: Python 3.10 or newer, `numpy`, `scipy` and `python-flint` (tested with Python 3.11, numpy 2.4.4, scipy 1.17.1, python-flint 0.9.0).

```bash
pip install -r requirements.txt
cd proof
python3 verify_arb.py cover.npz        # rigorous check of all 53,093 boxes, about 30 s
```

Expected output (also in [`proof/logs/verify_arb.log`](proof/logs/verify_arb.log)):

```
enclosure self-test: True
python-flint 0.9.0  precision 96 bits
boxes: 53093  exact volume equal to initial box: True
tiling replay: leaves used 53093 of 53093 -> True
verified boxes: 53093, failures: 0, time 27s
```

Further checks (all optional, run from `proof/`):

| command | what it does | time |
|---|---|---|
| `python3 run_search.py cover_new.npz` | regenerates the cover with the floating-point branch-and-bound search | a few seconds |
| `python3 independent_verify.py cover.npz 12` | independent verifier (separate implementation, affine arithmetic, up to 12 extra bisections per box) | about 3 min |
| `python3 lemma_tests.py` | numerical tests of the analytic lemmas on 240,000 random hexagons | about 5 min |
| `python3 validate_cover.py` | pointwise consistency check of the cover on 20,000 random hexagons | about 1 min |
| `python3 endtoend.py` | random $6\times 6$ doubly stochastic matrices: every non-real eigenvalue with a convexly independent eigenvector lies in $\Pi_6$ | a few minutes |
| `python3 family_margin.py` | the margin of P90 vanishes on the $120^\circ$-symmetric family | seconds |
| `python3 trouble_search.py 1 20 1e-3` | searches for hexagons where all certificate types are weak (needs `pip install cma`) | under a minute |
| `python3 mutation_tests.py` | corrupts the certificate data in various ways and checks that the verifier rejects it; also measures how sensitive the cover is to the constants of the lemmas | about 12 min |

To compare a regenerated cover with the stored one:

```bash
python3 -c "import numpy as np; a=np.load('cover.npz'); b=np.load('cover_new.npz'); print(all(np.array_equal(a[k], b[k]) for k in a.files))"
```

## Files

| file | content |
|---|---|
| `paper/lpk_n6.tex`, `paper/lpk_n6.pdf` | the write-up |
| `paper/make_figure.py`, `paper/fig_overview.pdf` | Figure 1 |
| `proof/p90core.py` | basic geometry of the hexagons; exact P90 margin |
| `proof/certs.py` | potentials and pointwise certificate margins |
| `proof/bnb.py`, `proof/run_search.py` | floating-point branch-and-bound search (not part of the trusted base) |
| `proof/cover.npz` | the certificate: 53,093 boxes with their certificate types |
| `proof/verify_arb.py` | rigorous verification in ball arithmetic, including the tiling check |
| `proof/independent_verify.py` | an independent re-implementation of the verification |
| `proof/lemma_tests.py`, `proof/validate_cover.py`, `proof/endtoend.py`, `proof/family_margin.py`, `proof/trouble_search.py` | numerical sanity checks |
| `proof/mutation_tests.py` | mutation tests of the verifier |
| `proof/logs/` | outputs of the runs reported in the write-up |

## Format of `cover.npz`

Parameters are $p=(\operatorname{Re}s_2,\operatorname{Im}s_2,\operatorname{Re}s_3,\operatorname{Im}s_3,\operatorname{Re}s_4,\operatorname{Im}s_4)$, with $s_1=1$, $s_5=-1-s_3$, $s_6=-s_2-s_4$. A box is stored by its centre $m$ and half-widths $h$ (exact binary floating-point numbers).

| array | rows | meaning |
|---|---|---|
| `lo0`, `hi0` | – | the initial box $[-1,1]\times[0,1]\times[-1,0]\times[0,7/8]\times[-1,1]^2$ |
| `cert0` | `[m(6), h(6), k, 0]` | vertex witness $s_{k+1}$ |
| `cert6` | `[m(6), h(6), k, t]` | edge witness $(1-t)s_{k+1}+t\,s_{k+2}$, $t\in 2^{-12}\mathbb Z$ |
| `cert1`, `cert2`, `cert3` | `[m(6), h(6), -1, 0]` | potential certificates A, B, C |
| `cert4`, `cert5` | `[m(6), h(6), -1, 0]` | largest-vertex certificates L1 (triangle $s_1s_3s_5$), L2 (triangle $s_2s_4s_6$) |
| `infeasible` | `[m(6), h(6), reason]` | box contains no admissible hexagon; reason `0–5`: $\lvert s_{k+1}\rvert^2>1$, `10–15`: a right turn, `20–25`: $\operatorname{cr}(s_{k+1},s_{k+2})<0$, `100+i`: a vertex on the wrong side of an edge, `200`: total angle exceeds $2\pi$ |

Indices in the code are 0-based: code index $k$ is the vertex $s_{k+1}$ of the write-up.

## What the proof relies on

1. The pen-and-paper lemmas in Sections 2–5 of the write-up.
2. The correctness of Arb / python-flint and of `proof/verify_arb.py` (about 350 lines). The search code `bnb.py` only produces the cover; it is not trusted. The coefficient matrices used by the verifier are computed in floating point and checked entry by entry against exact rational arithmetic before use.

The independent verifier and the numerical tests are extra safeguards and are not needed for the logic of the proof.

## Use of AI tools

This work was carried out with substantial assistance from the AI system Claude (Anthropic): the reduction to Proposition P90, the potential lemmas and the programs were developed in interactive sessions with Claude, and `independent_verify.py` was written by a separate Claude instance instructed to look for errors. The author is responsible for the content.

## License

The code in this repository is released under the MIT License (see [`LICENSE`](LICENSE)). The write-up in `paper/` is not covered by that license; all rights to it are reserved by the author.

## Citation

Hengrui Zhu, *Eigenvalues of 6×6 doubly stochastic matrices with convexly independent eigenvectors*, preprint, October 2026.
