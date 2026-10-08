"""Mutation tests for the verification.

Part (a)/(c): corrupted copies of the certificate (wrong witness, wrong certificate type, wrong reason for
discarding, deleted / duplicated / halved boxes).  A correct verifier must reject every one of them.
Part (b): modified copies of verify_arb.py (wrong constants or index conventions).  These measure how
sensitive the stored cover is to the details of the lemmas; a mutation that is NOT detected here is not an
error, but it means that the correctness of the corresponding detail rests on the written proof alone.

Usage (from proof/):  python3 mutation_tests.py      (about 5 minutes; writes only to a temporary directory)"""
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
import verify_arb as V

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="p90_mutations_")
rng = np.random.default_rng(4242)
d = dict(np.load(os.path.join(HERE, "cover.npz")))


def run(npz, script=os.path.join(HERE, "verify_arb.py")):
    p = subprocess.run([sys.executable, script, npz], capture_output=True, text=True, timeout=1800, cwd=TMP)
    out = (p.stdout + p.stderr).strip().splitlines()
    keep = [l for l in out if l.startswith(("verified", "AssertionError"))]
    return " | ".join(keep[-2:]) if keep else "no output (exit %d)" % p.returncode


def save(name, dd):
    path = os.path.join(TMP, name + ".npz")
    np.savez(path, **dd)
    return path


def direct(rows, kind):
    """number of rows rejected by verify_arb.check_box"""
    return sum(not V.check_box(r, kind) for r in rows)


def copy():
    return {k: v.copy() for k, v in d.items()}


results = []
# ---- (a) corrupted certificate data: every mutation must be detected
dd = copy(); idx = rng.choice(len(dd["cert0"]), 300, replace=False)
dd["cert0"][idx, 12] = (dd["cert0"][idx, 12] + 1) % 6
results.append(("(a1) vertex witness k -> k+1 on 300 boxes", "%d/300" % direct(dd["cert0"][idx], 0), run(save("a1", dd))))
dd = copy(); idx = rng.choice(len(dd["cert6"]), 300, replace=False)
dd["cert6"][idx, 13] = (dd["cert6"][idx, 13] + 0.5) % 1.0
results.append(("(a2) edge witness t -> t+1/2 mod 1 on 300 boxes", "%d/300" % direct(dd["cert6"][idx], 6), run(save("a2", dd))))
dd = copy(); idx = rng.choice(len(dd["cert6"]), 300, replace=False)
dd["cert6"][idx, 13] = np.clip(dd["cert6"][idx, 13] + 1 / 4096, 0, 1)
results.append(("(a2') edge witness t -> t+2^-12 on 300 boxes [expected: mostly still valid]",
                "%d/300" % direct(dd["cert6"][idx], 6), run(save("a2b", dd))))
for lab, src, dst in (("(a3) 300 A boxes relabelled B", 1, 2), ("(a4) 300 A boxes relabelled C", 1, 3),
                      ("(a5) 300 B boxes relabelled C", 2, 3), ("(a6) 300 C boxes relabelled B", 3, 2),
                      ("(a7) all L1 boxes relabelled L2", 4, 5), ("(a8) 300 A boxes relabelled vertex k=1", 1, 0)):
    dd = copy()
    n = len(dd["cert%d" % src]); idx = rng.choice(n, min(300, n), replace=False)
    moved = dd["cert%d" % src][idx].copy()
    if dst == 0:
        moved[:, 12] = 0
    dd["cert%d" % src] = np.delete(dd["cert%d" % src], idx, 0)
    dd["cert%d" % dst] = np.concatenate([dd["cert%d" % dst], moved])
    results.append((lab, "%d/%d" % (direct(moved, dst), len(moved)), run(save("a_%d_%d" % (src, dst), dd))))
dd = copy(); idx = rng.choice(len(dd["infeasible"]), 300, replace=False)
ids = np.r_[np.arange(1, 6), 10 + np.arange(6), 20 + np.arange(6), 100 + np.arange(24), 200]
old = dd["infeasible"][idx, 12].copy(); new = rng.choice(ids, 300); new = np.where(new == old, new + 1, new)
dd["infeasible"][idx, 12] = new
results.append(("(a9) 300 discard reasons changed", "%d/300" % direct(dd["infeasible"][idx], "inf"), run(save("a9", dd))))
dd = copy(); idx = rng.choice(len(dd["cert0"]), 300, replace=False)
moved = dd["cert0"][idx, :13].copy(); moved[:, 12] = 200
dd["cert0"] = np.delete(dd["cert0"], idx, 0); dd["infeasible"] = np.concatenate([dd["infeasible"], moved])
results.append(("(a10) 300 vertex-witness boxes declared discarded", "%d/300" % direct(moved, "inf"), run(save("a10", dd))))
# ---- (c) structural corruption of the tiling
dd = copy(); dd["cert0"] = np.delete(dd["cert0"], 12345, 0)
results.append(("(c1) delete one vertex-witness box", "-", run(save("c1", dd))))
dd = copy(); dd["infeasible"] = np.delete(dd["infeasible"], 777, 0)
results.append(("(c2) delete one discarded box", "-", run(save("c2", dd))))
dd = copy(); dd["cert1"] = np.concatenate([dd["cert1"], dd["cert1"][:1]])
results.append(("(c3) duplicate one A box", "-", run(save("c3", dd))))
dd = copy(); r = dd["cert3"][5].copy(); ax = int(np.argmax(r[6:12]))
dd["cert3"] = np.delete(dd["cert3"], 5, 0); r[6 + ax] /= 2; r[ax] -= r[6 + ax]
dd["cert3"] = np.concatenate([dd["cert3"], r[None]])
results.append(("(c4) replace one C box by its lower half", "-", run(save("c4", dd))))

# ---- (b) modified verifiers (sensitivity of the cover to the constants / conventions)
src = open(os.path.join(HERE, "verify_arb.py")).read()
muts = {
    "(b1) A constants: sign flipped": ("CA = [-(k * PI) / 6 for k in range(6)]", "CA = [+(k * PI) / 6 for k in range(6)]"),
    "(b2) A constants: all zero": ("CA = [-(k * PI) / 6 for k in range(6)]", "CA = [ZERO for k in range(6)]"),
    "(b3) A constants: shifted by one index": ("CA = [-(k * PI) / 6 for k in range(6)]", "CA = [-(((k + 1) % 6) * PI) / 6 for k in range(6)]"),
    "(b4) B constants: sign flipped": ("CB = [(c * PI) / 3 for c in (0, 0, 1, 1, 2, 2)]", "CB = [-(c * PI) / 3 for c in (0, 0, 1, 1, 2, 2)]"),
    "(b5) B constants: all zero": ("CB = [(c * PI) / 3 for c in (0, 0, 1, 1, 2, 2)]", "CB = [ZERO for c in range(6)]"),
    "(b6) C constants: sign flipped": ("CC = [(c * PI) / 3 for c in (0, 1, 1, 2, 2, 3)]", "CC = [-(c * PI) / 3 for c in (0, 1, 1, 2, 2, 3)]"),
    "(b7) C constants: all zero": ("CC = [(c * PI) / 3 for c in (0, 1, 1, 2, 2, 3)]", "CC = [ZERO for c in range(6)]"),
    "(b8) A checks j=k+2 instead of k+3": ("qlo(k, (k + 3) % 6, m, h) >= 0 for k in cand)", "qlo(k, (k + 2) % 6, m, h) >= 0 for k in cand)"),
    "(b9) B checks the other triangle": ("for k in cand if k in T1)", "for k in cand if k in T2)"),
    "(b10) candidate sets always empty": ("            if U < 0:            # certainly Psi_k < Psi_j on the box", "            if True:"),
}
for lab, (a, b) in muts.items():
    assert a in src, lab
    path = os.path.join(TMP, "verify_" + re.sub(r"\W+", "_", lab)[:24] + ".py")
    open(path, "w").write(src.replace(a, b))
    results.append((lab, "-", run(os.path.join(HERE, "cover.npz"), path)))

for lab, det, out in results:
    print("%-75s rows rejected: %-8s | verifier: %s" % (lab, det, out), flush=True)
print("temporary files in", TMP)
