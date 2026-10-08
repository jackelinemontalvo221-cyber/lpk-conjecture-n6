"""Figure 1 of the write-up: (a) a convexly independent eigenvector and the hexagon K_2(v);
(b) how the upper half of Pi_6 is split among the three arguments.  Usage: python3 make_figure.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

# ---------------------------------------------------------------- (a)
ax = axes[0]
ang = np.radians([8, 62, 128, 175, 238, 300])
rad = np.array([1.0, 0.82, 1.05, 0.78, 0.95, 0.88])
v = rad * np.exp(1j * ang)
v = v - v.mean()                          # sum of entries = 0
s = v + np.roll(v, -1)                    # s_j = v_j + v_{j+1}
def poly(ax, z, **kw):
    z = np.r_[z, z[:1]]
    ax.plot(z.real, z.imag, **kw)
poly(ax, v, color="0.55", lw=1.0, ls="--")
poly(ax, s, color="k", lw=1.4)
t1, t2 = s[0::2], s[1::2]
ax.fill(t1.real, t1.imag, color="tab:blue", alpha=0.18)
ax.fill(t2.real, t2.imag, color="tab:orange", alpha=0.18)
poly(ax, t1, color="tab:blue", lw=1.0)
poly(ax, t2, color="tab:orange", lw=1.0)
ax.plot(v.real, v.imag, "o", color="0.45", ms=3)
ax.plot(s.real, s.imag, "ko", ms=3)
for j in range(6):
    ax.annotate(r"$v_%d$" % (j + 1), (v[j].real, v[j].imag), xytext=(0.80 * v[j].real, 0.80 * v[j].imag),
                ha="center", va="center", color="0.35", fontsize=7.5)
    ax.annotate(r"$s_%d$" % (j + 1), (s[j].real, s[j].imag), xytext=(1.13 * s[j].real, 1.13 * s[j].imag),
                ha="center", va="center", fontsize=9)
ax.plot([0], [0], "k+", ms=6)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"(a) $K_1(v)$ (dashed) and $K_2(v)=\mathrm{conv}(T_1\cup T_2)$", fontsize=9, pad=12)

# ---------------------------------------------------------------- (b)
ax = axes[1]
roots = np.exp(1j * np.pi * np.arange(7) / 3)
th = np.linspace(0, np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color="0.75", lw=0.8)
ax.plot([-1.15, 1.15], [0, 0], color="0.6", lw=0.6)
sectors = [(0, 1, "tab:green", "Dmitriev–Dynkin\nsector bound", (0.62, -0.2)),
           (1, 2, "tab:blue", r"$\mathrm{P}_{90}$ via $K_2$:  $\mathrm{Im}\,\lambda\leq\sqrt{3}/2$", (0.0, 1.13)),
           (2, 3, "tab:red", "signed $3\\times 3$ matrices\nvia $K_3$ and $\\lambda\\mapsto-\\bar\\lambda$", (-0.62, -0.2))]
for a, b, col, lab, pos in sectors:
    tri = np.array([0, roots[a], roots[b]])
    ax.fill(tri.real, tri.imag, color=col, alpha=0.22)
    ax.plot(np.r_[tri.real, 0], np.r_[tri.imag, 0], color=col, lw=1.0)
    ax.text(pos[0], pos[1], lab, ha="center", va="center", fontsize=7.5, color=col)
for k, lab in [(0, r"$1$"), (1, r"$e^{i\pi/3}$"), (2, r"$e^{2i\pi/3}$"), (3, r"$-1$")]:
    z = roots[k]
    ax.annotate(lab, (z.real, z.imag), xytext=(z.real * 1.12, z.imag * 1.12 - 0.06 * (k in (0, 3))),
                ha="center", va="center", fontsize=8)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.32, 1.3)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title(r"(b) the three sectors of $\Pi_6\cap\{\mathrm{Im}\,\lambda\geq 0\}$", fontsize=9)

fig.tight_layout()
fig.savefig("fig_overview.pdf")
fig.savefig("fig_overview.png", dpi=200)
print("written fig_overview.pdf / .png")
