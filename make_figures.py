"""
Figures 1-4 of the revised manuscript (PNG, 400 dpi), from npz_model.py and
the result files written by run_patterns2d.py and run_peclet.py.

Usage: python3 make_figures.py [1 2 3 4]   (default: all)
"""
import json
import os
import sys
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import npz_model as m

RES = os.path.join(os.path.dirname(__file__), "results")
W = fs.WONG


def fig_domain():
    """Box Omega = (0,L1) x (0,L2) x (0,L3) in oblique projection: axes,
    dimensions, boundary parts, and the overturning cell of Section 7.3."""
    L1, L2, L3 = 4.0, 3.0, 1.6
    ang, sc = np.deg2rad(35), 0.55

    def P(x, y, z):
        return np.array([x + sc * y * np.cos(ang), z + sc * y * np.sin(ang)])

    V = {i: P(*v) for i, v in enumerate([[0, 0, 0], [L1, 0, 0], [L1, L2, 0], [0, L2, 0],
                                          [0, 0, L3], [L1, 0, L3], [L1, L2, L3], [0, L2, L3]])}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    faces = [([0, 1, 2, 3], "#8c8c8c", 0.30), ([3, 2, 6, 7], "#d9d9d9", 0.25),
             ([0, 3, 7, 4], W["blue"], 0.22), ([1, 2, 6, 5], W["orange"], 0.45),
             ([4, 5, 6, 7], W["skyblue"], 0.35), ([0, 1, 5, 4], "#ffffff", 0.0)]
    for idx, col, al in faces:
        ax.add_patch(Polygon([V[i] for i in idx], closed=True, facecolor=col, alpha=al,
                             edgecolor="none"))
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    hidden = {(0, 3), (2, 3), (3, 7)}
    for e in edges:
        p, q = V[e[0]], V[e[1]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="k", lw=1.0, ls="--" if e in hidden else "-")
    # overturning cell in the front plane x2 = 0
    t = np.linspace(0, 2 * np.pi, 300)
    cx, cz, rx, rz = L1 / 2, L3 / 2, 0.34 * L1, 0.28 * L3
    ring = np.array([P(cx + rx * np.cos(s), 0, cz + rz * np.sin(s)) for s in t])
    ax.plot(ring[:, 0], ring[:, 1], color=W["vermillion"], lw=1.6)
    for s0, s1 in ((0.0, 0.25), (np.pi, np.pi + 0.25), (np.pi / 2, np.pi / 2 + 0.25), (1.5 * np.pi, 1.5 * np.pi + 0.25)):
        p0 = P(cx + rx * np.cos(s0), 0, cz + rz * np.sin(s0)); p1 = P(cx + rx * np.cos(s1), 0, cz + rz * np.sin(s1))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=W["vermillion"]))
    ax.text(*(P(0, 0, 0) + [-0.3, -0.38]), "overturning cell $\\psi$ (plane $x_2=0$)",
            color=W["vermillion"], ha="left", va="top", fontsize=10)
    ax.text(*(P(cx + rx, 0, cz + rz) + [-0.1, 0.12]), "upwelling", color=W["vermillion"], fontsize=9,
            ha="center")
    # coordinate axes
    o = P(-1.6, 0, -0.1)
    for d, lab, off in (((1, 0, 0), "$x_1$ (east)", (0.06, -0.08)),
                        ((0, 1, 0), "$x_2$ (north)", (0.04, 0.0)),
                        ((0, 0, 1), "$x_3$ (up)", (-0.3, 0.1))):
        q = o + (P(*d) - P(0, 0, 0)) * 0.75
        ax.add_patch(FancyArrowPatch(o, q, arrowstyle="-|>", mutation_scale=12, color="k", lw=1.2))
        ax.text(q[0] + off[0], q[1] + off[1], lab, fontsize=10)
    ax.text(o[0] - 0.12, o[1] - 0.12, "$\\mathcal{O}$", fontsize=11, ha="right")
    # dimensions
    ax.text(*(P(L1 / 2, 0, 0) + [0, -0.2]), "$L_1$", ha="center", fontsize=11)
    ax.text(*(P(L1, L2 / 2, 0) + [0.12, -0.1]), "$L_2$", fontsize=11)
    ax.text(*(P(0, 0, L3 / 2) + [-0.08, 0]), "$L_3$", fontsize=11, ha="right")
    # boundary parts
    ax.text(*P(L1 / 2, L2 * 0.65, L3), "$\\Gamma_s$: sea surface", ha="center", va="center", fontsize=10)
    ax.text(*(P(0, 0, 0) + [-0.3, -0.68]), "$\\Gamma_b$: sea floor (bottom face)", ha="left",
            va="top", fontsize=10, color="0.25")
    ax.text(*(P(L1, L2, L3 * 0.5) + [0.1, 0]), "$\\Gamma_c$: coast", fontsize=10, color="#9a6200")
    ax.text(*(P(0, L2 * 0.55, L3 * 0.75) + [-0.12, 0]), "$\\Gamma_o$: open\nocean", fontsize=10,
            color=W["blue"], ha="right")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-2.6, L1 + sc * L2 + 1.5); ax.set_ylim(-1.0, L3 + sc * L2 * np.sin(ang) + 0.3)
    fs.save(fig, "fig1_domain.png")


def fig_dispersion():
    p, d = m.PARAMS, m.DELTA
    c = m.equilibrium()
    chi_c, k_c, tau_c = m.chi_threshold("prey")
    k = np.linspace(0.005, 4.0, 4000)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for chi, col, ls, lab in ((0.0, W["black"], "-", r"$\chi_z=0$"),
                              (chi_c, W["blue"], "--", rf"$\chi_z=\chi_z^c\approx{chi_c:.2f}$"),
                              (4.0, W["vermillion"], "-", r"$\chi_z=4$")):
        gr = m.growth_rate(k, chiz=chi)
        ax[0].plot(k, gr, color=col, ls=ls, label=lab)
    gr4 = m.growth_rate(k, chiz=4.0)
    ax[0].fill_between(k, 0, gr4, where=gr4 > 0, color=W["vermillion"], alpha=0.18,
                       label="unstable band, $\\chi_z=4$")
    kf = k[gr4.argmax()]
    ax[0].plot([kf], [gr4.max()], "o", color=W["vermillion"])
    ax[0].annotate(rf"$k_f\approx{kf:.2f}$", (kf, gr4.max()), xytext=(8, -4),
                   textcoords="offset points")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("wavenumber $k$"); ax[0].set_ylabel(r"$\max_j \mathrm{Re}\,\lambda_j(k)$")
    ax[0].set_title("(a) dispersion relation at $c^\\star$")
    ax[0].legend(loc="lower left", frameon=False)
    # neutral curve a3 = 0 in the (k, chi_z) plane; chi_z < 0 (attraction) is stable
    s = d * k ** 2
    chi_neutral = d * m.Q_prey(s) / m.sigma(c[1])
    ax[1].plot(k, chi_neutral, color=W["blue"], label="neutral curve $a_3(k^2)=0$")
    ax[1].fill_between(k, chi_neutral, 8, color=W["vermillion"], alpha=0.18,
                       label="stationary instability")
    ax[1].plot([k_c], [chi_c], "o", color=W["blue"])
    ax[1].annotate(rf"$(k_c,\chi_z^c)\approx({k_c:.2f},{chi_c:.2f})$", (k_c, chi_c),
                   xytext=(10, -14), textcoords="offset points")
    ax[1].axhspan(-2 * d, 2 * d, color=W["green"], alpha=0.35, lw=0)
    ax[1].text(3.95, 2 * d + 0.15, r"(HS): $|\chi_z|<2\delta$", color=W["green"], ha="right")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].text(0.1, -2.2, "attraction ($\\chi_z<0$): stable for all $k$", fontsize=10)
    ax[1].text(0.1, 7.3, "avoidance ($\\chi_z>0$)", fontsize=10)
    ax[1].set_ylim(-3, 8); ax[1].set_xlim(0, 4)
    ax[1].set_xlabel("wavenumber $k$"); ax[1].set_ylabel(r"zooplankton sensitivity $\chi_z$")
    ax[1].set_title("(b) instability region")
    ax[1].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fs.save(fig, "fig2_dispersion.png")


def fig_patterns():
    av = np.load(os.path.join(RES, "patterns2d_fields.npz"))["N256_dt0.01"]
    at = np.load(os.path.join(RES, "attraction_fields.npz"))["N192"]
    panels = [(av[2], 20, "(a) zooplankton $c_3$, avoidance", False),
              (av[1], 20, "(b) phytoplankton $c_2$, avoidance", False),
              (at[0], 60, "(c) nutrient $c_1$, attraction", False),
              (at[1], 60, "(d) phytoplankton $c_2$, attraction", True)]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 8.0))
    for a, (fld, L, ttl, contour) in zip(axs.ravel(), panels):
        im = a.imshow(fld.T, origin="lower", extent=[0, L, 0, L], cmap="viridis",
                      interpolation="nearest")
        if contour:
            x = (np.arange(fld.shape[0]) + 0.5) * L / fld.shape[0]
            a.contour(x, x, fld.T, levels=[0.0], colors="white", linewidths=1.2)
        a.set_title(ttl); a.set_xlabel("$x_1$"); a.set_ylabel("$x_2$")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.set_label("non-dim. concentration")
    fig.tight_layout()
    fs.save(fig, "fig3_patterns.png")


def fig_peclet():
    res = json.load(open(os.path.join(RES, "peclet.json")))
    flds = np.load(os.path.join(RES, "peclet_fields.npz"))
    tag = "fine" if res.get("fine") else "coarse"
    rows = res[tag]
    show = [rows[0], min(rows, key=lambda r: abs(r["A"] - 2.0)), rows[-1]]
    fl = [flds[f"{tag}_A{r['A']}"] for r in show]
    vmin = min(x.min() for x in fl); vmax = max(x.max() for x in fl)
    fig = plt.figure(figsize=(12.5, 5.6))
    for j, (r, x) in enumerate(zip(show, fl)):
        a = fig.add_axes([0.06 + 0.29 * j, 0.58, 0.26, 0.36])
        im = a.imshow(x.T, origin="lower", extent=[0, 20, 0, 8], cmap="viridis",
                      vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        a.set_title(f"({'abc'[j]}) $c_3$, Pe $={r['Pe']:.0f}$")
        a.set_xlabel("$x_1$ (cross-shore)"); a.set_ylabel("$x_3$")
    cax = fig.add_axes([0.93, 0.58, 0.012, 0.36])
    cb = fig.colorbar(im, cax=cax); cb.set_label("$c_3$")
    a = fig.add_axes([0.08, 0.1, 0.78, 0.36])
    a2 = a.twinx()
    for key, mk, col, lab in (("coarse", "o", W["vermillion"], "$128\\times52$"),
                              ("fine", "s", W["blue"], "$256\\times104$")):
        rr = res.get(key) or []
        if not rr:
            continue
        pe = [r["Pe"] for r in rr]
        a.plot(pe, [r["amp"] for r in rr], mk + "-", color=col, label="amplitude, " + lab)
        a2.plot(pe, [r["frac"] for r in rr], mk + ":", color=col, mfc="none",
                label="band fraction, " + lab)
    a.set_xscale("symlog", linthresh=5)
    a.set_xlabel("P\u00e9clet number $\\mathrm{Pe}=U\\lambda_f/\\delta$ (linear below 5, logarithmic above)")
    a.set_ylabel("amplitude std$(c_3)/c_3^\\star$"); a2.set_ylabel("Turing-band fraction")
    a.set_ylim(0, 1); a2.set_ylim(0, 1)
    a.set_title("(d) time-averaged pattern measures, $t\\in[200,250]$")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, loc="lower left", ncol=2, fontsize=9)
    fs.save(fig, "fig4_peclet.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for w in which:
        {"1": fig_domain, "2": fig_dispersion, "3": fig_patterns, "4": fig_peclet}[w]()
