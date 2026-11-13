"""Fig. 2 - The model and the null (schematic). Usage: python fig2_model_and_null.py OUT_DIR"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans",
                     "font.size": 8, "axes.linewidth": 0.6})
TEAL, AMBER, PURPLE, GREY = "#1b9e9e", "#e08a1e", "#7a5cb8", "#a0a0a0"
RED, BLUE = "#c0392b", "#2c5aa0"
rng = np.random.default_rng(7)


def label(ax, s, dx=-0.02):
    ax.text(dx, 1.04, s, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")


# ---------- (A) nodes and edges ----------
def panel_a(ax):
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.48, 1.32); ax.set_aspect("equal"); ax.axis("off")
    pos = {"CgB": (0., 1.), "ChB": (1., 1.), "CgH": (0., 0.), "ChH": (1., 0.)}
    names = {"CgB": "caregiver\nbrain", "ChB": "child\nbrain",
             "CgH": "caregiver\nheart", "ChH": "child\nheart"}
    col = {"CgB": TEAL, "ChB": TEAL, "CgH": AMBER, "ChH": AMBER}
    r = 0.17

    def edge(a, b, c, lw=1.2):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=9, color=c, lw=lw,
                     connectionstyle="arc3,rad=0.18",
                     shrinkA=r * 72 * 1.05, shrinkB=r * 72 * 1.05))
    # shrink is in points; approximate via a helper below instead
    pairs = [("CgB", "ChB", TEAL), ("CgH", "ChH", AMBER),
             ("CgB", "ChH", PURPLE), ("CgH", "ChB", PURPLE),
             ("CgB", "CgH", GREY), ("ChB", "ChH", GREY)]
    for a, b, c in pairs:
        for s, t in ((a, b), (b, a)):
            (x0, y0), (x1, y1) = np.array(pos[s]), np.array(pos[t])
            v = (np.array([x1, y1]) - [x0, y0]); v /= np.linalg.norm(v)
            p0 = np.array([x0, y0]) + v * (r + 0.02)
            p1 = np.array([x1, y1]) - v * (r + 0.03)
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                         mutation_scale=9, color=c, lw=1.2,
                         connectionstyle="arc3,rad=0.15",
                         linestyle="--" if c == GREY else "-"))
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), r, fc="white", ec=col[k], lw=1.8, zorder=3))
        ax.text(x, y, names[k], ha="center", va="center", fontsize=6.5, zorder=4)
    ax.text(0, 1.27, "caregiver", ha="center", fontsize=8, style="italic")
    ax.text(1, 1.27, "child", ha="center", fontsize=8, style="italic")
    hl = [(TEAL, "brain–brain"), (AMBER, "heart–heart"),
          (PURPLE, "cross-modal"), (GREY, "within-person (not tested)")]
    for i, (c, t) in enumerate(hl):
        x = -0.3 + (i % 2) * 0.85; y = -0.32 - (i // 2) * 0.1
        ax.plot([x, x + 0.1], [y, y], color=c, lw=1.5,
                ls="--" if c == GREY else "-")
        ax.text(x + 0.13, y, t, va="center", fontsize=6.5)
    ax.set_title("Four-node dyad model, 12 directed edges", fontsize=8.5)


# ---------- (B) stacked-window least squares ----------
def panel_b(ax):
    ax.set_xlim(0, 102); ax.set_ylim(0, 62); ax.axis("off")
    t = np.linspace(0, 60, 150)
    x = np.cumsum(rng.normal(size=150)) * 0.25
    x = (x - x.min()) / (x.max() - x.min())
    T0, TW = 2, 0.75
    ax.plot(T0 + t * TW, 52 + 6 * x, color=TEAL, lw=0.8)
    ax.text(T0, 60.5, "one 60 s segment (150 samples)", fontsize=7)
    for w in range(11):
        y = 46 - w * 3.3
        ax.add_patch(Rectangle((T0 + 5 * w * TW, y), 10 * TW, 2.4,
                     fc=RED if w == 4 else TEAL, alpha=0.35 if w == 4 else 0.25, ec="none"))
    ax.text(T0, 8.5, "11 windows × 10 s, 50% overlap,\neach linearly detrended",
            fontsize=6.5, va="top")
    wy = 46 - 4 * 3.3 + 1.2
    xs = T0 + (20 + np.arange(25) * 0.4) * TW
    ax.plot(xs, np.full(25, wy), ".", ms=1.6, color="k")
    ti = 14
    ax.plot(xs[ti - 4:ti], np.full(4, wy), "o", ms=3, mfc="white", mec=RED, mew=0.8)
    ax.plot(xs[ti], wy, "o", ms=3.5, color=RED)
    ax.annotate("one regression row:\nx_w(t) and its lags x_w(t−1 … t−p),\nall from the same window",
                xy=(xs[ti], wy + 0.8), xytext=(29, 41), fontsize=6.3, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    X0, Y0, H, WY, WX = 70, 6, 46, 4, 13
    bh = H / 11
    for w in range(11):
        y = Y0 + H - (w + 1) * bh
        fc, al = (RED, 0.35) if w == 4 else (TEAL, 0.22)
        ax.add_patch(Rectangle((X0, y), WY, bh * 0.92, fc=fc, alpha=al, ec="k", lw=0.3))
        ax.add_patch(Rectangle((X0 + WY + 1, y), WX, bh * 0.92, fc=fc, alpha=al, ec="k", lw=0.3))
    ax.text(X0 + WY / 2 - 0.5, Y0 + H + 1.2, "Y", ha="center", fontsize=8, style="italic")
    ax.text(X0 + WY + 1.5 + WX / 2, Y0 + H + 1.2, "X (p lags)", ha="center", fontsize=7, style="italic")
    xr = X0 + WY + 1 + WX
    yb0 = Y0 + H - 5 * bh; yb1 = yb0 + bh * 0.92
    ax.plot([xr + 0.8, xr + 1.8, xr + 1.8, xr + 0.8], [yb0, yb0, yb1, yb1], color=RED, lw=0.8)
    ax.text(xr + 2.5, (yb0 + yb1) / 2, "rows of\none\nwindow", ha="left",
            va="center", fontsize=6, color=RED)
    ax.add_patch(FancyArrowPatch((51, 27), (X0 - 2, 27), arrowstyle="-|>",
                 mutation_scale=9, color="k", lw=0.8))
    ax.text((51 + X0 - 2) / 2, 28.3, "stack rows\nof all windows", ha="center", fontsize=6.3)
    ax.text(X0 + (WY + 1 + WX) / 2, Y0 - 1.5, "one least-squares fit\n→ A(r), Σ  (Eq. 3)",
            ha="center", va="top", fontsize=7)
    ax.set_title("Least squares on stacked windows", fontsize=8.5)


# ---------- (C) N x N table and pairing permutation ----------
def panel_c(axm, axh):
    N = 8
    T = rng.normal(0, 1, (N, N)); T[np.diag_indices(N)] += 0.9
    axm.imshow(T, cmap="Greys", vmin=-2.5, vmax=3.5)
    for b in range(N):
        axm.add_patch(Rectangle((b - 0.5, b - 0.5), 1, 1, fc="none", ec=RED, lw=1.2))
    perm = rng.permutation(N)
    while np.any(perm == np.arange(N)):
        perm = rng.permutation(N)
    axm.plot(perm, np.arange(N), "o", ms=4, mfc="none", mec=BLUE, mew=1.1)
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_xlabel("caregiver c", fontsize=7); axm.set_ylabel("child b", fontsize=7)
    axm.set_title("table T_bc for one edge", fontsize=8.5)
    real = np.mean(np.diag(T))
    nulls = np.array([T[np.arange(N), rng.permutation(N)].mean() for _ in range(1000)])
    axh.hist(nulls, bins=30, color=BLUE, alpha=0.5)
    axh.set_ylim(0, axh.get_ylim()[1] * 1.3)
    axh.axvline(real, color=RED, lw=1.4)
    axh.text(real, axh.get_ylim()[1] * 0.92, "real\nmean ", color=RED, fontsize=6.5, va="top", ha="right")
    axh.set_yticks([]); axh.set_xlabel("mean over re-paired dyads", fontsize=7)
    axh.tick_params(labelsize=6)
    for s in ("top", "right", "left"):
        axh.spines[s].set_visible(False)
    axh.set_title("pairing-permutation null", fontsize=8.5)
    axm.text(0.5, -0.2, "red square: real dyad\nblue circle: one re-pairing",
             transform=axm.transAxes, fontsize=6, va="top", ha="center")


# ---------- (D) sign-flip test of d ----------
def panel_d(axs, axh):
    N = 25
    d = rng.normal(0.35, 0.6, N)
    flips = rng.choice([-1, 1], N)
    y = rng.uniform(-0.15, 0.15, N)
    axs.axvline(0, color="k", lw=0.5)
    axs.plot(d, 1 + y, "o", ms=3.5, color=RED, alpha=0.8)
    axs.plot(d * flips, y, "o", ms=3.5, mfc="none", mec=BLUE, mew=0.9)
    axs.set_yticks([0, 1]); axs.set_yticklabels(["one sign\nflip", "observed"], fontsize=6.5)
    axs.set_ylim(-0.5, 1.5)
    axs.set_xlabel("d = Δforward − Δreverse (per dyad)", fontsize=7)
    axs.tick_params(labelsize=6)
    for s in ("top", "right"):
        axs.spines[s].set_visible(False)
    axs.set_title("dominant direction", fontsize=8.5)
    nulls = np.array([(d * rng.choice([-1, 1], N)).mean() for _ in range(1000)])
    axh.hist(nulls, bins=30, color=BLUE, alpha=0.5)
    axh.set_ylim(0, axh.get_ylim()[1] * 1.3)
    axh.axvline(d.mean(), color=RED, lw=1.4)
    axh.text(d.mean(), axh.get_ylim()[1] * 0.92, "observed\nmean d ", color=RED, fontsize=6.5, va="top", ha="right")
    axh.set_yticks([]); axh.set_xlabel("mean d after sign flips", fontsize=7)
    axh.tick_params(labelsize=6)
    for s in ("top", "right", "left"):
        axh.spines[s].set_visible(False)
    axh.set_title("sign-flip null (two-sided)", fontsize=8.5)


def main(out, name="fig2_model_and_null", dpi=200):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.48, 6.4))  # 190 mm wide
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], hspace=0.3, wspace=0.18,
                          left=0.07, right=0.98, top=0.95, bottom=0.08)
    axA = fig.add_subplot(gs[0, 0]); panel_a(axA); label(axA, "A", dx=-0.12)  # the lower legend makes panel A narrower
    axB = fig.add_subplot(gs[0, 1]); panel_b(axB); label(axB, "B")
    gC = gs[1, 0].subgridspec(1, 2, wspace=0.35, width_ratios=[1, 1.1])
    axCm = fig.add_subplot(gC[0]); axCh = fig.add_subplot(gC[1])
    panel_c(axCm, axCh)
    gD = gs[1, 1].subgridspec(1, 2, wspace=0.35, width_ratios=[1.25, 1])
    axDs = fig.add_subplot(gD[0]); axDh = fig.add_subplot(gD[1])
    panel_d(axDs, axDh)
    for ax_l, ax_t, L in ((axCm, axCh, "C"), (axDs, axDh, "D")):
        fig.text(ax_l.get_position().x0 - 0.045, ax_t.get_position().y1 + 0.012, L,
                 fontsize=11, fontweight="bold", va="bottom")
    for ext in ("svg", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
