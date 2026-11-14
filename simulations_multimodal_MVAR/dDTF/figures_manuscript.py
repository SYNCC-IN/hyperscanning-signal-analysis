"""Manuscript figures not produced by the S-scripts.

fig_level1_signals.png  : one simulated Level-1 dyad (S1 setting): time courses (60 s) and model spectra.
fig_level2_signals.png  : Level-2 signals: EEG-like oscillations and their PSD, envelopes and their PSD,
                          cardiac (RSA-type) series and its PSD.
fig3_S1_recovery.png    : S1 Part B, planted structure and detection rates of dDTF and GPDC side by side.
fig_gain_conversion.png/.svg : Supplementary figure: prediction improvement (and Granger causality F) vs. gain per
                          edge type, and the median delta at 60 s vs. gain (from Results/comparison/gain_conversion.csv).
fig02_model_and_null    : Fig. 2 (schematic, drawn by fig2_model_and_null.py).
fig06 ... fig14         : publication figures of the results (S3-S13), SVG + PNG (300 dpi), from the summary and raw
                          CSVs of the full runs; no simulation is run here.
figS4_frequency_gap     : Fig. S4 (Supplement S3): S2 edge value relative to gap 0 and direction index vs. frequency gap.

Usage: python figures_manuscript.py [all | NAME ...] [OUT_DIR [RESULTS_DIR]]
  NAME is a function name below (level1, level2, fig3, fig_gain_conversion, fig02, fig06 ... fig14); default all.
  OUT_DIR defaults to <DRIVE>/Figures and RESULTS_DIR to <DRIVE>/Results.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.envelopes import downsample, filter_individual_band, hilbert_envelope
from src.sim_estimation import freq_matrices
from src.sim_generators import build_coupling, simulate_var
from src.sim_cardiac import simulate_cardiac_dyad, represent, FS_HI
from src.sim_inference import wilson_interval
from src.synthetic_mvar import edges_to_coupling, generate_coupled_oscillators, resonant_self_edges, self_ar2_coeffs

DRIVE = Path("/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/Mój dysk/TXT/2026_Multimodal_MVAR")
NAMES = [a for a in sys.argv[1:] if not ("/" in a or a.startswith("."))]
PATHS = [a for a in sys.argv[1:] if a not in NAMES]
OUT = Path(PATHS[0]) if PATHS else DRIVE / "Figures"; OUT.mkdir(parents=True, exist_ok=True)
RES = Path(PATHS[1]) if len(PATHS) > 1 else DRIVE / "Results"
LABELS = ["child brain", "caregiver brain", "child heart", "caregiver heart"]
COL = ["#1b7f79", "#5fb3ad", "#c77c02", "#f0b45a"]          # EEG teal, cardiac amber; child dark, caregiver light
FS = 2.5
BAND = (0.15, 0.5)
ENV_FS = 2.0                                                 # Level-2 log-amplitude VAR rate (as in S02/S11)
ENV_RADII, ENV_F0 = [0.80, 0.75], [0.30, 0.35]               # child, caregiver: within the S02/S11 ranges (0.70-0.85, 0.20-0.40 Hz)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def level1():
    rng = np.random.default_rng(11)
    radii, f0 = [0.55, 0.55, 0.86, 0.86], [0.30, 0.38, 0.32, 0.36]
    edges = [(1, 0, 1, 0.4), (3, 2, 1, 0.3), (3, 0, 2, 0.3)]
    coup = build_coupling(radii, f0, FS, edges, 3)
    x = simulate_var(coup, 150, rng)
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    t = np.arange(150) / FS
    freqs = np.linspace(0.005, FS / 2, 300)
    h = np.linalg.inv(freq_matrices(coup, freqs, FS))
    s = np.real(np.einsum("fij,fij->fi", h, h.conj()))           # diagonal of H H^H (unit innovations)
    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.2, 1], hspace=0.35, wspace=0.25)
    for i in range(4):
        ax = fig.add_subplot(gs[i, 0]); ax.plot(t, x[i], color=COL[i], lw=1)
        ax.set_ylabel(LABELS[i].replace(" ", "\n"), fontsize=8); ax.set_yticks([])
        if i < 3: ax.set_xticklabels([])
    ax.set_xlabel("time (s)")
    ax = fig.add_subplot(gs[:, 1])
    for i in range(4):
        #ax.semilogy(freqs, s[:, i] / s[:, i].max(), color=COL[i], lw=1.5, label=LABELS[i])
        ax.plot(freqs, s[:, i] / s[:, i].max(), color=COL[i], lw=1.5, label=LABELS[i])
    ax.axvspan(*BAND, color="0.85", zorder=0, label="analysed band")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("model spectrum (normalized)"); ax.legend(fontsize=7, frameon=False)
    fig.text(0.02, 0.96, "A", weight="bold", fontsize=11); fig.text(0.66, 0.96, "B", weight="bold", fontsize=11)
    fig.savefig(OUT / "fig_level1_signals.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def psd(x, fs, nper):
    f, p = welch(x, fs, nperseg=nper)
    return f, p


def level2():
    fs_c, dur = 128.0, 120.0
    cf = [8.0, 11.0]                                               # child 8 Hz, caregiver 11 Hz (3 Hz gap)
    edges = resonant_self_edges(ENV_RADII, ENV_F0, ENV_FS) + [(1, 0, 1, 0.5)]   # caregiver envelope drives child envelope
    raw = generate_coupled_oscillators(cf, edges_to_coupling(edges, 2), 4.0, fs_c, int(dur * fs_c), 1.5,
                                       envelope_fs=ENV_FS, carrier_order=4, seed=5)
    env = []
    for i in range(2):
        e = hilbert_envelope(filter_individual_band(raw[i], fs_c, cf[i], 1.5, 4))
        env.append((e, downsample(e, fs_c, FS)[0]))
    rng = np.random.default_rng(3)
    card, _, _ = simulate_cardiac_dyad(rng, 0.35, 0.25, dur)        # child breathes faster than caregiver
    ibi = [represent(card[i], "A", "ibi") for i in range(2)]
    fig, axes = plt.subplots(3, 2, figsize=(9, 7), gridspec_kw={"width_ratios": [2.2, 1], "hspace": 0.55, "wspace": 0.25})
    tt = np.arange(raw.shape[1]) / fs_c; sl = (tt >= 10) & (tt < 15)
    for i in range(2):
        off = 4 * (1 - i)
        z = raw[i] / raw[i].std()
        axes[0, 0].plot(tt[sl], z[sl] + off, color=COL[i], lw=0.6)
        axes[0, 0].plot(tt[sl], env[i][0][sl] / raw[i].std() + off, color="k", lw=0.8)
        f, p = psd(raw[i], fs_c, 512); 
        # axes[0, 1].semilogy(f, p / p.max(), color=COL[i], label=f"{LABELS[i]} ({cf[i]:.0f} Hz)")
        axes[0, 1].plot(f, p / p.max(), color=COL[i], label=f"{LABELS[i]} ({cf[i]:.0f} Hz)")
        e2 = env[i][1]; te = np.arange(len(e2)) / FS; m = te < 60
        axes[1, 0].plot(te[m], (e2[m] - e2.mean()) / e2.std() + off, color=COL[i], lw=1)
        f, p = psd(e2 - e2.mean(), FS, 128)
        axes[1, 1].plot(f, p / p.max(), color=COL[i], label=LABELS[i])
        a1, a2 = self_ar2_coeffs(ENV_RADII[i], ENV_F0[i], ENV_FS)          # own log-amplitude resonance (theory)
        zf = np.exp(-2j * np.pi * f / ENV_FS)
        p_ar = 1.0 / np.abs(1.0 - a1 * zf - a2 * zf ** 2) ** 2
        axes[1, 1].plot(f, p_ar / p_ar.max(), color=COL[i], lw=0.7, ls="--")
        y = ibi[i]; ti = np.arange(len(y)) / FS; m = ti < 60
        axes[2, 0].plot(ti[m], y[m] / y.std() + off, color=COL[2 + i], lw=1)
        f, p = psd(y, FS, 64); 
        # axes[2, 1].semilogy(f, p / p.max(), color=COL[2 + i], label=LABELS[2 + i])
        axes[2, 1].plot(f, p / p.max(), color=COL[2 + i], label=LABELS[2 + i])
    axes[0, 1].set_xlim(0, 20)
    for r, (ttl, xl) in enumerate([("EEG-like oscillations (5 s excerpt) with Hilbert envelope", "time (s)"),
                                   ("EEG envelopes at 2.5 Hz (60 s)", "time (s)"),
                                   ("cardiac (IBI) series at 2.5 Hz, option A (60 s)", "time (s)")]):
        axes[r, 0].set_title(ttl, fontsize=9, loc="left"); axes[r, 0].set_yticks([]); axes[r, 0].set_xlabel(xl)
        axes[r, 1].set_xlabel("frequency (Hz)"); axes[r, 1].set_ylabel("PSD (normalized)"); axes[r, 1].legend(fontsize=7, frameon=False)
    for r in (1, 2):
        axes[r, 1].axvspan(*BAND, color="0.85", zorder=0)
    for r, lab in enumerate("ABC"):
        fig.text(0.02, [0.9, 0.61, 0.32][r], lab, weight="bold", fontsize=11)
    fig.savefig(OUT / "fig_level2_signals.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig3():
    planted = {(1, 0): 0.4, (3, 2): 0.3, (3, 0): 0.3}
    rates = {}
    for est in ("dDTF", "GPDC"):
        d = pd.read_csv(RES / est / "S_01" / "partB_detection_rate_by_edge.csv").set_index("edge").study_reject
        m = np.full((4, 4), np.nan)
        names = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
        for s in range(4):
            for t in range(4):
                if s != t: m[t, s] = d[f"{names[s]}->{names[t]}"]
        rates[est] = m
    gain = np.full((4, 4), np.nan)
    for s in range(4):
        for t in range(4):
            if s != t: gain[t, s] = planted.get((s, t), 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    cmaps = ["Greys", "viridis", "viridis"]
    titles = ["A  Planted coupling (gain)", "B  dDTF", "C  GPDC"]
    for ax, mat, cm, ttl in zip(axes, [gain, rates["dDTF"], rates["GPDC"]], cmaps, titles):
        im = ax.imshow(mat, cmap=cm, vmin=0, vmax=0.5 if cm == "Greys" else 1)
        for (t, s), v in np.ndenumerate(mat):
            if t == s: continue
            pl = (s, t) in planted
            txt = (f"{v:.1f}" if v > 0 else "0") if cm == "Greys" else f"{v:.2f}"
            col = ("w" if v > 0.25 else "k") if cm == "Greys" else ("k" if v > 0.6 else "w")
            ax.text(s, t, txt, ha="center", va="center", fontsize=8, color=col, weight="bold" if pl else "normal")
            if pl and cm != "Greys":
                ax.add_patch(plt.Rectangle((s - .5, t - .5), 1, 1, fill=False, ec="red", lw=1.5))
        ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=40, ha="right")
        ax.set_yticks(range(4)); ax.set_yticklabels(LABELS if ax is axes[0] else [])
        ax.set_xlabel("source"); ax.set_title(ttl, fontsize=9, loc="left")
        ax.spines[:].set_visible(True)
    axes[0].set_ylabel("target")
    fig.colorbar(im, ax=axes[1:], fraction=0.025, label="proportion of studies with p < 0.05")
    fig.savefig(OUT / "fig3_S1_recovery.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_gain_conversion():
    table = pd.read_csv(RES / "comparison" / "gain_conversion.csv")
    sweep = table[table.simulation == "single-edge sweep"]
    s4 = table[table.simulation == "S4"].sort_values("gain")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    for edge_type, col, ls in [("brain->brain", COL[0], "-"), ("heart->heart", COL[2], "-"), ("heart->brain", "#7a5195", "--")]:
        sub = pd.concat([pd.DataFrame([dict(gain=0.0, PI_mean=0.0, PI_q05=0.0, PI_q95=0.0)]), sweep[sweep.edge_type == edge_type]])
        ax.fill_between(sub.gain, 100 * sub.PI_q05, 100 * sub.PI_q95, color=col, alpha=0.2, lw=0)
        ax.plot(sub.gain, 100 * sub.PI_mean, color=col, lw=1.5, ls=ls, label=edge_type.replace("->", " → "))
    ax.set_xlabel("coupling gain"); ax.set_ylabel("prediction improvement PI (%)")
    ax.set_title("A  Prediction improvement and Granger causality F", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    sec = ax.secondary_yaxis("right", functions=(lambda pi: -np.log(1.0 - np.clip(pi, 0, 99.9) / 100.0),
                                                 lambda f: 100.0 * (1.0 - np.exp(-f))))
    sec.set_ylabel("Granger causality F")
    ax = axes[1]
    for est, col in [("dDTF", "0.3"), ("GPDC", "#b2182b")]:
        ax.fill_between(s4.gain, s4[f"delta_{est}_q25"], s4[f"delta_{est}_q75"], color=col, alpha=0.2, lw=0)
        ax.plot(s4.gain, s4[f"delta_{est}_med"], color=col, lw=1.5, marker="o", ms=3, label=est)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("coupling gain"); ax.set_ylabel("Δ (Box–Cox units)")
    ax.set_title("B  Median Δ at 60 s (brain → brain, N = 25)", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_gain_conversion.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Publication figures (Fig. 2, Figs. 6-14): common style and helpers
# ---------------------------------------------------------------------------
PUB = {"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5,
       "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
       "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
       "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0, "lines.markersize": 3.5,
       "legend.frameon": False, "axes.titlepad": 3}
FULL_W = 7.48                                         # 190 mm
EST = ["dDTF", "GPDC"]
EST_COL = {"dDTF": "#1b9e9e", "GPDC": "#7a5cb8"}
EEG_COL, CARD_COL = "#1b9e9e", "#e08a1e"              # as in Fig. 1
FPR_BAND = (0.03, 0.07)
ALPHA = 0.05


def shade(colour, k, n):
    """k-th of n shades of a colour, from dark (k = 0) to light."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(rgb + (1 - rgb) * (0.6 * k / max(n - 1, 1)))


def wilson_err(rate, n):
    """Asymmetric error-bar half-lengths of 95% Wilson intervals for proportions `rate` out of `n`."""
    rate, n = np.broadcast_arrays(np.asarray(rate, float), np.asarray(n, float))
    bounds = np.array([wilson_interval(r * m, m) if np.isfinite(r) else (np.nan, np.nan) for r, m in zip(rate.ravel(), n.ravel())])
    return np.vstack([rate.ravel() - bounds[:, 0], bounds[:, 1] - rate.ravel()])


def wilson_band(ax, x, rate, n, colour):
    """Shaded 95% Wilson band around a rate curve."""
    err = wilson_err(rate, n)
    ax.fill_between(x, np.asarray(rate) - err[0], np.asarray(rate) + err[1], color=colour, alpha=0.18, lw=0)


def fpr_guides(ax, horizontal=True):
    """Shade the 3-7% band and draw alpha = 0.05 as a dashed line."""
    if horizontal:
        ax.axhspan(*FPR_BAND, color="0.88", zorder=0, lw=0); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6, zorder=0)
    else:
        ax.axvspan(*FPR_BAND, color="0.88", zorder=0, lw=0); ax.axvline(ALPHA, color="0.3", ls="--", lw=0.6, zorder=0)


def panel(ax, letter, x=-0.16, y=1.04):
    """Bold panel letter at the top left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    """Save a publication figure as SVG and 300-dpi PNG."""
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def comparison(name):
    """One table of Results/comparison (dDTF and GPDC columns side by side)."""
    return pd.read_csv(RES / "comparison" / name)


def dodge(i, n, width=0.6):
    """Offset of item i of n within a category of the given width."""
    return (i - (n - 1) / 2) * width / max(n, 1)


def fig02():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with plt.rc_context():                                # fig2_model_and_null sets its own style on import
        import fig2_model_and_null
        fig2_model_and_null.main(OUT, "fig02_model_and_null", 300)


def fig06():
    s3 = comparison("S_03_summary.csv")
    fw = comparison("S_03_summary_fwer.csv")
    n_rep = lambda n: 1000 if n in (10, 16) else 200
    cells = [(n, s) for n in (10, 25, 50) for s in (60, 120, 300)]
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 6.3))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.62, left=0.08, right=0.98, top=0.9, bottom=0.19)
        top = gs[0].subgridspec(1, 3, wspace=0.28)
        bottom = gs[1].subgridspec(1, 4, wspace=0.55, width_ratios=[1, 0.9, 0.9, 1])
        mean8 = s3.groupby(["n_dyads", "seg_s", "mode"]).mean(numeric_only=True)
        for j, (level, col, title) in enumerate([("study", "study_fpr", "study level"), ("dyad", "dyad_fpr", "dyad level"),
                                                 ("group", "group_fpr", "group level (two halves)")]):
            ax = fig.add_subplot(top[j]); fpr_guides(ax)
            for k, (est, mode) in enumerate([(e, m) for e in EST for m in ("pooled", "exclude")]):
                xs, ys, ns = [], [], []
                for c, (n, seg) in enumerate(cells):
                    if level == "group" and n == 10:
                        continue                          # 5 vs. 5 dyads: not shown
                    xs.append(c + dodge(k, 4)); ys.append(mean8.loc[(n, seg, mode), f"{col} {est}"])
                    ns.append(n_rep(n) * 8 * (n if level == "dyad" else 1))
                ax.errorbar(xs, ys, yerr=wilson_err(ys, ns), fmt="o", ms=3, lw=0.6, color=EST_COL[est],
                            mfc=EST_COL[est] if mode == "pooled" else "white", mew=0.8,
                            label=f"{est}, {'all pseudo-dyads' if mode == 'pooled' else 'excluding own members'}")
            ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"{s}" for n, s in cells], fontsize=6.5)
            for g, n in enumerate((10, 25, 50)):
                ax.text(3 * g + 1, -0.2, f"N = {n}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
                if g:
                    ax.axvline(3 * g - 0.5, color="0.75", lw=0.4)
            ax.set_ylim(0, 0.1); ax.set_xlim(-0.6, 8.6); ax.set_title(title)
            ax.set_xlabel("segment (s)", labelpad=14)
            if j == 0:
                ax.set_ylabel("false-positive rate"); panel(ax, "A")
                fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.995), fontsize=6.5)
        # B: group contrast with the within-group null vs. dyads per group
        ax = fig.add_subplot(bottom[0]); fpr_guides(ax)
        within = mean8.xs("within_group", level="mode")
        for est in EST:
            for seg, ls, mk in [(60, "-", "o"), (120, "--", "s"), (300, ":", "^")]:
                rows = [(n / 2, within.loc[(n, seg), f"group_fpr {est}"], n_rep(n) * 8) for n in (16, 25, 50) if (n, seg) in within.index]
                x, y, m = map(np.array, zip(*rows))
                ax.errorbar(x + (0.4 if est == "GPDC" else -0.4), y, yerr=wilson_err(y, m), color=EST_COL[est], ls=ls, marker=mk,
                            ms=3, lw=0.8, mfc="white" if est == "GPDC" else EST_COL[est], label=f"{est}, {seg} s")
        ax.set_xticks([8, 12.5, 25]); ax.set_xticklabels(["8", "12–13", "25"]); ax.set_ylim(0, 0.14)
        ax.set_xlabel("dyads per group"); ax.set_ylabel("false-positive rate"); ax.set_title("group contrast,\nwithin-group null")
        ax.legend(fontsize=5.5, ncol=2, loc="upper center", handlelength=1.8, columnspacing=0.6); panel(ax, "B", x=-0.3)
        # C: family-wise rate over the 8 cross-person edges
        cells_c = [(10, 60), (10, 120), (10, 300), (16, 60), (25, 60), (25, 120), (25, 300), (50, 60), (50, 120), (50, 300)]
        series = [("study", "pooled", "o", "study level"), ("group", "pooled", "s", "group contrast, pooled null"),
                  ("group", "within_group", "^", "group contrast, within-group null")]
        for j, (method, title, ylim) in enumerate([("none", "uncorrected", (0, 0.55)), ("maxT", "max-T", (0, 0.15))]):
            ax = fig.add_subplot(bottom[1 + j])
            if method == "maxT":
                fpr_guides(ax)
            for k, (est, (level, mode, mk, lab)) in enumerate([(e, s) for e in EST for s in series]):
                sub = fw[(fw.level == level) & (fw["mode"] == mode) & (fw.method == method)].set_index(["n_dyads", "seg_s"])
                pts = [(c, sub.loc[cell, f"fwer {est}"], n_rep(cell[0])) for c, cell in enumerate(cells_c)
                       if not (level == "group" and cell[0] == 10)]
                x, y, m = map(np.array, zip(*pts))
                ax.errorbar(x + dodge(k, 6, 0.75), y, yerr=wilson_err(y, m), fmt=mk, ms=2.6, lw=0.5, color=EST_COL[est],
                            mfc=EST_COL[est] if est == "dDTF" else "white", mew=0.7, label=f"{est}, {lab}")
            ax.set_xticks(range(len(cells_c))); ax.set_xticklabels([f"{n}/{s}" for n, s in cells_c], rotation=90, fontsize=5.5)
            ax.set_ylim(*ylim); ax.set_xlabel("N / segment (s)"); ax.set_title(f"family-wise rate,\n{title}")
            if j == 0:
                ax.set_ylabel("rate (any of 8 edges)"); panel(ax, "C", x=-0.32)
            else:
                ax.legend(fontsize=5.5, loc="upper center", ncol=3, bbox_to_anchor=(-0.35, -0.4), handletextpad=0.2, columnspacing=0.8)
        # D: distribution of the robust z under no coupling
        ax = fig.add_subplot(bottom[3])
        grid = np.linspace(-5, 5, 201)
        ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color="0.3", ls="--", lw=0.7, label="standard normal")
        for est in EST:
            npz = np.load(RES / est / "S_03" / "z_values_pooled.npz", allow_pickle=True)
            z = np.concatenate([np.asarray(npz[k], float).ravel() for k in npz.files if k != "edges"])
            ax.hist(z[np.isfinite(z)], bins=np.linspace(-5, 5, 81), density=True, histtype="step", color=EST_COL[est], lw=1.0, label=est)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 0.55); ax.set_xlabel("robust z"); ax.set_ylabel("density")
        ax.set_title("robust z,\nno coupling"); ax.legend(fontsize=5.5, loc="upper left"); panel(ax, "D", x=-0.3)
        save(fig, "fig06_calibration")


def fig07():
    s4 = comparison("S_04_summary.csv")
    s4b = comparison("S_04b_summary.csv")
    n_rep, n_dyads_b = 200, 25
    seg_style = {60: ("-", "o"), 120: ("--", "s"), 300: (":", "^")}
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 7.6))
        gs = fig.add_gridspec(3, 1, hspace=0.62, left=0.08, right=0.98, top=0.95, bottom=0.07, height_ratios=[1, 1, 1])
        row1 = gs[0].subgridspec(1, 4, wspace=0.35)
        for b, (design, ns, title) in enumerate([("A", (10, 25, 50), "design A, one group"),
                                                ("B", (25, 50), "design B, group contrast")]):
            for e, est in enumerate(EST):
                ax = fig.add_subplot(row1[2 * b + e]); ax.axhline(0.8, color="0.6", lw=0.5, ls=":")
                for k, n in enumerate(ns):
                    for seg, (ls, mk) in seg_style.items():
                        sub = s4[(s4.design == design) & (s4.n_dyads == n) & (s4.seg_s == seg)].sort_values("gain")
                        col = shade(EST_COL[est], k, len(ns))
                        wilson_band(ax, sub.gain, sub[f"power {est}"], n_rep, col)
                        ax.plot(sub.gain, sub[f"power {est}"], color=col, ls=ls, marker=mk, ms=2.5,
                                label=f"N = {n}, {seg} s")
                ax.set_ylim(-0.02, 1.02); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_xlabel("gain")
                ax.set_title(f"{title}\n{est}")
                if e == 0:
                    ax.set_ylabel("power (gain 0: false-positive rate)" if b == 0 else "power"); panel(ax, "AB"[b], x=-0.3)
                if e == 1:
                    ax.legend(fontsize=5, loc="lower right", handlelength=1.8, labelspacing=0.2)
        row2 = gs[1].subgridspec(1, 3, wspace=0.38)
        # C: max-T vs. per-edge power at 60 s
        ax = fig.add_subplot(row2[0]); ax.plot([0, 1], [0, 1], color="0.6", lw=0.5)
        for est in EST:
            for design, mk, ns in [("A", "o", (10, 25, 50)), ("B", "s", (25, 50))]:
                sub = s4[(s4.design == design) & (s4.seg_s == 60) & s4.n_dyads.isin(ns) & (s4.gain > 0)]
                ax.errorbar(sub[f"power {est}"], sub[f"power_maxT {est}"], yerr=wilson_err(sub[f"power_maxT {est}"], n_rep),
                            fmt=mk, ms=3, lw=0.5, color=EST_COL[est], mfc=EST_COL[est] if design == "A" else "white", mew=0.7,
                            label=f"{est}, design {design}")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.set_xlabel("power per edge"); ax.set_ylabel("power after max-T (8 edges)"); ax.set_title("max-T correction, 60 s")
        ax.legend(fontsize=5.5, loc="upper left"); panel(ax, "C", x=-0.3)
        # D, E: partial coupling
        for j, (variant, xlabel, title) in enumerate([("partial_dyads", "share of dyads coupled", "coupling in part of the dyads"),
                                                      ("partial_time", "share of the segment coupled", "coupling in part of the segment")]):
            ax = fig.add_subplot(row2[1 + j]); ax.axhline(0.8, color="0.6", lw=0.5, ls=":")
            for est in EST:
                for gain, mk in [(0.3, "o"), (0.4, "s")]:
                    sub = s4b[(s4b.variant == variant) & (s4b.gain == gain)].sort_values("level")
                    ax.errorbar(sub.level + (0.008 if est == "GPDC" else -0.008), sub[f"power {est}"], yerr=wilson_err(sub[f"power {est}"], n_rep),
                                color=EST_COL[est], marker=mk, ms=3, lw=0.8, label=f"{est}, gain {gain}, per edge")
                    ax.errorbar(sub.level + (0.008 if est == "GPDC" else -0.008), sub[f"power_maxT {est}"],
                                yerr=wilson_err(sub[f"power_maxT {est}"], n_rep), color=EST_COL[est], marker=mk, ms=3, lw=0.8,
                                ls="--", mfc="white", label=f"{est}, gain {gain}, max-T")
            ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel(xlabel); ax.set_title(title)
            if j == 0:
                ax.set_ylabel("power"); panel(ax, "D", x=-0.3)
            else:
                partial_handles = ax.get_legend_handles_labels(); panel(ax, "E", x=-0.3)
        row3 = gs[2].subgridspec(1, 3, wspace=0.38)
        # F: dyad-level detection in coupled and uncoupled dyads; reverse-edge rate vs. coupled share
        ax = fig.add_subplot(row3[0]); fpr_guides(ax)
        dyads = s4b[s4b.variant == "partial_dyads"]
        for est in EST:
            for gain, mk in [(0.3, "o"), (0.4, "s")]:
                sub = dyads[dyads.gain == gain].sort_values("level")
                ax.errorbar(sub.level, sub[f"dyad_rate_coupled {est}"], yerr=wilson_err(sub[f"dyad_rate_coupled {est}"], n_rep * n_dyads_b * sub.level),
                            color=EST_COL[est], marker=mk, ms=3, lw=0.8, label=f"{est}, gain {gain}")
                unc = sub[sub.level < 1]
                ax.errorbar(unc.level, unc[f"dyad_rate_uncoupled {est}"],
                            yerr=wilson_err(unc[f"dyad_rate_uncoupled {est}"], n_rep * n_dyads_b * (1 - unc.level)),
                            color=EST_COL[est], marker=mk, ms=3, lw=0.8, ls="--", mfc="white")
        ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_ylim(0, 1.0); ax.set_xlabel("share of dyads coupled")
        ax.set_ylabel("dyads detected (dyad level)"); ax.set_title("coupled (solid), uncoupled (dashed)")
        ax.legend(fontsize=5.5, loc="center right"); panel(ax, "F", x=-0.3)
        ax = fig.add_subplot(row3[1])
        for est in EST:
            for variant, ls, lab in [("partial_dyads", "-", "share of dyads"), ("partial_time", "--", "share of segment")]:
                sub = s4b[(s4b.variant == variant) & (s4b.gain == 0.4)].sort_values("level")
                ax.errorbar(sub.level, sub[f"reverse_rate {est}"], yerr=wilson_err(sub[f"reverse_rate {est}"], n_rep), color=EST_COL[est],
                            ls=ls, marker="o", ms=3, lw=0.8, mfc=EST_COL[est] if ls == "-" else "white", label=f"{est}, {lab}")
        ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("coupled share")
        ax.set_ylabel("reverse edge detected"); ax.set_title("reverse edge, gain 0.4"); ax.legend(fontsize=5.5, loc="center right")
        ax = fig.add_subplot(row3[2]); ax.axis("off")
        ax.legend(*partial_handles, loc="center", fontsize=6, title="D, E", title_fontsize=6.5, handlelength=2)
        save(fig, "fig07_power")


S8_ORDER = ["baseline", "estimator=DTF", "estimator=ffDTF", "estimator=dDTF", "estimator=GPDC", "order=2", "order=3", "order=6",
            "order=8", "order=AIC", "order=HQ", "order=SC", "window=6s", "window=15s", "window=20s", "window=whole segment",
            "overlap=0.25", "overlap=0.75", "fit=Yule-Walker", "fit=Yule-Walker, window=20s", "detrend=linear", "detrend=constant",
            "box-cox=none", "band=0.05-0.15", "band=0.1-0.3", "band=0.3-0.5"]
S8_LABEL = {"baseline": "default", "estimator=DTF": "DTF",
            "estimator=ffDTF": "ffDTF", "estimator=dDTF": "dDTF (GPDC run)", "estimator=GPDC": "GPDC (dDTF run)", "order=2": "p = 2",
            "order=3": "p = 3", "order=6": "p = 6", "order=8": "p = 8", "order=AIC": "p by AIC", "order=HQ": "p by HQ",
            "order=SC": "p by SC", "window=6s": "6 s windows", "window=15s": "15 s windows", "window=20s": "20 s windows",
            "window=whole segment": "no windows", "overlap=0.25": "25% overlap", "overlap=0.75": "75% overlap",
            "fit=Yule-Walker": "Yule–Walker, 10 s", "fit=Yule-Walker, window=20s": "Yule–Walker, 20 s",
            "detrend=linear": "linear detrending", "detrend=constant": "constant detrending", "box-cox=none": "no Box–Cox", "band=0.05-0.15": "band 0.05–0.15 Hz",
            "band=0.1-0.3": "band 0.1–0.3 Hz", "band=0.3-0.5": "band 0.3–0.5 Hz"}


def s8_dots(ax, s8, column, n, guides=None):
    """Paired dDTF/GPDC markers with Wilson intervals, one row per S8 variant (default highlighted)."""
    y = np.arange(len(S8_ORDER))
    table = s8.set_index("variant").loc[S8_ORDER]
    ax.axhspan(-0.5, 0.5, color="#fff2cc", zorder=0, lw=0)
    if guides == "fpr":
        fpr_guides(ax, horizontal=False)
    elif guides == "alpha":
        ax.axvline(ALPHA, color="0.3", ls="--", lw=0.6)
    for k, est in enumerate(EST):
        r = table[f"{column} {est}"].to_numpy()
        ax.errorbar(r, y + (0.18 if k else -0.18), xerr=wilson_err(r, n), fmt="o", ms=2.8, lw=0.6, color=EST_COL[est], label=est)
    ax.set_yticks(y); ax.set_ylim(len(S8_ORDER) - 0.5, -0.5)


def fig08():
    s5 = comparison("S_05_summary.csv")
    sweep = comparison("S_05_summary_sweep.csv")
    s8 = comparison("S_08_summary.csv")
    s11 = comparison("S_11_summary_partA.csv")
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 7.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.0], wspace=0.2, left=0.155, right=0.975, top=0.95, bottom=0.07)
        ax = fig.add_subplot(gs[0])
        s8_dots(ax, s8, "reverse_detection", 600, guides="alpha")
        ax.set_yticklabels([S8_LABEL[v] for v in S8_ORDER], fontsize=6.3)
        ax.set_xlim(-0.02, 0.8); ax.set_xlabel("reverse edge detected"); ax.set_title("reverse edges, S8 variants")
        ax.legend(loc="center", fontsize=6); panel(ax, "A", x=-0.6, y=1.01)
        right = gs[1].subgridspec(3, 2, hspace=0.75, wspace=0.42)
        # B: absent indirect edge of the cascade
        ax = fig.add_subplot(right[0, 0]); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        casc = s5[(s5.scenario == "cascade") & (s5.role == "absent")]
        lines = [("4-node [est]", "dDTF", EST_COL["dDTF"], "-", "o", "dDTF"), ("4-node [est]", "GPDC", EST_COL["GPDC"], "-", "o", "GPDC"),
                 ("4-node ffDTF", "dDTF", "0.35", "--", "s", "ffDTF"), ("4-node DTF", "dDTF", "0.6", ":", "^", "DTF"),
                 ("bivariate [est]", "dDTF", EST_COL["dDTF"], "--", "D", "bivariate dDTF"),
                 ("bivariate [est]", "GPDC", EST_COL["GPDC"], "--", "D", "bivariate GPDC")]
        for analysis, est, col, ls, mk, lab in lines:
            sub = casc[casc.analysis == analysis].sort_values("gain")
            ax.errorbar(sub.gain, sub[f"study_rate {est}"], yerr=wilson_err(sub[f"study_rate {est}"], 200), color=col, ls=ls, marker=mk,
                        ms=3, lw=0.8, mfc="white" if ls != "-" else col, label=lab)
        ax.set_xticks([0.2, 0.4, 0.6]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("gain"); ax.set_ylabel("absent edge detected")
        ax.set_title("cascade: indirect edge"); ax.legend(fontsize=5, loc="center right", labelspacing=0.2); panel(ax, "B", x=-0.32)
        # C: S11 Part A, all methods
        ax = fig.add_subplot(right[0, 1]); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        cats = [("null_stimulus", "null_edge", "none +\nstimulus"), ("power", "planted", "planted"),
                ("indirect", "planted", "cascade,\nplanted"), ("indirect", "absent_indirect", "cascade,\nindirect")]
        methods = [("[est] (4-node)", "dDTF", EST_COL["dDTF"], "o", "dDTF, 4 nodes"), ("[est] (4-node)", "GPDC", EST_COL["GPDC"], "o", "GPDC, 4 nodes"),
                   ("pairwise Granger", "dDTF", "0.2", "s", "pairwise Granger"), ("lagged cross-correlation", "dDTF", "0.45", "^", "lagged cross-corr."),
                   ("envelope coherence", "dDTF", "0.7", "D", "envelope coherence")]
        for k, (method, est, col, mk, lab) in enumerate(methods):
            vals = np.array([s11[(s11.scenario == sc) & (s11.role == role) & (s11.method == method)][f"study_rate {est}"].iloc[0] for sc, role, _ in cats])
            ax.errorbar(np.arange(4) + dodge(k, 5, 0.7), vals, yerr=wilson_err(vals, 100), fmt=mk, ms=3, lw=0.6, color=col, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[2] for c in cats], fontsize=5.8); ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("edge detected"); ax.set_title("joint model vs. simpler methods"); ax.legend(fontsize=5, loc="center left", labelspacing=0.2)
        panel(ax, "C", x=-0.32)
        # D: omitted driver, without and with Z
        ax = fig.add_subplot(right[1, 0]); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        om = s5[s5.scenario == "omitted_driver"]
        for est in EST:
            for analysis, ls, lab in [("4-node [est]", "-", "without Z"), ("5-node [est] (Z included)", "--", "with Z")]:
                for role, mk in [("absent", "o"), ("absent reverse", "s")]:
                    sub = om[(om.analysis == analysis) & (om.role == role)].sort_values("gain")
                    ax.errorbar(sub.gain, sub[f"study_rate {est}"], yerr=wilson_err(sub[f"study_rate {est}"], 200), color=EST_COL[est], ls=ls,
                                marker=mk, ms=3, lw=0.8, mfc="white" if role == "absent reverse" else EST_COL[est],
                                )
        handles = [matplotlib.lines.Line2D([], [], color=EST_COL[e], lw=1, label=e) for e in EST] + [
            matplotlib.lines.Line2D([], [], color="0.3", ls="-", lw=0.8, label="without Z (4 nodes)"),
            matplotlib.lines.Line2D([], [], color="0.3", ls="--", lw=0.8, label="with Z (5 nodes)"),
            matplotlib.lines.Line2D([], [], color="0.3", ls="", marker="o", ms=3, label="cg → child heart"),
            matplotlib.lines.Line2D([], [], color="0.3", ls="", marker="s", ms=3, mfc="white", label="child → cg heart")]
        ax.set_xticks([0.2, 0.4, 0.6]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("gain of Z"); ax.set_ylabel("heart–heart edge detected")
        ax.set_title("omitted driver Z (60 s, p = 4)"); ax.legend(handles=handles, fontsize=5, loc="center left", labelspacing=0.15, bbox_to_anchor=(0.12, 0.6))
        panel(ax, "D", x=-0.32)
        # E: sweep with Z included: segment length and model order, with collinearity diagnostics
        withz = sweep[(sweep.analysis == "5-node [est] (Z included)") & (sweep.role == "absent")]
        for j, (axis, sel, xlabel) in enumerate([("seg_s", withz.p == 4, "segment (s), p = 4"), ("p", withz.seg_s == 60, "model order p, 60 s")]):
            ax = fig.add_subplot(right[1 + j, 1] if j == 0 else right[2, 0]); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
            for est in EST:
                for gain, mk in [(0.4, "o"), (0.6, "s")]:
                    sub = withz[sel & (withz.gain == gain)].sort_values(axis)
                    ax.errorbar(sub[axis], sub[f"study_rate {est}"], yerr=wilson_err(sub[f"study_rate {est}"], 200), color=EST_COL[est],
                                marker=mk, ms=3, lw=0.8, mfc=EST_COL[est] if gain == 0.6 else "white", label=f"{est}, gain {gain}")
            ax.set_xticks(sorted(withz[axis].unique())); ax.set_ylim(-0.02, 1.05); ax.set_xlabel(xlabel)
            ax.set_ylabel("cg → child heart detected"); ax.set_title("with Z: " + ("segment length" if j == 0 else "model order"))
            if j == 0:
                ax.legend(fontsize=5, loc="upper left"); panel(ax, "E", x=-0.32)
        ax = fig.add_subplot(right[2, 1])
        diag = sweep[(sweep.analysis == "5-node [est] (Z included)") & (sweep.role == "absent") & (sweep.gain == 0.6)]
        for axis, sel, mk, lab in [("seg_s", diag.p == 4, "o", "vs. segment"), ("p", diag.seg_s == 60, "s", "vs. order")]:
            sub = diag[sel].sort_values(axis)
            ax.plot(range(3), sub["vif_cg_heart_real dDTF"], color="0.25", marker=mk, ms=3, label=f"VIF {lab}")
            ax.plot(range(3), sub["cond_real dDTF"], color="0.6", marker=mk, ms=3, ls="--", label=f"condition no. {lab}")
        ax.set_xticks(range(3)); ax.set_xticklabels(["60 s / p 2", "120 s / p 4", "300 s / p 6"], fontsize=6)
        ax.set_ylabel("collinearity (real dyads)"); ax.set_title("5-node model, gain 0.6")
        ax.legend(fontsize=5, loc="upper left", labelspacing=0.15); ax.set_ylim(0, 75)
        save(fig, "fig08_specificity")


def proxies(markers, styles, second_ls="--"):
    """Legend handles: estimator colours, marker meanings and two line styles."""
    handles = [matplotlib.lines.Line2D([], [], color=EST_COL[e], lw=1, label=e) for e in EST]
    handles += [matplotlib.lines.Line2D([], [], color="0.3", ls="", marker=mk, ms=3, mfc="0.3" if filled else "white", label=lab)
                for mk, filled, lab in markers]
    handles += [matplotlib.lines.Line2D([], [], color="0.3", ls=ls, lw=0.8, label=lab) for ls, lab in zip(("-", second_ls), styles)]
    return handles


def fig09():
    s6 = comparison("S_06_summary.csv")
    nc = comparison("S_06b_summary_no_coupling.csv")
    cp = comparison("S_06b_summary_coupled.csv")
    s7 = comparison("S_07_summary.csv")
    s7 = s7[s7.sd_log == 0.5]
    n_rep = 200
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 7.6))
        gs = fig.add_gridspec(3, 1, hspace=0.6, left=0.08, right=0.98, top=0.95, bottom=0.07)
        row1 = gs[0].subgridspec(1, 2, wspace=0.3)
        # A: smoothness, uncorrected vs. corrected asymmetry
        ax = fig.add_subplot(row1[0]); ax.axhline(0, color="0.3", lw=0.5)
        for est in EST:
            for z, mk in [(True, "o"), (False, "s")]:
                sub = s6[s6.zscore == z].sort_values("radius")
                ax.plot(sub.radius, sub[f"uncorrected_asymmetry {est}"], color=EST_COL[est], marker=mk, ms=3,
                        mfc=EST_COL[est] if z else "white", label=f"{est}, {'z-scored' if z else 'not z-scored'}, uncorrected")
                ax.plot(sub.radius, sub[f"corrected_asymmetry {est}"], color=EST_COL[est], marker=mk, ms=3, ls="--",
                        mfc=EST_COL[est] if z else "white", lw=0.7, label=f"{est}, {'z-scored' if z else 'not z-scored'}, corrected")
        ax.set_xticks(sorted(s6.radius.unique())); ax.set_xlabel("pole radius of the caregiver heart")
        ax.set_ylabel("direction asymmetry"); ax.set_title("smoothness: direction asymmetry")
        ax.legend(handles=proxies([("o", True, "z-scored"), ("o", False, "not z-scored")], ("uncorrected", "corrected")),
                  fontsize=5.5, loc="upper left", labelspacing=0.2); ax.set_ylim(-0.06, 0.32); panel(ax, "A", x=-0.13)
        # B: false-positive rate of both edges vs. pole radius
        ax = fig.add_subplot(row1[1]); fpr_guides(ax)
        for est in EST:
            for col_name, mk, lab in [("fpr_smooth_to_rough", "o", "smooth → rough"), ("fpr_rough_to_smooth", "s", "rough → smooth")]:
                for z, ls in [(True, "-"), (False, ":")]:
                    sub = s6[s6.zscore == z].sort_values("radius")
                    off = 0.004 if est == "GPDC" else -0.004
                    ax.errorbar(sub.radius + off, sub[f"{col_name} {est}"], yerr=wilson_err(sub[f"{col_name} {est}"], n_rep) if z else None,
                                color=EST_COL[est], marker=mk, ms=3, lw=0.8 if z else 0.5, ls=ls, mfc=EST_COL[est] if z else "white",
                                label=f"{est}, {lab}{'' if z else ' (not z-scored)'}")
        ax.set_xticks(sorted(s6.radius.unique())); ax.set_ylim(0, 0.14); ax.set_xlabel("pole radius of the caregiver heart")
        ax.set_ylabel("false-positive rate (corrected)"); ax.set_title("smoothness: false positives after correction")
        ax.set_ylim(0, 0.16)
        ax.legend(handles=proxies([("o", True, "smooth → rough"), ("s", True, "rough → smooth")], ("z-scored", "not z-scored"), ":"),
                  fontsize=5.5, loc="upper left", ncol=2, labelspacing=0.2); panel(ax, "B", x=-0.13)
        row2 = gs[1].subgridspec(1, 3, wspace=0.38)
        # C: shared stimulus, same-film nulls
        ratio_ls = {1.0: "-", 2.0: "--", 3.0: ":"}
        for j, (col_name, title) in enumerate([("study_fpr", "stimulus: study level"), ("group_fpr", "stimulus: group contrast")]):
            ax = fig.add_subplot(row2[j]); fpr_guides(ax)
            for est in EST:
                for mode, mk, lab in [("pooled", "o", "pooled"), ("within_group", "s", "within-group")]:
                    for ratio, ls in ratio_ls.items():
                        sub = s7[(s7.null_source == "same film") & (s7["mode"] == mode) & (s7.ratio == ratio)].sort_values("gain")
                        ax.plot(sub.gain + (0.02 if est == "GPDC" else -0.02), sub[f"{col_name} {est}"], color=EST_COL[est], ls=ls,
                                marker=mk, ms=2.8, lw=0.8, mfc=EST_COL[est] if mode == "pooled" else "white",
                                label=f"{est}, {lab}, ratio {ratio:g}")
                        if ratio == 3.0:
                            wilson_band(ax, sub.gain + (0.02 if est == "GPDC" else -0.02), sub[f"{col_name} {est}"], n_rep, EST_COL[est])
            ax.set_xticks([0, 0.5, 1, 2]); ax.set_ylim(0, 0.62); ax.set_xlabel("stimulus gain"); ax.set_title(title)
            if j == 0:
                ax.set_ylabel("false-positive rate"); panel(ax, "C", x=-0.3)
                handles = [matplotlib.lines.Line2D([], [], color=EST_COL[e], lw=1, label=e) for e in EST] + [
                    matplotlib.lines.Line2D([], [], color="0.3", marker="o", ls="", ms=3, label="pooled null"),
                    matplotlib.lines.Line2D([], [], color="0.3", marker="s", ls="", ms=3, mfc="white", label="within-group null")] + [
                    matplotlib.lines.Line2D([], [], color="0.3", ls=ls, lw=0.8, label=f"group gain ratio {r:g}") for r, ls in ratio_ls.items()]
                ax.legend(handles=handles, fontsize=5, loc="upper left", labelspacing=0.2)
        # D: other-film null
        ax = fig.add_subplot(row2[2]); fpr_guides(ax)
        for est in EST:
            for ratio, ls in ratio_ls.items():
                sub = s7[(s7.null_source == "other film") & (s7.ratio == ratio)].sort_values("gain")
                ax.errorbar(sub.gain + (0.02 if est == "GPDC" else -0.02), sub[f"study_fpr {est}"], yerr=wilson_err(sub[f"study_fpr {est}"], n_rep),
                            color=EST_COL[est], ls=ls, marker="o", ms=2.8, lw=0.8, label=f"{est}, ratio {ratio:g}")
        ax.set_xticks([0, 0.5, 1, 2]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("stimulus gain"); ax.set_ylabel("false-positive rate")
        ax.set_title("null from another film: study level"); ax.legend(fontsize=5, loc="lower right", labelspacing=0.2); panel(ax, "D", x=-0.3)
        row3 = gs[2].subgridspec(1, 2, wspace=0.3)
        # E: unequal noise, asymmetry
        ax = fig.add_subplot(row3[0]); ax.axhline(0, color="0.3", lw=0.5)
        for est in EST:
            for noise_set, mk, lab in [("cg_heart", "o", "noise on caregiver heart"), ("child_brain", "s", "noise on child brain")]:
                sub = nc[nc.noise_set == noise_set].sort_values("noise_sd")
                ax.plot(sub.noise_sd, sub[f"uncorrected_asymmetry {est}"], color=EST_COL[est], marker=mk, ms=3, label=f"{est}, {lab}, uncorrected")
                ax.plot(sub.noise_sd, sub[f"corrected_asymmetry {est}"], color=EST_COL[est], marker=mk, ms=3, ls="--", lw=0.7, mfc="white",
                        label=f"{est}, {lab}, corrected")
        ax.set_xticks(sorted(nc.noise_sd.unique())); ax.set_xlabel("noise SD / node SD"); ax.set_ylim(-0.06, 0.32)
        ax.set_ylabel("direction asymmetry"); ax.set_title("unequal noise, no coupling (pole radius 0.9)")
        ax.legend(handles=proxies([("o", True, "noise on caregiver heart"), ("s", True, "noise on child brain")], ("uncorrected", "corrected")),
                  fontsize=5.5, loc="center right", labelspacing=0.2, bbox_to_anchor=(1.0, 0.42)); panel(ax, "E", x=-0.13)
        # F: unequal noise with planted edges: power and reverse-edge rate
        ax = fig.add_subplot(row3[1]); ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        edge_mk = {"cg:ROI->child:ROI": ("o", "cg brain → child brain"), "cg:HRV->child:HRV": ("s", "cg heart → child heart"),
                   "cg:HRV->child:ROI": ("^", "cg heart → child brain")}
        for est in EST:
            for edge, (mk, lab) in edge_mk.items():
                sub = cp[cp.edge == edge].sort_values("noise_sd")
                off = 0.015 if est == "GPDC" else -0.015
                ax.errorbar(sub.noise_sd + off, sub[f"reverse_rate {est}"], yerr=wilson_err(sub[f"reverse_rate {est}"], n_rep), color=EST_COL[est],
                            marker=mk, ms=3, lw=0.8, mfc="white", ls="--", label=f"{est}, reverse of {lab}")
            sub = cp.groupby("noise_sd")[f"power {est}"].min()
            ax.plot(sub.index, sub.to_numpy(), color=EST_COL[est], lw=1.2, label=f"{est}, power (all three edges)")
        ax.set_xticks([0, 0.5, 1]); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("noise SD / node SD (both caregiver nodes)")
        ax.set_ylabel("rate"); ax.set_title("unequal noise, three planted edges")
        ax.legend(fontsize=4.8, loc="center right", labelspacing=0.2); panel(ax, "F", x=-0.18)
        save(fig, "fig09_confounds")


def fig10():
    s8 = comparison("S_08_summary.csv")
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 5.4), sharey=True, gridspec_kw={"wspace": 0.12})
        fig.subplots_adjust(left=0.2, right=0.98, top=0.93, bottom=0.09)
        for ax, (column, n, title, guides, xlim) in zip(axes, [
                ("null_fpr", 6 * 200, "false-positive rate (no coupling)", "fpr", (0, 0.1)),
                ("power_planted", 3 * 200, "power (three planted edges)", None, (0.9, 1.005)),
                ("reverse_detection", 3 * 200, "reverse edges detected", "alpha", (-0.02, 0.8))]):
            s8_dots(ax, s8, column, n, guides)
            ax.set_xlim(*xlim); ax.set_title(title); ax.set_xlabel("rate")
            ax.set_xticks({"null_fpr": [0, 0.025, 0.05, 0.075], "power_planted": [0.92, 0.96, 1.0]}.get(column, [0, 0.2, 0.4, 0.6, 0.8]))
        axes[0].set_yticklabels([S8_LABEL[v] for v in S8_ORDER], fontsize=6.5)
        axes[2].legend(loc="center", fontsize=6.5)
        for ax, letter in zip(axes, "ABC"):
            panel(ax, letter, x=-0.08, y=1.01)
        save(fig, "fig10_sensitivity")


S11_METHODS = [("[est] (4-node)", "dDTF", EST_COL["dDTF"], "MVAR, dDTF"), ("[est] (4-node)", "GPDC", EST_COL["GPDC"], "MVAR, GPDC"),
               ("envelope coherence", "dDTF", "0.7", "coherence"), ("lagged cross-correlation", "dDTF", "0.5", "lagged xcorr"),
               ("pairwise Granger", "dDTF", "0.3", "pairwise GC"), ("PLV (raw signals)", "dDTF", "#c0392b", "PLV (raw)")]


def fig11():
    s11 = comparison("S_11_summary_partB.csv")
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.27)
        for ax, (truth, title) in zip(axes, [("amplitude", "amplitude coupling"), ("phase", "phase coupling"), ("none", "no coupling")]):
            if truth == "none":
                fpr_guides(ax)
            else:
                ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
            for k, (method, est, col, lab) in enumerate(S11_METHODS):
                for gap, mfc in [(0.0, col), (3.0, "white")]:
                    r = s11[(s11.truth == truth) & (s11.gap == gap) & (s11.method == method)][f"study_rate {est}"].iloc[0]
                    ax.errorbar(k + (0.13 if gap else -0.13), r, yerr=wilson_err([r], 100), fmt="o", ms=3.5, lw=0.6, color=col, mfc=mfc, mew=0.8)
            ax.set_xticks(range(len(S11_METHODS))); ax.set_xticklabels([m[3] for m in S11_METHODS], rotation=55, ha="right", fontsize=6.3)
            ax.set_ylim(-0.02, 1.05); ax.set_title(title); ax.set_xlim(-0.6, len(S11_METHODS) - 0.4)
        axes[0].set_ylabel("coupling detected (study level)")
        axes[2].legend(handles=[matplotlib.lines.Line2D([], [], color="0.3", ls="", marker="o", ms=3.5, label="gap 0 Hz"),
                                matplotlib.lines.Line2D([], [], color="0.3", ls="", marker="o", ms=3.5, mfc="white", label="gap 3 Hz")],
                       loc="upper right", fontsize=6)
        for ax, letter in zip(axes, "ABC"):
            panel(ax, letter, x=-0.1)
        save(fig, "fig11_envelope_vs_phase")


def fig12():
    s9 = comparison("S_09_summary.csv")
    s10 = comparison("S_10_summary.csv")
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 6.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.45, left=0.17, right=0.98, top=0.94, bottom=0.06)
        top = gs[0].subgridspec(1, 3, wspace=0.12)
        for j, (truth, title) in enumerate([("none", "no coupling"), ("pulling", "frequency pulling (cg → child)"),
                                            ("amplitude", "RSA amplitude (cg → child)")]):
            ax = fig.add_subplot(top[j])
            if truth == "none":
                fpr_guides(ax)
            else:
                ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
            for est in EST:
                for col_name, ls, mfc, lab in [("forward_rate", "-", None, "cg → child heart"), ("reverse_rate", "--", "white", "child → cg heart")]:
                    sub = s9[(s9.truth == truth) & (s9.band == "mid")].sort_values("gap")
                    ax.errorbar(sub.gap + (0.004 if est == "GPDC" else -0.004), sub[f"{col_name} {est}"], yerr=wilson_err(sub[f"{col_name} {est}"], 100),
                                color=EST_COL[est], ls=ls, marker="o", ms=3, lw=0.8, mfc=mfc or EST_COL[est], label=f"{est}, {lab}")
            ax.set_ylim(-0.02, 1.05); ax.set_xticks(sorted(s9.gap.unique())); ax.set_xlabel("breathing-rate gap (Hz)"); ax.set_title(title)
            if j == 0:
                ax.set_ylabel("edge detected (study level)"); panel(ax, "A", x=-0.42)
            else:
                ax.set_yticklabels([])
            if j == 2:
                ax.legend(fontsize=5.5, loc="upper right", labelspacing=0.2)
        bottom = gs[1].subgridspec(1, 2, wspace=0.1)
        rows = [(o, tr) for o in ("A", "Aprime", "B") for tr in ("linear", "amp_to_eeg", "none")]
        cols = [(d, s) for d in ("forward_rate", "reverse_rate") for s in (0.3, 0.6, 1.0)]
        opt_lab = {"A": "A", "Aprime": "A′", "B": "B"}
        tr_lab = {"linear": "breathing-paced", "amp_to_eeg": "RSA amplitude", "none": "none"}
        for j, est in enumerate(EST):
            ax = fig.add_subplot(bottom[j])
            mat = np.full((len(rows), len(cols)), np.nan)
            for r, (o, tr) in enumerate(rows):
                for c, (d, s) in enumerate(cols):
                    sel = s10[(s10.option == o) & (s10.truth == tr) & (s10.strength == (0.0 if tr == "none" else s))]
                    if tr == "none" and s != 0.3:
                        continue
                    mat[r, c] = sel[f"{d} {est}"].iloc[0]
            ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="auto")
            for (r, c), v in np.ndenumerate(mat):
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=5.8, color="w" if v > 0.55 else "k")
            for r in (2.5, 5.5):
                ax.axhline(r, color=EST_COL[est], lw=1)
            ax.axvline(2.5, color=EST_COL[est], lw=1)
            ax.set_xticks(range(6)); ax.set_xticklabels(["0.3", "0.6", "1.0"] * 2, fontsize=6.3)
            ax.set_xlabel("strength:   cardiac → brain (forward)  |  brain → cardiac (reverse)", fontsize=6.5)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{opt_lab[o]}: {tr_lab[tr]}" for o, tr in rows] if j == 0 else [], fontsize=6.3)
            ax.set_title(f"{est}: cg cardiac ⇄ child brain (none: one value)")
            ax.spines[:].set_visible(True); ax.tick_params(length=0)
            if j == 0:
                panel(ax, "B", x=-0.42)
        save(fig, "fig12_cardiac")


def fig13():
    s12 = comparison("S_12_summary.csv")
    s12["label"] = [f"{f:g}/{r:g}" for f, r in zip(s12.g_fwd, s12.g_rev)]
    order = [("brain-brain", 0.3, 0.0), ("brain-brain", 0.3, 0.1), ("brain-brain", 0.3, 0.2), ("brain-brain", 0.3, 0.3),
             ("brain-brain", 0.2, 0.1), ("brain-brain", 0.2, 0.2), ("heart-heart", 0.3, 0.0), ("heart-heart", 0.3, 0.15), ("heart-heart", 0.3, 0.3)]
    s12 = s12.set_index(["pair", "g_fwd", "g_rev"]).loc[order].reset_index()
    x = np.arange(len(s12))
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9), gridspec_kw={"wspace": 0.3})
        fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.3)
        ax = axes[0]; ax.axhline(ALPHA, color="0.3", ls="--", lw=0.6)
        for k, est in enumerate(EST):
            for col_name, mfc, lab in [("forward_rate", None, "forward"), ("reverse_rate", "white", "reverse")]:
                ax.errorbar(x + dodge(2 * k + (col_name == "reverse_rate"), 4, 0.7), s12[f"{col_name} {est}"], yerr=wilson_err(s12[f"{col_name} {est}"], 200),
                            fmt="o", ms=3, lw=0.6, color=EST_COL[est], mfc=mfc or EST_COL[est], label=f"{est}, {lab} edge")
        ax.set_title("edges detected"); ax.set_ylabel("rate"); ax.legend(fontsize=5.5, loc="center left", labelspacing=0.2)
        for ax, (col_name, title) in zip(axes[1:], [("forward_dominant", "forward direction dominant"),
                                                    ("asymmetry_rate", "asymmetry test significant")]):
            if col_name == "asymmetry_rate":
                fpr_guides(ax)
            else:
                ax.axhline(0.5, color="0.6", lw=0.5, ls=":")
            for k, est in enumerate(EST):
                ax.errorbar(x + dodge(k, 2, 0.4), s12[f"{col_name} {est}"], yerr=wilson_err(s12[f"{col_name} {est}"], 200), fmt="o", ms=3, lw=0.6,
                            color=EST_COL[est], label=est)
            ax.set_title(title)
        axes[2].legend(fontsize=6, loc="center right")
        for ax, letter in zip(axes, "ABC"):
            ax.set_xticks(x); ax.set_xticklabels(s12.label, fontsize=6, rotation=60); ax.set_ylim(-0.02, 1.05)
            ax.axvline(5.5, color="0.75", lw=0.5); panel(ax, letter, x=-0.2)
            ax.text(2.5, -0.36, "brain ⇄ brain", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
            ax.text(7, -0.36, "heart ⇄ heart", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
        save(fig, "fig13_mutual_coupling")


S13_GROUP = [("perm_film_maxT", "max-T within film"), ("perm_24_maxT", "max-T over 24"), ("perm_mean_maxT", "max-T on film mean")]
S13_FILM = [("perm_contrast_maxT", "sign-flip max-T")]
S13_INTER = [("perm_interaction_maxT", "interaction max-T")]
S13_BAYES = [("bayes_edge", "per-edge Bayes"), ("bayes_pooled_eff", "pooled Bayes"), ("bayes_pooled_int", "(1|edge)")]
S13_MARK = {"perm_film_maxT": ("o", "0.15"), "perm_24_maxT": ("s", "0.35"), "perm_mean_maxT": ("^", "0.55"),
            "perm_contrast_maxT": ("D", "0.15"), "perm_interaction_maxT": ("v", "0.15"),
            "bayes_edge": ("o", "#c0392b"), "bayes_pooled_eff": ("s", "#e67e22"), "bayes_pooled_int": ("^", "#f1c40f")}


def fig14():
    det = {est: pd.read_csv(RES / est / "S_13" / "summary_detection.csv") for est in EST}
    fwn = comparison("S_13_summary_familywise_null.csv")
    shr = comparison("S_13_summary_shrinkage.csv")
    cats = [("c", "group", "c: group"), ("d", "group", "d: group\n(2 edges)"), ("e", "group", "e: group\n(film 1)"),
            ("e", "film", "e: film"), ("e", "interaction", "e: interaction"), ("f", "film", "f: film")]
    methods = {"group": S13_GROUP + S13_BAYES, "film": S13_FILM + S13_BAYES, "interaction": S13_INTER + S13_BAYES}
    with plt.rc_context(PUB):
        fig = plt.figure(figsize=(FULL_W, 6.6))
        gs = fig.add_gridspec(2, 1, hspace=0.55, left=0.08, right=0.98, top=0.95, bottom=0.14)
        top = gs[0].subgridspec(1, 2, wspace=0.12)
        for j, est in enumerate(EST):
            ax = fig.add_subplot(top[j]); d = det[est]
            planted = d[d.planted & d.dv.isin(["delta", "delta_std"])]
            for c, (sc, effect, _) in enumerate(cats):
                ms_ = methods[effect]
                for k, (m, lab) in enumerate(ms_):
                    sub = planted[(planted.scenario == sc) & (planted.effect == effect) & (planted.method == m)]
                    if sub.empty:
                        continue
                    rate = sub.rate.mean()
                    mk, col = S13_MARK[m]
                    ax.errorbar(c + dodge(k, len(ms_), 0.75), rate, yerr=wilson_err([rate], 100 * len(sub)), fmt=mk, ms=3, lw=0.6, color=col,
                                mfc=col if m.startswith("perm") else "white", mew=0.8)
            ax.set_xticks(range(len(cats))); ax.set_xticklabels([c[2] for c in cats], fontsize=6.2); ax.set_ylim(-0.02, 1.05)
            ax.set_title(f"{est}: planted effects detected")
            if j == 0:
                ax.set_ylabel("detection rate"); panel(ax, "A", x=-0.12)
            else:
                ax.set_yticklabels([])
        handles = [matplotlib.lines.Line2D([], [], ls="", marker=S13_MARK[m][0], color=S13_MARK[m][1], ms=3.5,
                                           mfc=S13_MARK[m][1] if m.startswith("perm") else "white", label=lab)
                   for m, lab in S13_GROUP + S13_FILM + S13_INTER + S13_BAYES]
        fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=5.6, bbox_to_anchor=(0.53, 0.555), columnspacing=0.8, handletextpad=0.2)
        bottom = gs[1].subgridspec(1, 3, wspace=0.4, width_ratios=[1.3, 1, 1])
        # B: family-wise rate on null edges, scenario a
        ax = fig.add_subplot(bottom[0]); fpr_guides(ax)
        raw = {est: pd.concat([pd.read_csv(f) for f in sorted((RES / est / "S_13").glob("raw_results_full_reps_*.csv"))]) for est in EST}
        rows = [("perm_film_maxT", "group", "group, within film"), ("perm_24_maxT", "group", "group, over 24"),
                ("perm_mean_maxT", "group", "group, film mean"), ("perm_contrast_maxT", "film", "film, sign-flip"),
                ("perm_interaction_maxT", "interaction", "interaction"), ("bayes_edge", "group", "per-edge Bayes"),
                ("bayes_pooled_eff", "group", "pooled Bayes"), ("bayes_pooled_int", "group", "(1|edge)")]
        for k, est in enumerate(EST):
            vals = []
            for m, effect, _ in rows:
                if m == "perm_film_maxT":                  # family-wise rate per film (mean over the three films)
                    r = raw[est]; r = r[(r.scenario == "a") & (r.method == m) & (r.dv == "delta")]
                    vals.append(r.groupby(["rep", "film"]).detected.any().mean())
                else:
                    sel = fwn[(fwn.scenario == "a") & (fwn.method == m) & (fwn.effect == effect) & fwn.dv.isin(["delta", "delta_std"])
                              & (fwn.scope == "all null edges")]
                    vals.append(sel[f"rate {est}"].iloc[0])
            vals = np.array(vals)
            n = np.array([300 if m == "perm_film_maxT" else 100 for m, _, _ in rows])
            ax.errorbar(np.arange(len(rows)) + dodge(k, 2, 0.4), vals, yerr=wilson_err(vals, n), fmt="o", ms=3, lw=0.6, color=EST_COL[est], label=est)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[2] for r in rows], rotation=50, ha="right", fontsize=5.8)
        ax.set_ylim(0, 0.5); ax.set_ylabel("any null edge flagged"); ax.set_title("no effect (scenario a)"); ax.legend(fontsize=6, loc="upper left")
        panel(ax, "B", x=-0.25)
        # C: estimated vs. true group effect on planted edges
        ax = fig.add_subplot(bottom[1]); ax.plot([-1.7, 0.2], [-1.7, 0.2], color="0.6", lw=0.5)
        planted_edges = {"c": ["cg:ROI->child:ROI"], "d": ["cg:ROI->child:ROI", "cg:HRV->child:HRV"], "e": ["cg:ROI->child:ROI"]}
        for m, lab in [("sample_mean_diff", "sample difference")] + S13_BAYES:
            mk, col = S13_MARK.get(m, ("x", "0.3"))
            for est in EST:
                sub = shr[(shr.method == m) & np.array([e in planted_edges.get(s, []) for s, e in zip(shr.scenario, shr.edge)])]
                ax.plot(sub[f"reference {est}"], sub[f"mean_estimate {est}"], ls="", marker=mk, ms=3.5, color=col,
                        mfc="white" if est == "GPDC" else col, mew=0.8, label=f"{lab}" if est == "dDTF" else None)
        ax.set_xlabel("true group effect (within-edge SD)"); ax.set_ylabel("mean estimate"); ax.set_title("estimate vs. truth, planted edges")
        ax.set_xlim(-1.7, 0.1); ax.set_ylim(-1.7, 0.1); ax.set_aspect("equal")
        ax.legend(fontsize=5, loc="lower right", labelspacing=0.2, title="filled dDTF, open GPDC", title_fontsize=5); panel(ax, "C", x=-0.38)
        # D: leakage into null edges in scenario c
        ax = fig.add_subplot(bottom[2]); fpr_guides(ax)
        leak = [("perm_film_maxT", "within\nfilm*"), ("perm_mean_maxT", "film\nmean"),
                ("bayes_edge", "per-edge\nBayes"), ("bayes_pooled_eff", "pooled\nBayes")]
        for k, est in enumerate(EST):
            for scope, mfc in [("all null edges", None), ("null edges except reverse of planted", "white")]:
                vals = np.array([fwn[(fwn.scenario == "c") & (fwn.method == m) & (fwn.effect == "group") & (fwn.scope == scope)
                                     & fwn.dv.isin(["delta", "delta_std"])][f"rate {est}"].iloc[0] for m, _ in leak])
                ax.errorbar(np.arange(len(leak)) + dodge(2 * k + (mfc is not None), 4, 0.7), vals, yerr=wilson_err(vals, 100), fmt="o", ms=3, lw=0.6,
                            color=EST_COL[est], mfc=mfc or EST_COL[est],
                            label=f"{est}, {'all null edges' if mfc is None else 'without reverse edge'}")
        ax.set_xticks(range(len(leak))); ax.set_xticklabels([l[1] for l in leak], fontsize=5.8); ax.set_ylim(-0.02, 1.5)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("group effect on any null edge"); ax.set_title("leakage (scenario c)")
        ax.legend(fontsize=5, loc="upper left", labelspacing=0.15); panel(ax, "D", x=-0.3)
        save(fig, "fig14_group_film")


S2_BANDWIDTH_HZ = 1.5                                 # BANDWIDTH_HZ of S02_envelope_vs_phase.py
S2_PATHS = [("envelope", EEG_COL, "o", "envelope"), ("phase", "#b8460b", "s", "raw signal, widening grid"),
            ("phase_fixedgrid", "#56b4e9", "^", "raw signal, fixed grid")]


def s2_stats(est):
    """Mean and SEM over reps of relative_fwd and direction_index per (truth, path, gap) of S2 for one estimator."""
    raw = pd.read_csv(RES / est / "S_02" / "raw_results.csv")
    return raw.groupby(["truth", "path", "gap"])[["relative_fwd", "direction_index"]].agg(["mean", "sem"])


def figS4():
    stats = {est: s2_stats(est) for est in EST}
    with plt.rc_context(PUB):
        fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.2), sharex="col", sharey="row",
                                 gridspec_kw={"wspace": 0.1, "hspace": 0.18})
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.1)
        for j, (truth, title) in enumerate([("amplitude", "amplitude coupling"), ("phase", "phase coupling"), ("none", "no coupling")]):
            axes[0, j].set_title(title)
            for i, column in enumerate(["relative_fwd", "direction_index"]):
                ax = axes[i, j]
                ax.axhline(1 if i == 0 else 0, color="0.6", lw=0.5, zorder=0)
                ax.axvline(S2_BANDWIDTH_HZ, color="0.5", ls=":", lw=0.8, zorder=0)
                for path, col, mk, _ in S2_PATHS:
                    for est, ls in zip(EST, ("-", "--")):
                        s = stats[est].loc[(truth, path)][column]
                        gap, m, e = s.index.to_numpy(), s["mean"].to_numpy(), s["sem"].to_numpy()
                        ax.fill_between(gap, m - e, m + e, color=col, alpha=0.18, lw=0)
                        ax.plot(gap, m, color=col, ls=ls, lw=0.9, marker=mk, ms=2.5, mew=0.6,
                                mfc=col if est == "dDTF" else "white")
            axes[1, j].set_xlabel("frequency gap between partners (Hz)"); axes[1, j].set_xticks(range(0, 11, 2))
        axes[0, 0].set_yscale("log"); axes[0, 0].set_ylim(0.02, 100)              # dDTF envelope reaches ~70 under no coupling
        axes[0, 0].set_yticks([0.03, 0.1, 0.3, 1, 3, 10, 30]); axes[0, 0].set_yticklabels(["0.03", "0.1", "0.3", "1", "3", "10", "30"])
        axes[0, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[1, 0].set_ylim(-1.05, 1.05); axes[1, 0].set_yticks([-1, -0.5, 0, 0.5, 1])
        axes[0, 0].set_ylabel("edge value relative to gap 0"); axes[1, 0].set_ylabel("direction index")
        handles = [matplotlib.lines.Line2D([], [], color=col, marker=mk, ms=3, lw=1, label=lab) for _, col, mk, lab in S2_PATHS]
        handles += [matplotlib.lines.Line2D([], [], color="0.3", ls=ls, lw=0.9, marker="o", ms=3, mfc=mfc, label=est)
                    for est, ls, mfc in zip(EST, ("-", "--"), ("0.3", "white"))]
        fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.535, 1.0), columnspacing=1.6, handlelength=2.6)
        panel(axes[0, 0], "A", x=-0.2, y=1.0); panel(axes[1, 0], "B", x=-0.2, y=1.0)
        save(fig, "figS4_frequency_gap")
    for est in EST:
        st = stats[est]
        print(f"{est}: amplitude/envelope relative_fwd {st.loc[('amplitude', 'envelope'), ('relative_fwd', 'mean')].agg(['min', 'max']).round(3).tolist()}")
        ph = st.loc[("amplitude", "phase"), ("relative_fwd", "mean")]
        print(f"{est}: amplitude/phase relative_fwd at 4.5/5/9.5 Hz {ph.loc[[4.5, 5.0, 9.5]].round(3).tolist()}, first gap < 0.05: {ph[ph < 0.05].index.min()}")
        fg = st.loc[("amplitude", "phase_fixedgrid"), ("relative_fwd", "mean")]
        print(f"{est}: amplitude/fixed grid relative_fwd max {fg.max():.3f} at {fg.idxmax()} Hz, at 9.5 Hz {fg.loc[9.5]:.3f}, min after max {fg.loc[fg.idxmax():].min():.3f}")
        for path in ("phase", "envelope"):
            di = st.loc[("none", path), ("direction_index", "mean")]
            print(f"{est}: none/{path} direction_index min {di.min():.3f}, max {di.max():.3f}")


FIGURES = {"level1": level1, "level2": level2, "fig3": fig3, "fig_gain_conversion": fig_gain_conversion,
           "fig02": fig02, "fig06": fig06, "fig07": fig07, "fig08": fig08, "fig09": fig09,
           "fig10": fig10, "fig11": fig11, "fig12": fig12, "fig13": fig13, "fig14": fig14, "figS4": figS4}

if __name__ == "__main__":
    for name in (list(FIGURES) if not NAMES or NAMES == ["all"] else NAMES):
        FIGURES[name]()
        print("done", name)
