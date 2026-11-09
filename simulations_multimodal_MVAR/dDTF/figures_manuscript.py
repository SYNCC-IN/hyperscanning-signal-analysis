"""Manuscript figures not produced by the S-scripts.

fig_level1_signals.png  : one simulated Level-1 dyad (S1 setting): time courses (60 s) and model spectra.
fig_level2_signals.png  : Level-2 signals: EEG-like oscillations and their PSD, envelopes and their PSD,
                          cardiac (RSA-type) series and its PSD.
fig3_S1_recovery.png    : S1 Part B, planted structure and detection rates of dDTF and GPDC side by side.

Usage: python figures_manuscript.py OUT_DIR
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
from src.synthetic_mvar import edges_to_coupling, generate_coupled_oscillators, resonant_self_edges, self_ar2_coeffs

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RES = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "simulations_multimodal_MVAR/dDTF/Results"
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


if __name__ == "__main__":
    level1(); level2(); fig3(); print("ok")
