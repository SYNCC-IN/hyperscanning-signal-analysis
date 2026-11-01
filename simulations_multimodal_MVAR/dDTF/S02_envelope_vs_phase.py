"""S02 - Envelope vs. phase estimation when the persons' centre frequencies differ (Table 2, S2).

Question: how does a centre-frequency (CF) gap between child and caregiver affect envelope-based and
phase-based dDTF, under two different ground truths?
  - 'amplitude': the coupling acts on the log-amplitude of the oscillators (the Stage 0 Part B generator).
  - 'phase': directed phase coupling between noisy oscillators (Kuramoto-type), amplitudes independent.
  - 'none': no coupling (reference level of the edge value).
Both estimators are applied to all truths. NOTE: with the amplitude truth the envelope estimator is aligned
with the ground truth by construction, so that arm shows robustness to the CF gap, not superiority.

Estimators (as in Stage 0 Part B, global fit, AIC order):
  envelope path: individual band-pass -> Hilbert envelope -> 2 Hz -> dDTF;
  phase path: dDTF on the narrow-band filtered raw signals, frequency grid that widens with the gap, edge value =
              MEAN over the grid (the Stage 0 protocol);
  phase_fixedgrid path: the same raw-signal dDTF on a FIXED grid, edge value = SUM over the grid (the share of the
              target's total inflow that comes from the source).
Why the third path: ffDTF is normalized by the inflow summed over ALL frequencies of the grid. A grid that widens with
the gap therefore lowers the mean edge value even with zero coupling (see truth 'none'), which confounds the Stage 0
phase-path result. Compare the 'none' rows of the two phase paths.
Edge of interest: node 1 (caregiver proxy) -> node 0 (child proxy); the reverse edge gives the direction index.

Usage: python S02_envelope_vs_phase.py [quick|full]
Results: Results/S_02/
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.envelopes import downsample, filter_individual_band, hilbert_envelope
from src.sim_estimation import edge_matrix
from src.sim_generators import simulate_phase_oscillators
from src.sim_harness import results_dir, run_parallel
from src.synthetic_mvar import edges_to_coupling, generate_coupled_oscillators

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
N_JOBS = 4
SIM_ID = 2
BASE_SEED = 20260102
SETTINGS = {
    "quick": {"n_rep": 6, "gaps": [0.0, 2.0, 4.0, 8.0], "duration_s": 60.0},
    "full": {"n_rep": 100, "gaps": list(np.arange(0.0, 10.0, 0.5)), "duration_s": 180.0},
}[MODE]
import os
if os.environ.get("SIM_NREP"):            # optional override (timing / diagnostics)
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
TRUTHS = ["none", "amplitude", "phase"]

CHILD_CF_HZ = 8.0
BANDWIDTH_HZ = 1.5                 # half-width of the individual band
CARRIER_ORDER = 4
FS_CARRIER = 128.0
ENV_FS = 2.0
OBS_SNR = 4.0
SELF_GAIN, EDGE_GAIN = 0.3, 0.5    # amplitude truth (log-amplitude VAR), as in Stage 0
PHASE_K_HZ = 1.0                   # phase truth: coupling strength K = 2*pi*PHASE_K_HZ (locks if |gap| < PHASE_K_HZ)
PHASE_NOISE_SD = 1.0               # rad / sqrt(s)
DRIFT_SD_HZ, DRIFT_TAU_S = 0.2, 10.0
AMP_SD, AMP_TAU_S = 0.2, 10.0      # independent slow amplitude fluctuations in the phase truth
MAX_ORDER_ENV, MAX_ORDER_RAW = 10, 15
FREQS_ENV = np.arange(0.01, ENV_FS / 2, 0.01)
FREQ_STEP_RAW = 0.1
FIXED_GRID = np.arange(1.0, 20.0, 0.1)    # phase_fixedgrid path: same grid for every gap
ENVELOPE_PASS_FRACTION = 0.5       # amplitude truth: envelope estimate stays >= 50% of its zero-gap value [proposed]


def generate(truth, cg_cf, n_samples, seed):
    """Two oscillators (child proxy, caregiver proxy) under the chosen ground truth."""
    if truth in ("none", "amplitude"):
        edges = [(0, 0, 1, SELF_GAIN), (1, 1, 1, SELF_GAIN)] + ([(1, 0, 1, EDGE_GAIN)] if truth == "amplitude" else [])
        return generate_coupled_oscillators([CHILD_CF_HZ, cg_cf], edges_to_coupling(edges, 2), OBS_SNR, FS_CARRIER,
                                            n_samples, BANDWIDTH_HZ, envelope_fs=ENV_FS, carrier_order=CARRIER_ORDER, seed=seed)
    rng = np.random.default_rng(seed)
    k_matrix = np.zeros((2, 2))
    k_matrix[0, 1] = 2 * np.pi * PHASE_K_HZ        # caregiver (1) drives child (0)
    signal, _, _ = simulate_phase_oscillators(rng, [CHILD_CF_HZ, cg_cf], k_matrix, FS_CARRIER, n_samples, PHASE_NOISE_SD,
                                              DRIFT_SD_HZ, DRIFT_TAU_S, AMP_SD, AMP_TAU_S)
    return signal + rng.normal(size=signal.shape) * signal.std(axis=1, keepdims=True) / OBS_SNR


def edges_from(design, freqs, fs, p_max, p=None, total=False):
    """dDTF of the 1->0 and 0->1 edges over `freqs` (global fit, AIC order unless `p` is given, no Box-Cox).

    Returns the mean over the grid, or the sum if `total`, and the order used.
    """
    matrix, _, p_used = edge_matrix(design, p, None, None, freqs, fs, "dDTF", (freqs[0], freqs[-1]), p_max=p_max, crit_type="AIC")
    scale = len(freqs) if total else 1.0
    return matrix[0, 1] * scale, matrix[1, 0] * scale, p_used


def run_task(task):
    """All gaps for one (truth, realization): returns one record per gap and estimator path."""
    truth, rep, seed = task["truth"], task["rep"], task["seed"]
    n_samples = int(SETTINGS["duration_s"] * FS_CARRIER)
    records = []
    for gap in SETTINGS["gaps"]:
        cg_cf = CHILD_CF_HZ + gap
        signals = generate(truth, cg_cf, n_samples, seed)
        env = []
        for node, cf in enumerate((CHILD_CF_HZ, cg_cf)):
            envelope = hilbert_envelope(filter_individual_band(signals[node], FS_CARRIER, cf, BANDWIDTH_HZ, CARRIER_ORDER))
            env.append(downsample(envelope, FS_CARRIER, ENV_FS)[0])
        n_env = min(len(env[0]), len(env[1]))
        env = np.stack([e[:n_env] for e in env])
        env = (env - env.mean(1, keepdims=True)) / env.std(1, keepdims=True)
        fwd_env, rev_env, _ = edges_from(env, FREQS_ENV, ENV_FS, MAX_ORDER_ENV)
        raw = np.stack([filter_individual_band(signals[n], FS_CARRIER, cf, BANDWIDTH_HZ, CARRIER_ORDER)
                        for n, cf in enumerate((CHILD_CF_HZ, cg_cf))])
        raw = (raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True)
        freqs_raw = np.arange(min(CHILD_CF_HZ, cg_cf) - BANDWIDTH_HZ - 1.0, max(CHILD_CF_HZ, cg_cf) + BANDWIDTH_HZ + 1.0, FREQ_STEP_RAW)
        fwd_raw, rev_raw, p_raw = edges_from(raw, freqs_raw, FS_CARRIER, MAX_ORDER_RAW)
        fwd_fix, rev_fix, _ = edges_from(raw, FIXED_GRID, FS_CARRIER, MAX_ORDER_RAW, p=p_raw, total=True)
        records.append({"truth": truth, "gap": gap, "rep": rep, "path": "envelope", "fwd": fwd_env, "rev": rev_env})
        records.append({"truth": truth, "gap": gap, "rep": rep, "path": "phase", "fwd": fwd_raw, "rev": rev_raw})
        records.append({"truth": truth, "gap": gap, "rep": rep, "path": "phase_fixedgrid", "fwd": fwd_fix, "rev": rev_fix})
    return records


def mean_sem(x):
    """Mean and standard error of a 1-D sample."""
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def main():
    """Run S02 and write all outputs to Results/S_02."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID)
    tasks = [{"truth": truth, "rep": rep, "seed": BASE_SEED + rep} for truth in TRUTHS for rep in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df["direction_index"] = (df.fwd - df.rev) / (df.fwd + df.rev)
    base = df[df.gap == df.gap.min()].set_index(["truth", "path", "rep"]).fwd
    df["relative_fwd"] = df.fwd.values / base.loc[list(zip(df.truth, df.path, df.rep))].values
    df.to_csv(out_dir / "raw_results.csv", index=False)
    summary = df.groupby(["truth", "path", "gap"]).agg(fwd=("fwd", "mean"), rev=("rev", "mean"), direction_index=("direction_index", "mean"),
                                                         relative_fwd=("relative_fwd", "mean"), n_rep=("rep", "size")).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    PATHS = ["envelope", "phase", "phase_fixedgrid"]
    colors = {"envelope": "C0", "phase": "C1", "phase_fixedgrid": "C2"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, truth in zip(axes, TRUTHS):
        for path in PATHS:
            sub = df[(df.truth == truth) & (df.path == path)]
            for column, style in (("fwd", "-"), ("rev", ":")):
                stats = np.array([mean_sem(g[column].values) for _, g in sub.groupby("gap")])
                gaps = sorted(sub.gap.unique())
                ax.plot(gaps, stats[:, 0], style, color=colors[path], marker="o", ms=3, label=f"{path} {'1->0' if column == 'fwd' else '0->1'}")
                ax.fill_between(gaps, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1], color=colors[path], alpha=0.2)
        ax.set_title(f"ground truth: {truth}"); ax.set_xlabel("centre-frequency gap (Hz)"); ax.axvline(BANDWIDTH_HZ, color="gray", ls=":", lw=1)
    axes[0].set_ylabel("mean band-averaged dDTF"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_edge_values_by_truth.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for path in PATHS:
        sub = df[(df.truth == "amplitude") & (df.path == path)]
        stats = np.array([mean_sem(g.relative_fwd.values) for _, g in sub.groupby("gap")]); gaps = sorted(sub.gap.unique())
        ax.plot(gaps, stats[:, 0], marker="o", color=colors[path], label=f"{path}-based dDTF")
        ax.fill_between(gaps, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1], color=colors[path], alpha=0.25)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8); ax.axhline(ENVELOPE_PASS_FRACTION, color="r", ls=":", lw=0.8)
    ax.set_xlabel("centre-frequency gap (Hz)"); ax.set_ylabel("dDTF 1->0, relative to own zero-gap value")
    ax.set_title("Amplitude ground truth (estimator matched by construction)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_relative_amplitude_truth.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, truth in zip(axes, TRUTHS):
        for path in PATHS:
            sub = summary[(summary.truth == truth) & (summary.path == path)]
            ax.plot(sub.gap, sub.direction_index, marker="o", color=colors[path], label=path)
        ax.axhline(0, color="gray", lw=0.8); ax.set_title(f"direction index, truth: {truth}"); ax.set_xlabel("gap (Hz)")
    axes[0].set_ylabel("(1->0 - 0->1) / (sum)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_direction_index.png", dpi=150); plt.close(fig)

    amp_env = summary[(summary.truth == "amplitude") & (summary.path == "envelope")]
    amp_phase = summary[(summary.truth == "amplitude") & (summary.path == "phase")]
    amp_fixed = summary[(summary.truth == "amplitude") & (summary.path == "phase_fixedgrid")]
    criteria = {
        f"amplitude truth: envelope dDTF stays >= {ENVELOPE_PASS_FRACTION:.0%} of zero-gap value across the sweep": bool(amp_env.relative_fwd.min() >= ENVELOPE_PASS_FRACTION),
        "amplitude truth (informative): phase-path dDTF drops >= 50% somewhere": bool(1 - amp_phase.relative_fwd.min() >= 0.5),
        "amplitude truth (informative): fixed-grid phase-path dDTF drops >= 50% somewhere": bool(1 - amp_fixed.relative_fwd.min() >= 0.5),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": {k: (list(map(float, v)) if k == "gaps" else v) for k, v in SETTINGS.items()},
                   "criteria": criteria}, f, indent=2)
    print(summary.round(4).to_string()); print(criteria)


if __name__ == "__main__":
    main()
