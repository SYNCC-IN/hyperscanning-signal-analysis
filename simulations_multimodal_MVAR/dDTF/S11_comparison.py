"""S11 - Comparison with simpler alternatives (Table 2, S11).

Competitors (all evaluated inside the same surrogate-null framework, statistic transformed by x**0.25 like dDTF's Box-Cox):
  (i)   pairwise (bivariate) Granger causality of the envelopes, log(RSS_restricted / RSS_full), order 4;
  (ii)  lagged cross-correlation of envelopes, max over lags 1..5 samples of |r| (source leads);
  (iii) magnitude-squared coherence of envelopes, averaged over 0.15-0.5 Hz (symmetric);
  (iv)  phase-locking value (PLV) of individually band-passed raw signals (symmetric; Part B only).
Reference: 4-node dDTF (the proposed method), or 4-node GPDC when run with the GPDC argument.

Part A (Level 1, envelope level, 4 nodes as in S3/S4/S5):
  'null_stimulus' - no coupling, common stimulus -> false-positive rate of cg->child edges;
  'power'         - planted cg:ROI->child:ROI and cg:HRV->child:HRV -> power and direction accuracy;
  'indirect'      - planted cg:ROI->child:ROI->child:HRV -> detection of the absent indirect edge cg:ROI->child:HRV.
Part B (Level 2, two oscillators with different centre frequencies as in S2; truths none / amplitude / phase;
  frequency gap 0 and 3 Hz): FPR and power of all methods, with envelopes (i-iii and dDTF) or raw signals (iv).

Usage: python S11_comparison.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_11/
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import coherence, hilbert, resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.envelopes import downsample, filter_individual_band, hilbert_envelope
from src.sim_generators import make_stimulus, sample_stimulus_gain, simulate_dyads, simulate_phase_oscillators, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import run_study, wilson_interval
from src.synthetic_mvar import edges_to_coupling, generate_coupled_oscillators

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 11
BASE_SEED = 20260111
SETTINGS = {
    "quick": {"n_rep": 4, "n_dyads": 12, "seg_s": 60, "b_dyads": 10, "b_duration_s": 60.0, "gaps": [0.0, 3.0]},
    "full": {"n_rep": 100, "n_dyads": 25, "seg_s": 60, "b_dyads": 25, "b_duration_s": 120.0, "gaps": [0.0, 3.0]},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

# --- Part A (Level 1)
FS = 2.5
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)                 # [TBD: confirm]
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
STIM_GAIN, STIM_SD_LOG = 1.0, 0.5
A_SCENARIOS = {
    "null_stimulus": {"planted": [], "stimulus": True, "report": {"cg:ROI->child:ROI": ((1, 0), "null_edge"), "cg:HRV->child:HRV": ((3, 2), "null_edge")}},
    "power": {"planted": [(1, 0, 1, 0.4), (3, 2, 1, 0.3)], "stimulus": False,
              "report": {"cg:ROI->child:ROI": ((1, 0), "planted"), "cg:HRV->child:HRV": ((3, 2), "planted")}},
    "indirect": {"planted": [(1, 0, 1, 0.4), (0, 2, 1, 0.4)], "stimulus": False,
                 "report": {"cg:ROI->child:ROI": ((1, 0), "planted"), "cg:ROI->child:HRV": ((1, 2), "absent_indirect")}},
}
# --- Part B (Level 2)
CHILD_CF_HZ, BANDWIDTH_HZ, CARRIER_ORDER, FS_CARRIER = 8.0, 1.5, 4, 128.0
OBS_SNR, SELF_GAIN, EDGE_GAIN = 4.0, 0.3, 0.5
PHASE_K_HZ, PHASE_NOISE_SD, DRIFT_SD_HZ, DRIFT_TAU_S, AMP_SD, AMP_TAU_S = 1.0, 1.0, 0.2, 10.0, 0.2, 10.0
FS_RAW = 32.0
B_EDGE = {"cg->child": (1, 0), "child->cg": (0, 1)}

STABILITY_MAX_ROOT = 1.0
Z_CRIT, ALPHA = 1.96, 0.05
FPR_MAX, POWER_MIN = 0.10, 0.80       # [proposed]


# ---------------------------------------------------------------------------
# Competitor statistics: design (k, n) -> matrix[target, source]
# ---------------------------------------------------------------------------
def _pt(x):
    """Monotone power transform (stands in for Box-Cox 0.25) for non-negative statistics."""
    return np.maximum(x, 0.0) ** 0.25


def gc_matrix(design, p=MODEL_ORDER):
    """Pairwise Granger causality log(RSS_restricted / RSS_full), target <- source."""
    k, n = design.shape
    lags = [np.stack([design[i, p - l - 1: n - l - 1] for l in range(p)], axis=1) for i in range(k)]
    out = np.zeros((k, k))
    for i in range(k):
        y = design[i, p:]
        rss_r = np.sum((y - lags[i] @ np.linalg.lstsq(lags[i], y, rcond=None)[0]) ** 2)
        for j in range(k):
            if j != i:
                x = np.hstack([lags[i], lags[j]])
                out[i, j] = np.log(rss_r / np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    return _pt(out)


def xcorr_matrix(design, max_lag=5):
    """Maximum over lags 1..max_lag of |corr(source(t - lag), target(t))|."""
    k, n = design.shape
    out = np.zeros((k, k))
    for t in range(k):
        for s in range(k):
            if s != t:
                out[t, s] = max(abs(np.corrcoef(design[s, :-l], design[t, l:])[0, 1]) for l in range(1, max_lag + 1))
    return _pt(out)


def coherence_matrix(design, fs=FS, band=BAND_HZ):
    """Band-averaged magnitude-squared coherence (symmetric)."""
    k = design.shape[0]
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            f, c = coherence(design[i], design[j], fs=fs, nperseg=int(WIN_LEN_S * fs), noverlap=int(WIN_LEN_S * fs) // 2)
            out[i, j] = out[j, i] = c[(f >= band[0]) & (f <= band[1])].mean()
    return _pt(out)


def plv_matrix(design):
    """Phase-locking value of raw narrow-band signals (symmetric)."""
    phases = np.angle(hilbert(design, axis=1))
    k = design.shape[0]
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = np.abs(np.mean(np.exp(1j * (phases[i] - phases[j]))))
    return _pt(out)


def callable_est(fn):
    """`edge_matrix` keyword arguments for a callable statistic."""
    return dict(p=0, win_len=None, step=None, freqs=None, fs=None, estimator=fn, band_hz=None)


def methods_level1():
    """Method name -> `edge_matrix` kwargs on envelope-level designs."""
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    return {f"{ESTIMATOR} (4-node)": dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX),
            "pairwise Granger": callable_est(gc_matrix), "lagged cross-correlation": callable_est(xcorr_matrix),
            "envelope coherence": callable_est(coherence_matrix)}


# ---------------------------------------------------------------------------
def run_task_a(task):
    """One replicated Part A study for a scenario: all methods on the same designs."""
    rng = np.random.default_rng(task["seed"])
    sc = A_SCENARIOS[task["scenario"]]
    n_samples = int(SETTINGS["seg_s"] * FS)
    stimulus = stim_gain = None
    if sc["stimulus"]:
        stimulus = make_stimulus(rng, n_samples, FS, 0.7, 0.3)
        stim_gain = sample_stimulus_gain(rng, SETTINGS["n_dyads"], [STIM_GAIN] * 4, STIM_SD_LOG)
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, sc["planted"], N_LAGS,
                                            stimulus=stimulus, stimulus_gain=stim_gain))
    edges = {name: edge for name, (edge, _) in sc["report"].items()}
    out = []
    for method, est in methods_level1().items():
        for r in run_study(designs, NODES, CHILD_NODES, est, edges, ["pooled"], None, rng, STABILITY_MAX_ROOT, Z_CRIT, ALPHA, 0):
            out.append(dict(r, part="A", scenario=task["scenario"], method=method, role=sc["report"][r["edge"]][1], rep=task["rep"]))
    return out


def generate_b(truth, gap, n_samples, seed):
    """Two oscillators (child proxy, caregiver proxy) under the chosen ground truth (as S2)."""
    cg_cf = CHILD_CF_HZ + gap
    if truth in ("none", "amplitude"):
        edges = [(0, 0, 1, SELF_GAIN), (1, 1, 1, SELF_GAIN)] + ([(1, 0, 1, EDGE_GAIN)] if truth == "amplitude" else [])
        return generate_coupled_oscillators([CHILD_CF_HZ, cg_cf], edges_to_coupling(edges, 2), OBS_SNR, FS_CARRIER, n_samples, BANDWIDTH_HZ,
                                            envelope_fs=2.0, carrier_order=CARRIER_ORDER, seed=seed)
    rng = np.random.default_rng(seed)
    k_matrix = np.zeros((2, 2)); k_matrix[0, 1] = 2 * np.pi * PHASE_K_HZ
    signal, _, _ = simulate_phase_oscillators(rng, [CHILD_CF_HZ, cg_cf], k_matrix, FS_CARRIER, n_samples, PHASE_NOISE_SD, DRIFT_SD_HZ, DRIFT_TAU_S, AMP_SD, AMP_TAU_S)
    return signal + rng.normal(size=signal.shape) * signal.std(axis=1, keepdims=True) / OBS_SNR


def run_task_b(task):
    """One replicated Part B study for (truth, gap): envelope methods and raw-signal PLV."""
    rng = np.random.default_rng(task["seed"])
    truth, gap = task["truth"], task["gap"]
    n_samples = int(SETTINGS["b_duration_s"] * FS_CARRIER)
    env_designs, raw_designs = [], []
    for _ in range(SETTINGS["b_dyads"]):
        signals = generate_b(truth, gap, n_samples, int(rng.integers(2 ** 31)))
        band = [filter_individual_band(signals[node], FS_CARRIER, cf, BANDWIDTH_HZ, CARRIER_ORDER) for node, cf in enumerate((CHILD_CF_HZ, CHILD_CF_HZ + gap))]
        env = [downsample(hilbert_envelope(b), FS_CARRIER, FS)[0] for b in band]
        n = min(len(e) for e in env)
        env_designs.append(np.stack([e[:n] for e in env]))
        raw_designs.append(np.stack([resample_poly(b, int(FS_RAW), int(FS_CARRIER)) for b in band]))
    env_designs = zscore_designs(np.stack(env_designs)); raw_designs = zscore_designs(np.stack(raw_designs))
    out = []
    for method, est in methods_level1().items():
        # the 2-node problem: dDTF over both nodes
        for r in run_study(env_designs, [0, 1], [0], est, B_EDGE, ["pooled"], None, rng, STABILITY_MAX_ROOT, Z_CRIT, ALPHA, 0):
            out.append(dict(r, part="B", scenario=f"{truth}, gap {gap:g} Hz", truth=truth, gap=gap, method=method, rep=task["rep"]))
    for r in run_study(raw_designs, [0, 1], [0], callable_est(plv_matrix), B_EDGE, ["pooled"], None, rng, STABILITY_MAX_ROOT, Z_CRIT, ALPHA, 0):
        out.append(dict(r, part="B", scenario=f"{truth}, gap {gap:g} Hz", truth=truth, gap=gap, method="PLV (raw signals)", rep=task["rep"]))
    return out


def main():
    """Run S11 and write all outputs to Results/<estimator>/S_11."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    n_rep = SETTINGS["n_rep"]
    cells_a = list(A_SCENARIOS)
    cells_b = [(t, g) for t in ("none", "amplitude", "phase") for g in SETTINGS["gaps"]]
    seeds = spawn_seeds(BASE_SEED, (len(cells_a) + len(cells_b)) * n_rep)
    tasks_a = [{"scenario": s, "rep": r, "seed": seeds[i * n_rep + r]} for i, s in enumerate(cells_a) for r in range(n_rep)]
    off = len(cells_a) * n_rep
    tasks_b = [{"truth": t, "gap": g, "rep": r, "seed": seeds[off + i * n_rep + r]} for i, (t, g) in enumerate(cells_b) for r in range(n_rep)]
    results_a = run_parallel(run_task_a, tasks_a, N_JOBS)
    results_b = run_parallel(run_task_b, tasks_b, N_JOBS)
    df_a = pd.DataFrame([r for recs in results_a for r in recs]); df_b = pd.DataFrame([r for recs in results_b for r in recs])
    df_a.to_csv(out_dir / "raw_results_partA.csv", index=False); df_b.to_csv(out_dir / "raw_results_partB.csv", index=False)

    def aggregate(df, keys):
        s = df.groupby(keys).agg(study_rate=("study_reject", "mean"), dyad_rate=("dyad_rate", "mean"), direction_accuracy=("dir_study", "mean"),
                                 median_z=("median_z", "median"), n_rep=("rep", "size")).reset_index()
        ci = s.apply(lambda r: wilson_interval(int(round(r.study_rate * r.n_rep)), int(r.n_rep)), axis=1)
        s["ci_low"], s["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
        return s
    sum_a = aggregate(df_a, ["scenario", "role", "method"]); sum_a.to_csv(out_dir / "summary_partA.csv", index=False)
    sum_b = aggregate(df_b[df_b.edge == "cg->child"], ["truth", "gap", "method"]); sum_b.to_csv(out_dir / "summary_partB.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (scenario, role, title) in zip(axes, [("null_stimulus", "null_edge", "false-positive rate (stimulus, no coupling)"),
                                                  ("power", "planted", "power (planted edges)"),
                                                  ("indirect", "absent_indirect", "false positives (absent indirect edge)")]):
        sub = sum_a[(sum_a.scenario == scenario) & (sum_a.role == role)]
        ax.bar(range(len(sub)), sub.study_rate, yerr=[(sub.study_rate - sub.ci_low).clip(lower=0), (sub.ci_high - sub.study_rate).clip(lower=0)], capsize=2)
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.method, rotation=30, ha="right", fontsize=7); ax.set_title(title, fontsize=9)
        ax.axhline(ALPHA, color="k", lw=0.6)
    axes[0].set_ylabel("study-level rejection rate")
    fig.tight_layout(); fig.savefig(out_dir / "fig_partA_methods.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, len(SETTINGS["gaps"]), figsize=(6 * len(SETTINGS["gaps"]), 4), sharey=True, squeeze=False)
    for ax, gap in zip(axes[0], SETTINGS["gaps"]):
        sub = sum_b[sum_b.gap == gap]
        for method, s2 in sub.groupby("method"):
            s2 = s2.set_index("truth").reindex(["none", "amplitude", "phase"])
            ax.plot(s2.index, s2.study_rate, marker="o", label=method)
        ax.axhline(ALPHA, color="k", lw=0.6); ax.set_title(f"centre-frequency gap {gap:g} Hz"); ax.set_xlabel("ground truth")
    axes[0, 0].set_ylabel("detection rate cg -> child"); axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_partB_methods.png", dpi=150); plt.close(fig)

    ref = f"{ESTIMATOR} (4-node)"
    def rate(df, **kw):
        s = df
        for k_, v in kw.items():
            s = s[s[k_] == v]
        return s.study_rate.to_numpy()
    comp_a = sum_a[sum_a.method != ref]
    criteria = {
        f"{ESTIMATOR}: FPR <= {FPR_MAX} under common stimulus": bool((rate(sum_a, scenario="null_stimulus", method=ref) <= FPR_MAX).all()),
        f"{ESTIMATOR}: power >= {POWER_MIN} on planted edges": bool((rate(sum_a, scenario="power", method=ref) >= POWER_MIN).all()),
        f"{ESTIMATOR}: absent indirect edge detected <= {FPR_MAX}": bool((rate(sum_a, scenario="indirect", role="absent_indirect", method=ref) <= FPR_MAX).all()),
        f"{ESTIMATOR} has the lowest absent-indirect-edge rate among methods": bool(
            rate(sum_a, scenario="indirect", role="absent_indirect", method=ref)[0] <= comp_a[(comp_a.scenario == "indirect") & (comp_a.role == "absent_indirect")].study_rate.min()),
        f"Part B: envelope {ESTIMATOR} FPR <= {FPR_MAX} (truth none)": bool((rate(sum_b, truth="none", method=ref) <= FPR_MAX).all()),
        f"Part B: envelope {ESTIMATOR} detects the amplitude truth at gap 3 Hz (>= {POWER_MIN})": bool(rate(sum_b, truth="amplitude", gap=3.0, method=ref)[0] >= POWER_MIN) if 3.0 in SETTINGS["gaps"] else False,
        f"Part B: PLV does not detect the amplitude truth at gap 3 Hz (<= {FPR_MAX})": bool(rate(sum_b, truth="amplitude", gap=3.0, method="PLV (raw signals)")[0] <= FPR_MAX) if 3.0 in SETTINGS["gaps"] else False,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(sum_a.round(3).to_string()); print(sum_b.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
