"""S01 - Recovery and orientation of planted edges (Table 2, S1).

Question: with a known directed structure, does the estimator recover the planted edges, in the
right direction, and is its source/target convention what we assume?

Part A (replicates Stage 0 Part A): long clean recordings, repository `direct_dtf` (GPDC: `gen_partial_directed_coherence`), global fit.
  (a) orientation check: a single edge 0 -> 1 must give a dominant dDTF in one axis (ratio >= 3);
  (b) four-node VAR mimicking the real design, recovered edges compared with their reverse.
Part B (realistic): ~150-sample segments, windowed-ACF fit at the common order, surrogate null, many
  replicated studies. Reports recovery against the reverse edge and against absent edges, and the
  detection rate of every directed edge.
Also checks that the fast estimator used in all simulations equals the repository estimator.

Usage: python S01_recovery_orientation.py [quick|full] [dDTF|GPDC]
Results: Results/<estimator>/S_01/
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.connectivity import Granger_estimator
from src.design import window_geometry
from src.mtmvar import direct_dtf, gen_partial_directed_coherence
from src.sim_estimation import connectivity_cube, fit_mvar
from src.sim_generators import simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import real_matrices, run_study, wilson_interval
from src.synthetic_mvar import edges_to_coupling, generate_var_process

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 1
BASE_SEED = 20260101
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 10},
    "full": {"n_rep": 200, "n_dyads": 25},
}[MODE]
import os
if os.environ.get("SIM_NREP"):            # optional override (timing / diagnostics)
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
CHILD_NODES = [0, 2]

# --- Part A (as in scripts/pipeline_exploratory_mlutimodal_dDTF/stage00_synthetic_validation.py)
REPOSITORY_ESTIMATOR = {"dDTF": direct_dtf, "GPDC": gen_partial_directed_coherence}[ESTIMATOR]   # Part A
PART_A_FS = 2.0
PART_A_N_SAMPLES = 4000
PART_A_SNR = 5.0
PART_A_MAX_ORDER = 10
PART_A_FREQ_STEP = 0.01
SELF_GAIN = 0.3
PART_A_EDGES = [(0, 0, 1, SELF_GAIN), (1, 1, 1, SELF_GAIN), (2, 2, 1, SELF_GAIN), (3, 3, 1, SELF_GAIN),
                (1, 0, 1, 0.5), (3, 2, 2, 0.5), (3, 0, 3, 0.35)]
ORIENTATION_EDGES = [(0, 1, 1, 0.6)]
ORIENTATION_SNR = 10.0
ORIENTATION_N_SAMPLES = 3000
ORIENTATION_MAX_ORDER = 5
ORIENTATION_RATIO = 3.0

# --- Part B (realistic design)
FS = 2.5
N_SAMPLES = 150                      # 60 s
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4                      # common order for real and surrogate fits
BAND_HZ = (0.15, 0.5)                # coupling band [TBD: confirm with the application paper]
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])   # ROI envelopes rough, IBI smooth
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
PLANTED = {"cg:ROI->child:ROI": (1, 0, 1, 0.4), "cg:HRV->child:HRV": (3, 2, 1, 0.3), "cg:HRV->child:ROI": (3, 0, 2, 0.3)}
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
CRITERION_MIN_RATE = 0.95            # planted edge must beat its reverse and the absent edges in >= 95% of studies [proposed]


def all_edges():
    """Every off-diagonal directed edge as ``name -> (source, target)``."""
    return {f"{NODE_NAMES[s]}->{NODE_NAMES[t]}": (s, t) for s in range(4) for t in range(4) if s != t}


def implementation_check():
    """Compare the fast estimator with the repository estimator on one simulated dyad."""
    rng = np.random.default_rng(BASE_SEED)
    design = zscore_designs(simulate_dyads(rng, 1, N_SAMPLES, FS, POLE_RADIUS_RANGES, F0_RANGES,
                                           list(PLANTED.values()), N_LAGS))[0]
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    coeffs, variance, _ = fit_mvar(design, MODEL_ORDER, win_len, step)
    errors = {}
    for estimator in ("dDTF", "ffDTF", "GPDC"):
        with contextlib.redirect_stdout(io.StringIO()):
            reference, _ = Granger_estimator(design, FREQS, FS, MODEL_ORDER, win_len, step, "linear", ESTIMATOR=estimator)
        fast = connectivity_cube(coeffs, variance, FREQS, FS, estimator)
        errors[estimator] = float(np.max(np.abs(fast - reference) / (np.abs(reference) + 1e-12)))
    assert max(errors.values()) < 1e-6, f"fast estimator deviates from the repository estimator: {errors}"
    return errors


def part_a():
    """Stage-0 Part A: orientation check and 4-node recovery with the repository estimator."""
    out = {}
    signals = generate_var_process(edges_to_coupling(ORIENTATION_EDGES, 2), ORIENTATION_SNR, ORIENTATION_N_SAMPLES, seed=BASE_SEED)
    freqs = np.arange(PART_A_FREQ_STEP, PART_A_FS / 2, PART_A_FREQ_STEP)
    with contextlib.redirect_stdout(io.StringIO()):
        cube = REPOSITORY_ESTIMATOR(signals, freqs, PART_A_FS, max_model_order=ORIENTATION_MAX_ORDER, crit_type="AIC")
    mean_01, mean_10 = cube[0, 1].mean(), cube[1, 0].mean()
    out["orientation"] = {f"{ESTIMATOR}[0,1]": float(mean_01), f"{ESTIMATOR}[1,0]": float(mean_10),
                          "target_source_confirmed": bool(mean_10 > ORIENTATION_RATIO * mean_01),
                          "ratio": float(mean_10 / mean_01)}
    coupling = edges_to_coupling(PART_A_EDGES, 4)
    signals = generate_var_process(coupling, PART_A_SNR, PART_A_N_SAMPLES, seed=BASE_SEED)
    freqs = np.arange(0, PART_A_FS / 2, PART_A_FREQ_STEP)
    with contextlib.redirect_stdout(io.StringIO()):
        cube = REPOSITORY_ESTIMATOR(signals, freqs, PART_A_FS, max_model_order=PART_A_MAX_ORDER, crit_type="AIC")
    mean = cube.mean(axis=2)
    out["recovery"] = {name: {"forward": float(mean[t, s]), "reverse": float(mean[s, t]), "forward_larger": bool(mean[t, s] > mean[s, t])}
                       for name, (s, t) in {"cg:ROI->child:ROI": (1, 0), "cg:HRV->child:HRV": (3, 2), "cg:HRV->child:ROI": (3, 0)}.items()}
    out["matrix"] = mean
    out["injected"] = np.abs(coupling).sum(axis=2)
    return out


def run_replication(seed):
    """One replicated study: simulate dyads with planted edges, analyse, return per-edge records and the raw mean matrix."""
    rng = np.random.default_rng(seed)
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], N_SAMPLES, FS, POLE_RADIUS_RANGES, F0_RANGES, list(PLANTED.values()), N_LAGS))
    records = run_study(designs, [0, 1, 2, 3], CHILD_NODES, est, all_edges(), ["pooled"], None, rng,
                        STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM)
    raw_est = dict(est, box_cox_lambda=-1)
    raw, roots, _ = real_matrices(designs, [0, 1, 2, 3], raw_est)
    return records, raw[roots < STABILITY_MAX_ROOT].mean(axis=0)


def heatmap(ax, matrix, title):
    """Draw a (target, source) matrix with node labels."""
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(4)); ax.set_xticklabels(NODE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(NODE_NAMES)
    ax.set_xlabel("source"); ax.set_ylabel("target"); ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    """Run S01 and write all outputs to Results/<estimator>/S_01."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    errors = implementation_check()
    a = part_a()
    results = run_parallel(run_replication, spawn_seeds(BASE_SEED, SETTINGS["n_rep"]), N_JOBS)
    records = pd.DataFrame([dict(r, rep=i) for i, (recs, _) in enumerate(results) for r in recs])
    raw_mean = np.mean([m for _, m in results], axis=0)
    records.to_csv(out_dir / "partB_records.csv", index=False)

    n_rep = SETTINGS["n_rep"]
    rows = []
    for name, (source, target, _, gain) in PLANTED.items():
        fwd = records[records.edge == name].set_index("rep")
        rev_name = f"{NODE_NAMES[target]}->{NODE_NAMES[source]}"
        reverse = records[records.edge == rev_name].set_index("rep")
        absent = [n for n, (s, t) in all_edges().items() if all((s, t) != (p[0], p[1]) for p in PLANTED.values())]
        beats_rev = (fwd.median_delta > reverse.median_delta).sum()
        beats_absent = np.mean([(fwd.median_delta > records[records.edge == n].set_index("rep").median_delta).mean() for n in absent])
        rows.append({"edge": name, "gain": gain, "detect_rate": fwd.study_reject.mean(),
                     "detect_ci_low": wilson_interval(int(fwd.study_reject.sum()), n_rep)[0],
                     "detect_ci_high": wilson_interval(int(fwd.study_reject.sum()), n_rep)[1],
                     "beats_reverse_rate": beats_rev / n_rep, "beats_absent_rate": beats_absent,
                     "reverse_detect_rate": reverse.study_reject.mean(),
                     "median_z": fwd.median_z.median(), "dir_study_rate": fwd.dir_study.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "partB_summary.csv", index=False)
    detect = records.groupby("edge").study_reject.mean()
    detect.to_csv(out_dir / "partB_detection_rate_by_edge.csv")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    inj = a["injected"].copy(); np.fill_diagonal(inj, 0); rec_mat = a["matrix"].copy(); np.fill_diagonal(rec_mat, 0)
    heatmap(axes[0], inj, "Injected coupling (|gain| summed over lags)"); heatmap(axes[1], rec_mat, f"Recovered mean {ESTIMATOR} (Part A)")
    fig.tight_layout(); fig.savefig(out_dir / "fig_partA_injected_vs_recovered.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    inj_b = np.zeros((4, 4))
    for s, t, _, g in PLANTED.values():
        inj_b[t, s] = g
    rate = np.zeros((4, 4))
    for name, (s, t) in all_edges().items():
        rate[t, s] = detect[name]
    raw_plot = raw_mean.copy(); np.fill_diagonal(raw_plot, 0)
    heatmap(axes[0], raw_plot, f"Mean raw {ESTIMATOR}, {n_rep} studies (Part B)"); heatmap(axes[1], rate, "Detection rate (pairing permutation, alpha=.05)")
    for (t, s), v in np.ndenumerate(rate):
        axes[1].text(s, t, f"{v:.2f}" if t != s else "", ha="center", va="center", color="w", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "fig_partB_recovery.png", dpi=150); plt.close(fig)

    criteria = {
        "orientation confirmed (dominant axis >= 3x)": a["orientation"]["target_source_confirmed"],
        "Part A planted edges larger than reverse": all(v["forward_larger"] for v in a["recovery"].values()),
        f"Part B planted edges beat reverse in >= {CRITERION_MIN_RATE:.0%} of studies": bool((summary.beats_reverse_rate >= CRITERION_MIN_RATE).all()),
        f"Part B planted edges beat absent edges in >= {CRITERION_MIN_RATE:.0%} of studies": bool((summary.beats_absent_rate >= CRITERION_MIN_RATE).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "implementation_check_max_rel_error": errors,
                   "partA_orientation": a["orientation"], "partA_recovery": a["recovery"],
                   "partB": summary.round(4).to_dict(orient="records"), "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
