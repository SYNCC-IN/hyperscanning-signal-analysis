"""S06b - Unequal signal-to-noise ratio between nodes (variant of S6; and of S1 with coupling).

White observation noise is added after the VAR simulation (not part of the VAR; `src.sim_generators.add_node_noise`),
with an SD of 0 / 0.25 / 0.5 / 1 / 2 times the node SD, before z-scoring.
No coupling: the S6 design (smooth caregiver heart, pole radius 0.9; rough child brain), N = 25, 60 s, the S6 edge pair
cg:HRV -> child:ROI and child:ROI -> cg:HRV; noise on the caregiver heart (set 'cg_heart') or on the child brain
(set 'child_brain').
With coupling: the S1 design (three planted edges at the default gains), noise SD 0 / 0.5 / 1 on both caregiver nodes.
Both parts use the seeds of the matching S6 cell / S1 replicates and draw the noise from a separate stream, so noise SD 0
reproduces S6 and S1 exactly.
Readouts: uncorrected (real smooth->rough minus rough->smooth) and surrogate-corrected (median delta difference)
direction asymmetry and the study-level false-positive rate of both edges; with coupling, power, direction accuracy
and reverse-edge rate of each planted edge.

Usage: python S06b_unequal_snr.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_06/noise/
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.sim_generators import add_node_noise, simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src import sim_progress
from src.sim_inference import run_study, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 6
SETTINGS = {
    "quick": {"n_rep": 4, "null_levels": [0.0, 1.0], "coupled_levels": [0.0, 1.0]},
    "full": {"n_rep": 200, "null_levels": [0.0, 0.25, 0.5, 1.0, 2.0], "coupled_levels": [0.0, 0.5, 1.0]},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
NOISE_SETS = {"cg_heart": [3], "child_brain": [0]}            # nodes that get noise (no-coupling part)
COUPLED_NOISE_NODES = [1, 3]                                  # caregiver brain and heart (coupled part)

# S6 design and seeds (S06_smoothness.py)
S06_BASE_SEED = 20260106
S06_SETTINGS = {"quick": {"n_rep": 6, "n_dyads": 15, "radii": [0.5, 0.9]},
                "full": {"n_rep": 200, "n_dyads": 25, "radii": [0.5, 0.65, 0.8, 0.9, 0.95]}}[MODE]
S06_RADIUS = 0.9                                              # the S6 cell used (z-scored)
S06_EDGES = {"smooth->rough (cg:HRV->child:ROI)": (3, 0), "rough->smooth (child:ROI->cg:HRV)": (0, 3)}
S06_F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
# S1 design and seeds (S01_recovery_orientation.py)
S01_BASE_SEED = 20260101
S01_N_DYADS = {"quick": 10, "full": 25}[MODE]
S01_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
S01_F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
S01_PLANTED = {"cg:ROI->child:ROI": (1, 0, 1, 0.4), "cg:HRV->child:HRV": (3, 2, 1, 0.3), "cg:HRV->child:ROI": (3, 0, 2, 0.3)}
NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
CROSS_PERSON = ["cg:ROI->child:ROI", "child:ROI->cg:ROI", "cg:HRV->child:HRV", "child:HRV->cg:HRV",
                "cg:HRV->child:ROI", "child:HRV->cg:ROI", "cg:ROI->child:HRV", "child:ROI->cg:HRV"]

FS = 2.5
SEG_S = 60
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05
N_PERM = 1000
FPR_MAX = 0.10                        # corrected false-positive rate in every condition [proposed]
ASYMMETRY_MAX = 0.02                  # |corrected asymmetry| in every condition [proposed]


def s06_radius_ranges(r_smooth):
    """Pole-radius ranges of S6: rough ROI nodes and child heart, smooth caregiver heart (node 3)."""
    return np.array([[0.45, 0.55], [0.45, 0.55], [0.45, 0.55], [max(r_smooth - 0.03, 0.4), min(r_smooth + 0.03, 0.97)]])


def s06_seed(rep):
    """Seed of S6 replicate `rep` of the (S06_RADIUS, z-scored) cell, as built in S06_smoothness.main."""
    cells = [(r, z) for r in S06_SETTINGS["radii"] for z in (True, False)]
    seeds = spawn_seeds(S06_BASE_SEED, len(cells) * S06_SETTINGS["n_rep"])
    return seeds[cells.index((S06_RADIUS, True)) * S06_SETTINGS["n_rep"] + rep]


def s01_edges():
    """S1's scored edges (every off-diagonal edge, in S1's order, so the permutation draws match S1)."""
    return {f"{NODE_NAMES[s]}->{NODE_NAMES[t]}": (s, t) for s in range(4) for t in range(4) if s != t}


def run_task(task):
    """One study: S6 null design or S1 coupled design, with noise on some nodes."""
    rng = np.random.default_rng(task["seed"])
    noise_rng = np.random.default_rng([task["seed"], 3])          # observation noise: separate stream
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    tag = {k: task[k] for k in ("part", "noise_set", "noise_sd", "rep")}
    if task["part"] == "no_coupling":
        raw = simulate_dyads(rng, S06_SETTINGS["n_dyads"], int(SEG_S * FS), FS, s06_radius_ranges(S06_RADIUS), S06_F0_RANGES, [], N_LAGS)
        designs = zscore_designs(add_node_noise(raw, NOISE_SETS[task["noise_set"]], task["noise_sd"], noise_rng))
        records = run_study(designs, NODES, CHILD_NODES, est, S06_EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM)
    else:
        raw = simulate_dyads(rng, S01_N_DYADS, int(SEG_S * FS), FS, S01_RADIUS_RANGES, S01_F0_RANGES, list(S01_PLANTED.values()), N_LAGS)
        designs = zscore_designs(add_node_noise(raw, COUPLED_NOISE_NODES, task["noise_sd"], noise_rng))
        records = run_study(designs, NODES, CHILD_NODES, est, s01_edges(), ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA,
                            ALPHA, N_PERM, family=CROSS_PERSON, multiplicity="maxT", multiplicity_rng=np.random.default_rng([task["seed"], 1]))
    return [dict(r, **tag) for r in records]


def rate(flags):
    """Rate and Wilson 95% interval of a boolean series."""
    k, n = int(flags.astype(bool).sum()), len(flags)
    low, high = wilson_interval(k, n)
    return k / n, low, high


def null_summary(df):
    """Asymmetry and false-positive rates per noise set and level (no coupling)."""
    fwd, rev = list(S06_EDGES)
    rows = []
    for (noise_set, sd), sub in df[df.part == "no_coupling"].groupby(["noise_set", "noise_sd"]):
        a, b = sub[sub.edge == fwd].set_index("rep"), sub[sub.edge == rev].set_index("rep")
        row = {"noise_set": noise_set, "noise_sd": sd, "n_rep": len(a),
               "uncorrected_asymmetry": float((a.real_mean - b.real_mean).mean()),
               "corrected_asymmetry": float((a.median_delta - b.median_delta).mean()),
               "corrected_asymmetry_se": float((a.median_delta - b.median_delta).std() / np.sqrt(len(a)))}
        for label, part in (("smooth_to_rough", a), ("rough_to_smooth", b)):
            row[f"fpr_{label}"], row[f"fpr_{label}_ci_low"], row[f"fpr_{label}_ci_high"] = rate(part.study_reject)
        rows.append(row)
    return pd.DataFrame(rows)


def coupled_summary(df):
    """Power, max-T power, direction accuracy and reverse-edge rate of each planted S1 edge per noise level."""
    rows = []
    sub_all = df[df.part == "coupled"]
    for sd, sub in sub_all.groupby("noise_sd"):
        for name in S01_PLANTED:
            source, target = name.split("->")
            fwd, rev = sub[sub.edge == name], sub[sub.edge == f"{target}->{source}"]
            row = {"noise_sd": sd, "edge": name, "n_rep": len(fwd), "direction_accuracy": float(fwd.dir_study.mean())}
            for label, flags in (("power", fwd.study_reject), ("power_maxT", fwd.study_reject_adj_maxT),
                                 ("reverse_rate", rev.study_reject), ("reverse_rate_maxT", rev.study_reject_adj_maxT)):
                row[label], row[f"{label}_ci_low"], row[f"{label}_ci_high"] = rate(flags)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(null, coupled, out_dir):
    """fig_unequal_snr: A asymmetry, B false-positive rate (no coupling), C power and reverse rate (S1 edges)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (noise_set, sub) in enumerate(null.groupby("noise_set")):
        axes[0].plot(sub.noise_sd, sub.uncorrected_asymmetry, marker="o", color=f"C{i}", label=f"uncorrected, noise on {noise_set}")
        axes[0].errorbar(sub.noise_sd, sub.corrected_asymmetry, yerr=1.96 * sub.corrected_asymmetry_se, marker="s", ls="--",
                         capsize=2, color=f"C{i}", label=f"corrected, noise on {noise_set}")
        for label, ls in (("smooth_to_rough", "-"), ("rough_to_smooth", ":")):
            axes[1].plot(sub.noise_sd, sub[f"fpr_{label}"], marker="o", ls=ls, color=f"C{i}", label=f"{label}, noise on {noise_set}")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].axhspan(-ASYMMETRY_MAX, ASYMMETRY_MAX, color="g", alpha=0.1)
    axes[0].set_xlabel("noise SD / node SD"); axes[0].set_ylabel(f"{ESTIMATOR} asymmetry (smooth->rough minus reverse)")
    axes[0].set_title("A  direction asymmetry, no coupling", fontsize=10); axes[0].legend(fontsize=7)
    axes[1].axhline(ALPHA, color="k", lw=0.6); axes[1].axhline(FPR_MAX, color="r", lw=0.6, ls="--")
    axes[1].set_xlabel("noise SD / node SD"); axes[1].set_ylabel("false-positive rate (study level, corrected)")
    axes[1].set_title("B  false positives, no coupling", fontsize=10); axes[1].legend(fontsize=7)
    for i, (edge, sub) in enumerate(coupled.groupby("edge")):
        axes[2].plot(sub.noise_sd, sub.power, marker="o", color=f"C{i}", label=f"{edge}: power")
        axes[2].plot(sub.noise_sd, sub.reverse_rate, marker="x", ls=":", color=f"C{i}", label=f"{edge}: reverse edge")
    axes[2].axhline(ALPHA, color="grey", lw=0.6); axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel("noise SD / node SD (caregiver nodes)"); axes[2].set_ylabel("study-level rate")
    axes[2].set_title("C  S1 edges with noise on the caregiver", fontsize=10); axes[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_unequal_snr.{ext}", dpi=150)
    plt.close(fig)


def main():
    """Run S06b and write all outputs to Results/<estimator>/S_06/noise."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR) / "noise"
    sim_progress.begin(f"S06b {ESTIMATOR}", Path(__file__).resolve().parent / "Results", mode=MODE, replicates=SETTINGS["n_rep"])
    out_dir.mkdir(exist_ok=True)
    s01_seeds = spawn_seeds(S01_BASE_SEED, SETTINGS["n_rep"])
    tasks = [{"part": "no_coupling", "noise_set": ns, "noise_sd": sd, "rep": r, "seed": s06_seed(r)}
             for ns in NOISE_SETS for sd in SETTINGS["null_levels"] for r in range(SETTINGS["n_rep"])]
    tasks += [{"part": "coupled", "noise_set": "caregiver", "noise_sd": sd, "rep": r, "seed": s01_seeds[r]}
              for sd in SETTINGS["coupled_levels"] for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)
    null, coupled = null_summary(df), coupled_summary(df)
    null.to_csv(out_dir / "summary_no_coupling.csv", index=False)
    coupled.to_csv(out_dir / "summary_coupled.csv", index=False)
    plot(null, coupled, out_dir)
    criteria = {
        f"[proposed] no coupling: corrected false-positive rate <= {FPR_MAX} in every condition (both edges)":
            bool((null[["fpr_smooth_to_rough", "fpr_rough_to_smooth"]].max(axis=1) <= FPR_MAX).all()),
        f"[proposed] no coupling: |corrected asymmetry| <= {ASYMMETRY_MAX} in every condition":
            bool((null.corrected_asymmetry.abs() <= ASYMMETRY_MAX).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "s06_radius": S06_RADIUS, "criteria": criteria}, f, indent=2)
    print(null.round(3).to_string()); print(coupled.round(3).to_string()); print(criteria)
    sim_progress.finish(out_dir, {"FPR (no coupling)": sim_progress.range_text(null[["fpr_smooth_to_rough", "fpr_rough_to_smooth"]].values.ravel().tolist()), "power (coupled)": sim_progress.range_text(coupled.power.tolist())}, criteria)


if __name__ == "__main__":
    main()
