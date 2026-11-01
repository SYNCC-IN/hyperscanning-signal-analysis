"""S03 - Calibration of the surrogate test under zero true coupling (Table 2, S3).

Question: with no coupling at all, is the proportion of false positives close to the nominal level, and
is z centred near 0?
Design: heterogeneous dyads (pole radius, resonance frequency vary between persons), four nodes
(child/caregiver envelope-like ROI nodes, child/caregiver IBI-like resonant nodes), NO coupling. Factors: number
of dyads N, segment length, and the null pool: 'pooled' (all foreign pairs, which include the real dyad's
members) vs 'exclude' (foreign pairs involving neither member of the scored dyad).
Readouts per edge: dyad-level false-positive rate (z > 1.96), study-level (one-sided signed-rank test of
delta > 0, alpha = .05), group-level (permutation test of a difference between two groups of N/2 dyads that
differ in nothing), and the distribution of z.

Usage: python S03_null_calibration.py [quick|full]
Results: Results/S_03/
"""

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

from src.design import window_geometry
from src.sim_generators import simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import (delta_and_z_all, edge_summary, real_matrices, surrogate_values, wilson_interval)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
N_JOBS = 4
SIM_ID = 3
BASE_SEED = 20260103
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": [8], "seg_s": [60], "n_perm": 100, "max_pairs": None},
    "full": {"n_rep": 200, "n_dyads": [10, 25, 50], "seg_s": [60, 120, 300], "n_perm": 500, "max_pairs": None},
}[MODE]
import os
if os.environ.get("SIM_NREP"):            # optional override for diagnostics
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
    SETTINGS["n_dyads"] = [int(x) for x in os.environ.get("SIM_NDYADS", "25").split(",")]
    SETTINGS["seg_s"] = [int(x) for x in os.environ.get("SIM_SEG", "60").split(",")]
NULL_MODES = ["pooled", "exclude"]

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
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
         "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1)}
STABILITY_MAX_ROOT = 1.0
Z_CRIT, ALPHA = 1.96, 0.05
FPR_BAND = (0.03, 0.07)               # accepted false-positive rate around nominal 0.05 [proposed]
MEDIAN_Z_MAX = 0.2                    # |median z| must not exceed this [proposed]


def run_task(task):
    """One replicated null study: records per (null mode, edge) and the z values of the pooled null."""
    rng = np.random.default_rng(task["seed"])
    n_dyads, seg_s = task["n_dyads"], task["seg_s"]
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator="dDTF", band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    designs = zscore_designs(simulate_dyads(rng, n_dyads, int(seg_s * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS))
    groups = (np.arange(n_dyads) >= n_dyads // 2).astype(int)
    real, roots, _ = real_matrices(designs, NODES, est)
    values, n_attempted, n_unstable = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng, SETTINGS["max_pairs"])
    stable = roots < STABILITY_MAX_ROOT
    records, z_store = [], None
    for mode in NULL_MODES:
        delta, z = delta_and_z_all(real, values, mode, groups)
        if mode == "pooled":
            z_store = np.stack([z[stable][:, t, s] for s, t in EDGES.values()], axis=1)
        for name, edge in EDGES.items():
            summary = edge_summary(delta[stable], z[stable], edge, groups[stable], Z_CRIT, ALPHA, rng, SETTINGS["n_perm"])
            summary.update({"n_dyads": n_dyads, "seg_s": seg_s, "mode": mode, "edge": name, "rep": task["rep"],
                            "n_real_unstable": int((~stable).sum()), "n_pairs_unstable": n_unstable, "n_pairs": n_attempted})
            records.append(summary)
    return records, z_store, n_dyads, seg_s


def main():
    """Run S03 and write all outputs to Results/S_03."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID)
    cells = [(n, s) for n in SETTINGS["n_dyads"] for s in SETTINGS["seg_s"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"n_dyads": n, "seg_s": s, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (n, s) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    results = run_parallel(run_task, tasks, N_JOBS)
    df = pd.DataFrame([r for recs, *_ in results for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    keys = ["n_dyads", "seg_s", "mode", "edge"]
    summary = df.groupby(keys).agg(
        dyad_fpr=("dyad_rate", "mean"), study_fpr=("study_reject", "mean"), group_fpr=("group_reject", "mean"),
        median_z=("median_z", "median"), sd_z=("sd_z", "mean"), unstable_pairs=("n_pairs_unstable", "mean"), n_rep=("rep", "size")).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.study_fpr * r.n_rep)), int(r.n_rep)), axis=1)
    summary["study_fpr_ci_low"], summary["study_fpr_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary.to_csv(out_dir / "summary.csv", index=False)

    z_all = {}
    for recs, z_store, n, s in results:
        z_all.setdefault((n, s), []).append(z_store)
    np.savez(out_dir / "z_values_pooled.npz", **{f"N{n}_seg{s}": np.concatenate(v) for (n, s), v in z_all.items()}, edges=np.array(list(EDGES)))

    # Figures: FPR by cell, z distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, column, title in zip(axes, ["dyad_fpr", "study_fpr", "group_fpr"], ["dyad level (z > 1.96)", "study level (signed-rank)", "group contrast (permutation)"]):
        for (mode, n), sub in summary.groupby(["mode", "n_dyads"]):
            sub = sub.groupby("seg_s")[column].mean()
            ax.plot(sub.index, sub.values, marker="o", ls="-" if mode == "pooled" else "--", label=f"{mode}, N={n}")
        ax.axhline(0.05, color="k", lw=0.8); ax.axhspan(*FPR_BAND, color="g", alpha=0.1)
        ax.set_title(title); ax.set_xlabel("segment length (s)")
    axes[0].set_ylabel("false-positive rate (mean over edges)"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_dir / "fig_fpr_by_cell.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ref = max(z_all, key=lambda c: (c[0] == 25, c[1] == 60, c[0]))
    z_ref = np.concatenate(z_all[ref])
    ax.hist(z_ref[np.isfinite(z_ref)].ravel(), bins=60, range=(-6, 6), density=True, alpha=0.6)
    grid = np.linspace(-6, 6, 200); ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=1)
    ax.set_title(f"z under zero coupling (N={ref[0]}, {ref[1]} s, all edges)"); ax.set_xlabel("z")
    fig.tight_layout(); fig.savefig(out_dir / "fig_z_distribution.png", dpi=150); plt.close(fig)

    in_band = lambda x: (x >= FPR_BAND[0]) & (x <= FPR_BAND[1])
    cell_means = summary.groupby(keys[:3])[["study_fpr", "group_fpr", "median_z"]].mean()
    criteria = {
        "study-level FPR within band in every cell (mean over edges)": bool(in_band(cell_means.study_fpr).all()),
        "group-level FPR within band in every cell (mean over edges)": bool(in_band(cell_means.group_fpr).all()),
        "|median z| <= limit in every cell": bool((cell_means.median_z.abs() <= MEDIAN_Z_MAX).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "fpr_band": FPR_BAND, "criteria": criteria,
                   "cells_outside_band_study": cell_means[~in_band(cell_means.study_fpr)].round(3).reset_index().to_dict(orient="records")}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
