"""S03 - Calibration of the surrogate test under zero true coupling (Table 2, S3).

Question: with no coupling at all, is the proportion of false positives close to the nominal level, and
is z centred near 0?
Design: heterogeneous dyads (pole radius, resonance frequency vary between persons), four nodes
(child/caregiver envelope-like ROI nodes, child/caregiver IBI-like resonant nodes), NO coupling. Factors: number
of dyads N, segment length, and the null pool: 'pooled' (all foreign pairs, which include the real dyad's
members) vs 'exclude' (foreign pairs involving neither member of the scored dyad) vs 'within_group' (pairs formed within the scored dyad's group).
Readouts per edge: dyad-level false-positive rate (empirical p < .05 among the surrogates), study-level (one-sided
pairing permutation test, alpha = .05), group-level (permutation test of a difference between two groups of N/2 dyads that
differ in nothing), and the distribution of z.
Small-N cells with more replicates (`extra_cells`): the N = 10 cells and a new N = 16 cell (8 vs 8 dyads, 60 s) run to
1000 replicates; replicates beyond the first 200 (and the whole N = 16 cell) use a separate seed stream, so the first 200
are unchanged. Every rate is reported with a 95% Wilson interval.
Multiplicity: the 8 cross-person edges form one family. Family-wise false-positive rate (any edge rejected) and the
per-edge rate after correction, for uncorrected tests, Bonferroni, max-T single-step, max-T step-down and min-p
(one shared set of permutations for the whole family; `src.sim_inference.pairing_permutation_maxT` and
`group_permutation_maxT`).

Usage: python S03_null_calibration.py [quick|full] [dDTF|GPDC]
Results: Results/<estimator>/S_03/
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
from src.sim_inference import (delta_and_z_all, real_matrices, score_study, surrogate_values, wilson_interval)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 3
BASE_SEED = 20260103
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": [8], "seg_s": [60], "max_pairs": None, "extra_cells": {(8, 60): 10, (10, 60): 6}},
    "full": {"n_rep": 200, "n_dyads": [10, 25, 50], "seg_s": [60, 120, 300], "max_pairs": None,
             "extra_cells": {(10, 60): 1000, (10, 120): 1000, (10, 300): 1000, (16, 60): 1000}},
}[MODE]                                   # extra_cells: (N, segment s) -> total replicates (separate seed stream)
N_REP_EXTRA_MAX = 10000                   # seed index stride of the extra stream
import os
if os.environ.get("SIM_NREP"):            # optional override for diagnostics
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
    SETTINGS["n_dyads"] = [int(x) for x in os.environ.get("SIM_NDYADS", "25").split(",")]
    SETTINGS["seg_s"] = [int(x) for x in os.environ.get("SIM_SEG", "60").split(",")]
    SETTINGS["extra_cells"] = {}
NULL_MODES = ["pooled", "exclude", "within_group"]

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
         "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1),
         "cg:ROI->child:HRV": (1, 2), "child:ROI->cg:HRV": (0, 3)}   # all 8 cross-person edges
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
FPR_BAND = (0.03, 0.07)               # accepted false-positive rate around nominal 0.05 [proposed]
FWER_BAND = (0.03, 0.07)              # accepted family-wise FPR of max-T (step-down) in every cell [proposed]
MULTIPLICITY = ["unadjusted", "bonferroni", "maxT_single_step", "maxT", "minP"]   # family tests (shared permutations)
METHOD_LABELS = {"none": "uncorrected", "bonferroni": "Bonferroni", "maxT_single_step": "max-T single-step",
                 "maxT": "max-T step-down", "minP": "min-p step-down"}
MEDIAN_Z_MAX = 0.2                    # |median z| must not exceed this [proposed]


def run_task(task):
    """One replicated null study: records per (null mode, edge) and the z values of the pooled null."""
    rng = np.random.default_rng(task["seed"])
    n_dyads, seg_s = task["n_dyads"], task["seg_s"]
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    designs = zscore_designs(simulate_dyads(rng, n_dyads, int(seg_s * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS))
    groups = (np.arange(n_dyads) >= n_dyads // 2).astype(int)
    real, roots, _ = real_matrices(designs, NODES, est)
    values, n_attempted, n_unstable = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng, SETTINGS["max_pairs"])
    stable = roots < STABILITY_MAX_ROOT
    real, values, groups = real[stable], values[stable][:, stable], groups[stable]
    _, z = delta_and_z_all(real, values, "pooled", groups)
    z_store = np.stack([z[:, t, s] for s, t in EDGES.values()], axis=1)
    records = []
    family_rng = np.random.default_rng([task["seed"], 1])     # separate stream: per-edge tests stay as without the family test
    for mode in NULL_MODES:
        summaries = score_study(real, values, real, mode, groups, EDGES, DYAD_ALPHA, ALPHA, rng, N_PERM,
                                family=list(EDGES), multiplicity=MULTIPLICITY, multiplicity_rng=family_rng)
        for name, summary in summaries.items():
            summary.update({"n_dyads": n_dyads, "seg_s": seg_s, "mode": mode, "edge": name, "rep": task["rep"],
                            "n_real_unstable": int((~stable).sum()), "n_pairs_unstable": n_unstable, "n_pairs": n_attempted})
            records.append(summary)
    return records, z_store, n_dyads, seg_s


def fwer_table(df):
    """Family-wise and per-edge false-positive rates per cell, null mode, level and correction method.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw records (one row per replicate x mode x edge, all edges in the family).

    Returns
    -------
    pandas.DataFrame
        Columns n_dyads, seg_s, mode, level ('study' or 'group'), method, fwer (share of studies with at least one
        rejection), fwer_ci_low/high (Wilson 95%), per_edge_rate (mean rejection rate over the family edges), n_rep.
    """
    rows = []
    for (n, s, mode), sub in df.groupby(["n_dyads", "seg_s", "mode"]):
        per_study = sub.groupby("rep").first()
        for level in ("study", "group"):
            for method in ["none"] + [m for m in MULTIPLICITY if m != "unadjusted"]:
                reject = sub[f"{level}_reject"] if method == "none" else sub[f"{level}_reject_adj_{method}"]
                hits = int(per_study[f"any_reject_{level}_{method}"].astype(bool).sum())
                low, high = wilson_interval(hits, len(per_study))
                rows.append({"n_dyads": n, "seg_s": s, "mode": mode, "level": level, "method": method,
                             "fwer": hits / len(per_study), "fwer_ci_low": low, "fwer_ci_high": high,
                             "per_edge_rate": float(reject.astype(float).mean()), "n_rep": len(per_study)})
    return pd.DataFrame(rows)


def plot_fwer(fwer, out_dir):
    """Family-wise FPR per cell: rows = level, columns = null mode, one line per correction method (linear axes)."""
    levels = ["study", "group"]
    fig, axes = plt.subplots(len(levels), len(NULL_MODES), figsize=(4.5 * len(NULL_MODES), 7), sharey=True, squeeze=False)
    for i, level in enumerate(levels):
        for j, mode in enumerate(NULL_MODES):
            ax = axes[i, j]
            sub = fwer[(fwer.level == level) & (fwer["mode"] == mode)]
            for method, s2 in sub.groupby("method", sort=False):
                s2 = s2.sort_values(["n_dyads", "seg_s"])
                x = np.arange(len(s2))
                ax.errorbar(x, s2.fwer, yerr=[s2.fwer - s2.fwer_ci_low, s2.fwer_ci_high - s2.fwer], marker="o", ms=3,
                            capsize=2, lw=1, label=METHOD_LABELS[method])
            ax.set_xticks(x); ax.set_xticklabels([f"N{n}\n{s}s" for n, s in zip(s2.n_dyads, s2.seg_s)], fontsize=7)
            ax.axhline(ALPHA, color="k", lw=0.8); ax.axhspan(*FWER_BAND, color="g", alpha=0.1)
            ax.set_title(f"{level} level, {mode} null", fontsize=9)
        axes[i, 0].set_ylabel("family-wise FPR (any of 8 edges)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_fwer_by_cell.{ext}", dpi=150)
    plt.close(fig)


def main():
    """Run S03 and write all outputs to Results/<estimator>/S_03."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [(n, s) for n in SETTINGS["n_dyads"] for s in SETTINGS["seg_s"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"n_dyads": n, "seg_s": s, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (n, s) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    extra_seeds = spawn_seeds([BASE_SEED, 1], len(SETTINGS["extra_cells"]) * N_REP_EXTRA_MAX)
    tasks += [{"n_dyads": n, "seg_s": s, "rep": r, "seed": extra_seeds[j * N_REP_EXTRA_MAX + r]}
              for j, ((n, s), total) in enumerate(SETTINGS["extra_cells"].items())
              for r in range(SETTINGS["n_rep"] if (n, s) in cells else 0, total)]
    results = run_parallel(run_task, tasks, N_JOBS)
    df = pd.DataFrame([r for recs, *_ in results for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    keys = ["n_dyads", "seg_s", "mode", "edge"]
    summary = df.groupby(keys).agg(
        dyad_fpr=("dyad_rate", "mean"), study_fpr=("study_reject", "mean"), group_fpr=("group_reject", "mean"),
        median_z=("median_z", "median"), sd_z=("sd_z", "mean"), unstable_pairs=("n_pairs_unstable", "mean"), n_rep=("rep", "size")).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.study_fpr * r.n_rep)), int(r.n_rep)), axis=1)
    summary["study_fpr_ci_low"], summary["study_fpr_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    ci = summary.apply(lambda r: wilson_interval(int(round(r.group_fpr * r.n_rep)), int(r.n_rep)), axis=1)
    summary["group_fpr_ci_low"], summary["group_fpr_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    small_n = summary[summary.n_rep > SETTINGS["n_rep"]]
    small_n.to_csv(out_dir / "summary_small_n.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    fwer = fwer_table(df)
    fwer.to_csv(out_dir / "summary_fwer.csv", index=False)

    z_all = {}
    for recs, z_store, n, s in results:
        z_all.setdefault((n, s), []).append(z_store)
    np.savez(out_dir / "z_values_pooled.npz", **{f"N{n}_seg{s}": np.concatenate(v) for (n, s), v in z_all.items()}, edges=np.array(list(EDGES)))

    # Figures: FPR by cell, z distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, column, title in zip(axes, ["dyad_fpr", "study_fpr", "group_fpr"], [f"dyad level (empirical p < {DYAD_ALPHA})", "study level (pairing permutation)", "group contrast (permutation)"]):
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

    plot_fwer(fwer, out_dir)
    in_band = lambda x: (x >= FPR_BAND[0]) & (x <= FPR_BAND[1])
    max_t = fwer[fwer.method == "maxT"]
    fwer_in_band = (max_t.fwer >= FWER_BAND[0]) & (max_t.fwer <= FWER_BAND[1])
    ci_overlaps = (max_t.fwer_ci_high >= FWER_BAND[0]) & (max_t.fwer_ci_low <= FWER_BAND[1])
    cell_means = summary.groupby(keys[:3])[["study_fpr", "group_fpr", "median_z"]].mean()
    criteria = {
        "study-level FPR within band in every cell (mean over edges)": bool(in_band(cell_means.study_fpr).all()),
        "group-level FPR within band in every cell (mean over edges)": bool(in_band(cell_means.group_fpr).all()),
        "|median z| <= limit in every cell": bool((cell_means.median_z.abs() <= MEDIAN_Z_MAX).all()),
        "max-T family-wise FPR within band in every cell (study level, 8 cross-person edges)": bool(fwer_in_band[max_t.level == "study"].all()),
        "max-T family-wise FPR within band in every cell (group contrast, 8 cross-person edges)": bool(fwer_in_band[max_t.level == "group"].all()),
    }
    diagnostics = {"max-T FWER: Wilson 95% CI overlaps the band in every cell (study level)": bool(ci_overlaps[max_t.level == "study"].all()),
                   "max-T FWER: Wilson 95% CI overlaps the band in every cell (group contrast)": bool(ci_overlaps[max_t.level == "group"].all()),
                   "max-T cells outside the band": max_t[~fwer_in_band][["level", "mode", "n_dyads", "seg_s", "fwer"]].round(3).to_dict(orient="records")}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": dict(SETTINGS, extra_cells={f"N{n}_{s}s": t for (n, s), t in SETTINGS["extra_cells"].items()}), "fpr_band": FPR_BAND, "fwer_band": FWER_BAND, "criteria": criteria,
                   "fwer_diagnostics": diagnostics,
                   "small_n_group_fpr_mean_over_edges": small_n.groupby(["n_dyads", "seg_s", "mode"]).agg(
                       group_fpr=("group_fpr", "mean"), n_rep=("n_rep", "first")).reset_index().assign(
                       ci=lambda d: [wilson_interval(int(round(r.group_fpr * r.n_rep)), int(r.n_rep)) for r in d.itertuples()]).to_dict(orient="records"),
                   "cells_outside_band_study": cell_means[~in_band(cell_means.study_fpr)].round(3).reset_index().to_dict(orient="records")}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
