"""S04 - Power and direction accuracy as a function of coupling strength, N and segment length (Table 2, S4).

Design A (one group): every dyad has the planted edge cg:ROI->child:ROI with gain g (g = 0 gives the false-positive rate).
Design B (group contrast): half of the dyads have gain g, the other half gain 0; the test is the permutation
test on the group difference of delta, using the within-group null.
Readouts: dyad-level detection rate (empirical p < .05), study-level power (pairing permutation, alpha = .05), group-contrast power,
direction accuracy (planted edge beats its reverse, dir_study), with Wilson intervals.
Multiplicity: all 8 cross-person edges form one family, tested on one shared set of permutations (max-T and min-p
step-down, Bonferroni). Readouts: planted-edge power before/after correction and the rejection rate of every non-planted
edge after max-T. The per-edge (uncorrected) tests are run for the planted edge and its reverse only, as before.
Small-N group contrast (`extra_b_cells`): design B with 5 vs 5 dyads at 60 s, gains 0 / 0.2 / 0.4, runs to 1000 replicates;
replicates beyond the first 200 run design B only, on a separate seed stream (the first 200 are unchanged).

Usage: python S04_power.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_04/
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
from src.sim_generators import simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src import sim_progress
from src.sim_inference import run_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 4
BASE_SEED = 20260104
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": [10], "seg_s": [60], "gains": [0.0, 0.3, 0.6], "extra_b_cells": {(10, 60, 0.0): 10}},
    "full": {"n_rep": 200, "n_dyads": [10, 25, 50], "seg_s": [60, 120, 300], "gains": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
             "extra_b_cells": {(10, 60, 0.0): 1000, (10, 60, 0.2): 1000, (10, 60, 0.4): 1000}},
}[MODE]                                   # extra_b_cells: (N, segment s, gain) -> total replicates of design B
N_REP_EXTRA_MAX = 10000                   # seed index stride of the extra stream
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
    SETTINGS["extra_b_cells"] = {}

FS = 2.5
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
TARGET_EDGE = "cg:ROI->child:ROI"
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1)}       # per-edge (uncorrected) tests
FAMILY = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
          "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1),
          "cg:ROI->child:HRV": (1, 2), "child:ROI->cg:HRV": (0, 3)}       # multiplicity family: all 8 cross-person edges
MULTIPLICITY = ["bonferroni", "maxT", "minP"]                         # step-down max-T / min-p
POWER_COLUMNS = {"none": "uncorrected", "maxT": "max-T", "minP": "min-p", "bonferroni": "Bonferroni"}
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
POWER_TARGET = 0.80                   # conventional target [proposed]
GAIN_FOR_POWER_CRITERION = 0.4        # [proposed]


def run_task(task):
    """One replicated study for (N, segment length, gain): design A and design B (only B if ``task["only_b"]``)."""
    rng = np.random.default_rng(task["seed"])
    n, seg_s, gain = task["n_dyads"], task["seg_s"], task["gain"]
    n_samples = int(seg_s * FS)
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    family_rng = np.random.default_rng([task["seed"], 1])     # separate stream: data and per-edge tests as without the family test
    mult = dict(family=FAMILY, multiplicity=MULTIPLICITY, multiplicity_rng=family_rng)
    out = []
    # Design A: all dyads coupled
    if not task.get("only_b"):
        designs = zscore_designs(simulate_dyads(rng, n, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [(1, 0, 1, gain)], N_LAGS))
        for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM, **mult):
            out.append(dict(r, design="A", n_dyads=n, seg_s=seg_s, gain=gain, rep=task["rep"]))
    # Design B: first half uncoupled, second half coupled
    half = n // 2
    d0 = simulate_dyads(rng, half, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS)
    d1 = simulate_dyads(rng, n - half, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [(1, 0, 1, gain)], N_LAGS)
    designs = zscore_designs(np.concatenate([d0, d1]))
    groups = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["within_group"], groups, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM, **mult):
        out.append(dict(r, design="B", n_dyads=n, seg_s=seg_s, gain=gain, rep=task["rep"]))
    return out


def min_gain(sub, column):
    """Smallest planted gain whose power in `column` reaches POWER_TARGET (None if never)."""
    hit = sub[sub[column] >= POWER_TARGET]
    return float(hit.gain.min()) if len(hit) else None


def nonplanted_table(df):
    """Rejection rate of every non-planted family edge after max-T (and min-p, Bonferroni), per design and cell.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw records (design A: study level; design B: group contrast).

    Returns
    -------
    pandas.DataFrame
        Columns design, n_dyads, seg_s, gain, edge, rate_<method> (with Wilson CI for max-T), rate_uncorrected
        (per-edge test, reverse edge only; NaN for edges without a per-edge test), n_rep.
    """
    rows = []
    for (design, n, s, g, edge), sub in df[df.edge != TARGET_EDGE].groupby(["design", "n_dyads", "seg_s", "gain", "edge"]):
        level = "group" if design == "B" else "study"
        row = {"design": design, "n_dyads": n, "seg_s": s, "gain": g, "edge": edge, "n_rep": len(sub),
               "rate_uncorrected": float(sub[f"{level}_reject"].astype(float).mean()) if sub[f"{level}_reject"].notna().all() else np.nan}
        for m in MULTIPLICITY:
            row[f"rate_{m}"] = float(sub[f"{level}_reject_adj_{m}"].astype(bool).mean())
        row["rate_maxT_ci_low"], row["rate_maxT_ci_high"] = wilson_interval(int(sub[f"{level}_reject_adj_maxT"].astype(bool).sum()), len(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_maxT(summary, nonplanted, out_dir):
    """Planted-edge power before (solid) and after max-T (dashed); non-planted edges after max-T at the middle N (linear axes)."""
    segs, ns = SETTINGS["seg_s"], SETTINGS["n_dyads"]
    fig, axes = plt.subplots(2, len(segs), figsize=(4.5 * len(segs), 7), sharey=True, squeeze=False)
    for row, design in enumerate(["A", "B"]):
        for col, s in enumerate(segs):
            ax = axes[row, col]
            for i, (n, sub) in enumerate(summary[(summary.design == design) & (summary.seg_s == s)].groupby("n_dyads")):
                ax.plot(sub.gain, sub.power, marker="o", color=f"C{i}", label=f"N={n} uncorrected")
                ax.plot(sub.gain, sub.power_maxT, marker="s", ls="--", color=f"C{i}", label=f"N={n} max-T")
            ax.axhline(POWER_TARGET, color="k", lw=0.6, ls=":"); ax.axhline(ALPHA, color="grey", lw=0.6)
            ax.set_title(f"design {design}: {'one group' if design == 'A' else 'group contrast'}, {s} s", fontsize=9)
            ax.set_xlabel("planted gain")
        axes[row, 0].set_ylabel("power, planted edge")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_power_maxT.{ext}", dpi=150)
    plt.close(fig)

    n_mid = ns[len(ns) // 2]
    fig, axes = plt.subplots(2, len(segs), figsize=(4.5 * len(segs), 7), sharey=True, squeeze=False)
    for row, design in enumerate(["A", "B"]):
        for col, s in enumerate(segs):
            ax = axes[row, col]
            for edge, sub in nonplanted[(nonplanted.design == design) & (nonplanted.seg_s == s) & (nonplanted.n_dyads == n_mid)].groupby("edge"):
                ax.plot(sub.gain, sub.rate_maxT, marker="o", lw=2.5 if edge == "child:ROI->cg:ROI" else 1, label=edge)
            ax.axhline(ALPHA, color="grey", lw=0.6)
            ax.set_title(f"design {design}, N={n_mid}, {s} s: non-planted edges after max-T", fontsize=9)
            ax.set_xlabel("planted gain (cg:ROI->child:ROI)")
        axes[row, 0].set_ylabel("rejection rate after max-T")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_nonplanted_maxT.{ext}", dpi=150)
    plt.close(fig)


def main():
    """Run S04 and write all outputs to Results/<estimator>/S_04."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    sim_progress.begin(f"S04 {ESTIMATOR}", Path(__file__).resolve().parent / "Results", mode=MODE, replicates=SETTINGS["n_rep"])
    cells = [(n, s, g) for n in SETTINGS["n_dyads"] for s in SETTINGS["seg_s"] for g in SETTINGS["gains"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"n_dyads": n, "seg_s": s, "gain": g, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (n, s, g) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    extra_seeds = spawn_seeds([BASE_SEED, 1], len(SETTINGS["extra_b_cells"]) * N_REP_EXTRA_MAX)
    tasks += [{"n_dyads": n, "seg_s": s, "gain": g, "rep": r, "only_b": True, "seed": extra_seeds[j * N_REP_EXTRA_MAX + r]}
              for j, ((n, s, g), total) in enumerate(SETTINGS["extra_b_cells"].items())
              for r in range(SETTINGS["n_rep"] if (n, s, g) in cells else 0, total)]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    tgt = df[df.edge == TARGET_EDGE]
    rows = []
    for (design, n, s, g), sub in tgt.groupby(["design", "n_dyads", "seg_s", "gain"]):
        level = "group" if design == "B" else "study"
        k = int(sub[f"{level}_reject"].sum())
        lo, hi = wilson_interval(k, len(sub))
        row = {"design": design, "n_dyads": n, "seg_s": s, "gain": g, "power": k / len(sub), "power_ci_low": lo, "power_ci_high": hi,
               "dyad_detection": sub.dyad_rate.mean(), "direction_accuracy": sub.dir_study.mean() if design == "A" else np.nan,
               "median_z": sub.median_z.median(), "n_rep": len(sub)}
        for m in MULTIPLICITY:
            k_m = int(sub[f"{level}_reject_adj_{m}"].astype(bool).sum())
            row[f"power_{m}"] = k_m / len(sub)
            row[f"power_{m}_ci_low"], row[f"power_{m}_ci_high"] = wilson_interval(k_m, len(sub))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summary[summary.n_rep > SETTINGS["n_rep"]].to_csv(out_dir / "summary_small_n_group.csv", index=False)
    nonplanted = nonplanted_table(df)
    nonplanted.to_csv(out_dir / "summary_nonplanted_maxT.csv", index=False)
    plot_maxT(summary, nonplanted, out_dir)

    fig, axes = plt.subplots(2, len(SETTINGS["seg_s"]), figsize=(4.5 * len(SETTINGS["seg_s"]), 7), sharey=True, squeeze=False)
    for row, design in enumerate(["A", "B"]):
        for col, s in enumerate(SETTINGS["seg_s"]):
            ax = axes[row, col]
            for n, sub in summary[(summary.design == design) & (summary.seg_s == s)].groupby("n_dyads"):
                ax.errorbar(sub.gain, sub.power, yerr=[(sub.power - sub.power_ci_low).clip(lower=0), (sub.power_ci_high - sub.power).clip(lower=0)], marker="o", capsize=2, label=f"N={n}")
            ax.axhline(POWER_TARGET, color="k", lw=0.6, ls=":"); ax.axhline(ALPHA, color="grey", lw=0.6)
            ax.set_title(f"design {design}: {'one group' if design == 'A' else 'group contrast'}, {s} s"); ax.set_xlabel("planted gain")
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel("power"); axes[1, 0].set_ylabel("power")
    fig.tight_layout(); fig.savefig(out_dir / "fig_power_curves.png", dpi=150); plt.close(fig)

    a = summary[summary.design == "A"]
    ref_gain = min(SETTINGS["gains"], key=lambda g: abs(g - GAIN_FOR_POWER_CRITERION))
    zero = a[a.gain == 0.0]
    criteria = {
        "false-positive rate at gain 0 <= 0.10 in every cell (design A)": bool((zero.power <= 0.10).all()),
        f"power >= {POWER_TARGET} at gain {ref_gain} in the largest N and segment (design A)":
            bool(a[(a.gain == ref_gain) & (a.n_dyads == max(SETTINGS["n_dyads"])) & (a.seg_s == max(SETTINGS["seg_s"]))].power.min() >= POWER_TARGET),
        "power increases with gain at the largest N (design A)":
            bool(a[(a.n_dyads == max(SETTINGS["n_dyads"])) & (a.seg_s == max(SETTINGS["seg_s"]))].sort_values("gain").power.is_monotonic_increasing),
    }
    thresholds = {f"{d}|N={n}|{s}s": min_gain(sub, "power") for (d, n, s), sub in summary.groupby(["design", "n_dyads", "seg_s"])}
    thresholds_adj = {m: {f"{d}|N={n}|{s}s": min_gain(sub, f"power_{m}") for (d, n, s), sub in summary.groupby(["design", "n_dyads", "seg_s"])}
                      for m in MULTIPLICITY}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": dict(SETTINGS, extra_b_cells={f"N{n}_{s}s_gain{g}": t for (n, s, g), t in SETTINGS["extra_b_cells"].items()}),
                   "small_n_group_contrast": summary[summary.n_rep > SETTINGS["n_rep"]][["n_dyads", "seg_s", "gain", "power", "power_ci_low", "power_ci_high",
                                                                                         "power_maxT", "power_maxT_ci_low", "power_maxT_ci_high", "n_rep"]].to_dict(orient="records"),
                   "family": list(FAMILY), "min_gain_for_80pct_power": thresholds,
                   "min_gain_for_80pct_power_adjusted": thresholds_adj, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)
    sim_progress.finish(out_dir, {"FPR at gain 0 (design A)": sim_progress.range_text(a[a.gain == 0].power.tolist()), "power (A, N=25, 60 s, gain 0.4)": sim_progress.range_text(a[(a.n_dyads == 25) & (a.seg_s == 60) & (a.gain == 0.4)].power.tolist())}, criteria)


if __name__ == "__main__":
    main()
