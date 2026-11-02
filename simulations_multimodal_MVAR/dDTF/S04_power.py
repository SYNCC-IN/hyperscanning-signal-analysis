"""S04 - Power and direction accuracy as a function of coupling strength, N and segment length (Table 2, S4).

Design A (one group): every dyad has the planted edge cg:ROI->child:ROI with gain g (g = 0 gives the false-positive rate).
Design B (group contrast): half of the dyads have gain g, the other half gain 0; the test is the permutation
test on the group difference of delta, using the within-group null.
Readouts: dyad-level detection rate (empirical p < .05), study-level power (pairing permutation, alpha = .05), group-contrast power,
direction accuracy (planted edge beats its reverse, dir_study), with Wilson intervals.

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
from src.sim_inference import run_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 4
BASE_SEED = 20260104
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": [10], "seg_s": [60], "gains": [0.0, 0.3, 0.6]},
    "full": {"n_rep": 200, "n_dyads": [10, 25, 50], "seg_s": [60, 120, 300], "gains": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

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
NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
TARGET_EDGE = "cg:ROI->child:ROI"
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1)}
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
POWER_TARGET = 0.80                   # conventional target [proposed]
GAIN_FOR_POWER_CRITERION = 0.4        # [proposed]


def run_task(task):
    """One replicated study for (N, segment length, gain): design A and design B."""
    rng = np.random.default_rng(task["seed"])
    n, seg_s, gain = task["n_dyads"], task["seg_s"], task["gain"]
    n_samples = int(seg_s * FS)
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    out = []
    # Design A: all dyads coupled
    designs = zscore_designs(simulate_dyads(rng, n, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [(1, 0, 1, gain)], N_LAGS))
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
        out.append(dict(r, design="A", n_dyads=n, seg_s=seg_s, gain=gain, rep=task["rep"]))
    # Design B: first half uncoupled, second half coupled
    half = n // 2
    d0 = simulate_dyads(rng, half, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS)
    d1 = simulate_dyads(rng, n - half, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES, [(1, 0, 1, gain)], N_LAGS)
    designs = zscore_designs(np.concatenate([d0, d1]))
    groups = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["within_group"], groups, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
        out.append(dict(r, design="B", n_dyads=n, seg_s=seg_s, gain=gain, rep=task["rep"]))
    return out


def main():
    """Run S04 and write all outputs to Results/<estimator>/S_04."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [(n, s, g) for n in SETTINGS["n_dyads"] for s in SETTINGS["seg_s"] for g in SETTINGS["gains"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"n_dyads": n, "seg_s": s, "gain": g, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (n, s, g) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    tgt = df[df.edge == TARGET_EDGE]
    rows = []
    for (design, n, s, g), sub in tgt.groupby(["design", "n_dyads", "seg_s", "gain"]):
        k = int(sub.group_reject.sum() if design == "B" else sub.study_reject.sum())
        lo, hi = wilson_interval(k, len(sub))
        rows.append({"design": design, "n_dyads": n, "seg_s": s, "gain": g, "power": k / len(sub), "power_ci_low": lo, "power_ci_high": hi,
                     "dyad_detection": sub.dyad_rate.mean(), "direction_accuracy": sub.dir_study.mean() if design == "A" else np.nan,
                     "median_z": sub.median_z.median(), "n_rep": len(sub)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

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
    thresholds = {f"{d}|N={n}|{s}s": (float(sub[sub.power >= POWER_TARGET].gain.min()) if (sub.power >= POWER_TARGET).any() else None)
                  for (d, n, s), sub in summary.groupby(["design", "n_dyads", "seg_s"])}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "min_gain_for_80pct_power": thresholds, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
