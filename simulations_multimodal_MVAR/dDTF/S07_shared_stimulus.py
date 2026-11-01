"""S07 - Shared stimulus, group-dependent stimulus gain, pooled vs within-group null, film matching (Table 2, S7).

No true cross-person coupling. All dyads watch the same film: a common stimulus drives every node with a
person-specific gain (log-normal variability). Two groups: group 1 has its stimulus gain multiplied by `ratio`.
Questions:
 (a) one group: does the film-matched surrogate test keep the false-positive rate at nominal level when the stimulus
     induces synchrony in real and surrogate dyads alike? (modes: pooled, exclude)
 (b) group contrast: if the groups differ only in stimulus gain, does the pooled null produce a spurious group
     difference, and does the within-group null remove it? (modes: pooled, within_group)
 (c) film matching: what happens when the null is built from dyads that watched a different film?
Readouts: study-level and group-contrast false-positive rates of the cross-person edges.

Usage: python S07_shared_stimulus.py [quick|full]       Results: Results/S_07/
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
from src.sim_generators import make_stimulus, sample_stimulus_gain, simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import (delta_and_z_all, edge_summary, real_matrices, run_study, surrogate_values, wilson_interval)

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
N_JOBS = 4
SIM_ID = 7
BASE_SEED = 20260107
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 16, "gains": [0.0, 1.0], "sd_log": [0.5], "ratios": [1.0, 2.0], "seg_s": 60, "n_perm": 100},
    "full": {"n_rep": 200, "n_dyads": 30, "gains": [0.0, 0.5, 1.0, 2.0], "sd_log": [0.0, 0.5], "ratios": [1.0, 2.0, 3.0], "seg_s": 60, "n_perm": 500},
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
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2), "child:HRV->cg:HRV": (2, 3)}
STIM_RADIUS, STIM_F0 = 0.7, 0.3
STABILITY_MAX_ROOT = 1.0
Z_CRIT, ALPHA = 1.96, 0.05
FPR_MAX = 0.10                        # [proposed]


def simulate_study(rng, gain, sd_log, ratio, stimulus):
    """One film's dyads with group-dependent stimulus gain; returns z-scored designs and group labels."""
    n = SETTINGS["n_dyads"]
    groups = (np.arange(n) >= n // 2).astype(int)
    multiplier = np.where(groups == 1, ratio, 1.0)
    stim_gain = sample_stimulus_gain(rng, n, [gain] * 4, sd_log, multiplier)
    raw = simulate_dyads(rng, n, int(SETTINGS["seg_s"] * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, [], N_LAGS, stimulus=stimulus, stimulus_gain=stim_gain)
    return zscore_designs(raw), groups


def run_task(task):
    """One replicated study for (stimulus gain, person variability, group ratio)."""
    rng = np.random.default_rng(task["seed"])
    gain, sd_log, ratio = task["gain"], task["sd_log"], task["ratio"]
    n_samples = int(SETTINGS["seg_s"] * FS)
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator="dDTF", band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    stimulus = make_stimulus(rng, n_samples, FS, STIM_RADIUS, STIM_F0)
    designs, groups = simulate_study(rng, gain, sd_log, ratio, stimulus)
    out = []
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled", "exclude", "within_group"], groups, rng,
                       STABILITY_MAX_ROOT, Z_CRIT, ALPHA, SETTINGS["n_perm"]):
        out.append(dict(r, null_source="same film", gain=gain, sd_log=sd_log, ratio=ratio, rep=task["rep"]))
    # (c) null from another film (different stimulus, same gains structure)
    other_stimulus = make_stimulus(rng, n_samples, FS, STIM_RADIUS, STIM_F0)
    other_designs, _ = simulate_study(rng, gain, sd_log, ratio, other_stimulus)
    real, roots, _ = real_matrices(designs, NODES, est)
    values, _, _ = surrogate_values(other_designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng, None)
    stable = roots < STABILITY_MAX_ROOT
    delta, z = delta_and_z_all(real, values, "pooled", groups)
    for name, edge in EDGES.items():
        s = edge_summary(delta[stable], z[stable], edge, groups[stable], Z_CRIT, ALPHA, rng, SETTINGS["n_perm"])
        out.append(dict(s, mode="pooled", edge=name, null_source="other film", gain=gain, sd_log=sd_log, ratio=ratio, rep=task["rep"]))
    return out


def main():
    """Run S07 and write all outputs to Results/S_07."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID)
    cells = [(g, s, r) for g in SETTINGS["gains"] for s in SETTINGS["sd_log"] for r in SETTINGS["ratios"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"gain": g, "sd_log": s, "ratio": r, "rep": i, "seed": seeds[c * SETTINGS["n_rep"] + i]}
             for c, (g, s, r) in enumerate(cells) for i in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    keys = ["null_source", "mode", "gain", "sd_log", "ratio"]
    summary = df.groupby(keys).agg(study_fpr=("study_reject", "mean"), group_fpr=("group_reject", "mean"), dyad_fpr=("dyad_rate", "mean"),
                                   median_z=("median_z", "median"), n_rep=("rep", "size")).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    base = summary[(summary.sd_log == SETTINGS["sd_log"][-1])]
    for (null_source, mode), sub in base.groupby(["null_source", "mode"]):
        one = sub[sub.ratio == 1.0].sort_values("gain")
        axes[0].plot(one.gain, one.study_fpr, marker="o", label=f"{null_source}, {mode}")
        contrast = sub[sub.ratio == max(SETTINGS["ratios"])].sort_values("gain")
        axes[1].plot(contrast.gain, contrast.group_fpr, marker="o", label=f"{null_source}, {mode}")
    for ax, title in zip(axes, ["one group (signed-rank)", f"group contrast, gain ratio {max(SETTINGS['ratios'])} (permutation)"]):
        ax.axhline(ALPHA, color="k", lw=0.6); ax.set_xlabel("stimulus gain"); ax.set_title(title)
    axes[0].set_ylabel("false-positive rate"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out_dir / "fig_stimulus_false_positives.png", dpi=150); plt.close(fig)

    same = summary[summary.null_source == "same film"]
    top_ratio = max(SETTINGS["ratios"])
    pooled_c = same[(same["mode"] == "pooled") & (same.ratio == top_ratio) & (same.gain > 0)].group_fpr
    within_c = same[(same["mode"] == "within_group") & (same.ratio == top_ratio) & (same.gain > 0)].group_fpr
    other = summary[(summary.null_source == "other film") & (summary.gain >= 1.0)].study_fpr
    criteria = {
        f"one group, same-film pooled null: study-level FPR <= {FPR_MAX} in every cell": bool((same[same["mode"] == "pooled"].study_fpr <= FPR_MAX).all()),
        f"group contrast, within-group null: FPR <= {FPR_MAX} in every cell": bool((same[same["mode"] == "within_group"].group_fpr <= FPR_MAX).all()),
        "group contrast: pooled-null FPR exceeds within-group FPR when gains differ": bool(pooled_c.mean() > within_c.mean()),
        f"other-film null inflates the FPR above {FPR_MAX} when the stimulus matters": bool((other > FPR_MAX).any()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
