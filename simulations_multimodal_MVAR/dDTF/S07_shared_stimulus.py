"""S07 - Shared stimulus, group-dependent stimulus gain, pooled vs within-group null, film matching (Table 2, S7).

No true cross-person coupling. All dyads watch the same film: a common stimulus drives every node with a
person-specific gain (log-normal variability). Two groups: group 1 has its stimulus gain multiplied by `ratio`.
Questions:
 (a) one group: does the film-matched surrogate test keep the false-positive rate at nominal level when the stimulus
     induces synchrony in real and surrogate dyads alike? (modes: pooled, exclude)
 (b) group contrast: if the groups differ only in stimulus gain, does the pooled null produce a spurious group
     difference, and does the within-group null remove it? (modes: pooled, within_group)
 (c) film matching: what happens when the null is built from the wrong film? Pseudo-dyads pair a child of the scored
     film with a caregiver who watched another film (independent stimulus), so they lack the stimulus-locked synchrony
     that real dyads share.
Readouts: study-level and group-contrast false-positive rates of the cross-person edges.

Usage: python S07_shared_stimulus.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_07/
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
from src.sim_inference import cross_film_values, real_matrices, run_study, score_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 7
BASE_SEED = 20260107
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 16, "gains": [0.0, 1.0], "sd_log": [0.5], "ratios": [1.0, 2.0], "seg_s": 60},
    "full": {"n_rep": 200, "n_dyads": 30, "gains": [0.0, 0.5, 1.0, 2.0], "sd_log": [0.0, 0.5], "ratios": [1.0, 2.0, 3.0], "seg_s": 60},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

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
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2), "child:HRV->cg:HRV": (2, 3)}
STIM_RADIUS, STIM_F0 = 0.7, 0.3
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
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
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    stimulus = make_stimulus(rng, n_samples, FS, STIM_RADIUS, STIM_F0)
    designs, groups = simulate_study(rng, gain, sd_log, ratio, stimulus)
    out = []
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled", "exclude", "within_group"], groups, rng,
                       STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
        out.append(dict(r, null_source="same film", gain=gain, sd_log=sd_log, ratio=ratio, rep=task["rep"]))
    # (c) null from the wrong film: child of this film x caregiver of another film (independent stimulus, same gain structure)
    other_stimulus = make_stimulus(rng, n_samples, FS, STIM_RADIUS, STIM_F0)
    other_designs, _ = simulate_study(rng, gain, sd_log, ratio, other_stimulus)
    real, roots, _ = real_matrices(designs, NODES, est)
    values = cross_film_values(designs, other_designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT)
    stable = roots < STABILITY_MAX_ROOT
    summaries = score_study(real[stable], values[stable][:, stable], real[stable], "pooled", groups[stable], EDGES,
                            DYAD_ALPHA, ALPHA, rng, N_PERM)
    for name, s in summaries.items():
        out.append(dict(s, mode="pooled", edge=name, null_source="other film", gain=gain, sd_log=sd_log, ratio=ratio, rep=task["rep"]))
    return out


def main():
    """Run S07 and write all outputs to Results/<estimator>/S_07."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    base = summary[(summary.sd_log == SETTINGS["sd_log"][-1])]
    panels = [(axes[0], base[(base.null_source == "same film") & (base.ratio == 1.0)], "study_fpr", "A  one group, same-film null (pairing permutation)"),
              (axes[1], base[(base.null_source == "other film") & (base.ratio == 1.0)], "study_fpr", "B  one group, other-film null (pairing permutation)"),
              (axes[2], base[(base.null_source == "same film") & (base.ratio == max(SETTINGS["ratios"]))], "group_fpr",
               f"C  group contrast, gain ratio {max(SETTINGS['ratios'])} (permutation)")]
    for ax, sub, column, title in panels:
        for mode, s2 in sub.groupby("mode"):
            s2 = s2.sort_values("gain")
            ci = np.array([wilson_interval(int(round(v * n)), int(n)) for v, n in zip(s2[column], s2.n_rep)])
            ax.errorbar(s2.gain, s2[column], yerr=[s2[column] - ci[:, 0], ci[:, 1] - s2[column]], marker="o", capsize=2, label=f"{mode} null")
        ax.axhline(ALPHA, color="k", lw=0.6); ax.axhline(FPR_MAX, color="r", lw=0.6, ls="--")
        ax.set_xlabel("stimulus gain"); ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    axes[0].set_ylabel(f"false-positive rate (sd_log {SETTINGS['sd_log'][-1]})")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_stimulus_false_positives.{ext}", dpi=150)
    plt.close(fig)

    same = summary[summary.null_source == "same film"]
    top_ratio = max(SETTINGS["ratios"])
    pooled_c = same[(same["mode"] == "pooled") & (same.ratio == top_ratio) & (same.gain > 0)].group_fpr
    within_c = same[(same["mode"] == "within_group") & (same.ratio == top_ratio) & (same.gain > 0)].group_fpr
    other = summary[(summary.null_source == "other film") & (summary.ratio == 1.0) & (summary.gain >= 1.0)].study_fpr
    criteria = {
        f"one group, same-film pooled null: study-level FPR <= {FPR_MAX} in every cell": bool((same[(same["mode"] == "pooled") & (same.ratio == 1.0)].study_fpr <= FPR_MAX).all()),
        f"group contrast, within-group null: FPR <= {FPR_MAX} in every cell": bool((same[same["mode"] == "within_group"].group_fpr <= FPR_MAX).all()),
        "group contrast: pooled-null FPR exceeds within-group FPR when gains differ": bool(pooled_c.mean() > within_c.mean()),
        f"one group, other-film null: study-level FPR above {FPR_MAX} when the stimulus matters (gain >= 1)": bool((other > FPR_MAX).any()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
