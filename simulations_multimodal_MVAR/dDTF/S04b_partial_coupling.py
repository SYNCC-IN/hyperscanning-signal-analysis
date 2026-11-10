"""S04b - Power when only some dyads, or only part of the recording, are coupled (variant of S4, design A).

Design A of S4 (one group, planted edge cg:ROI -> child:ROI, lag 1), N = 25, 60 s, gains 0.3 and 0.4:
  partial_dyads: a fraction q = 1, 0.75, 0.5, 0.25 of the dyads (chosen at random per study) is coupled, the others
                 have gain 0;
  partial_time:  every dyad is coupled during one contiguous episode covering 100 / 75 / 50 / 25 % of the 60 s, at a
                 random start per dyad; the coupling coefficient switches inside one continuous simulation
                 (`src.sim_generators.simulate_dyads_partial`), no concatenated segments.
The studies use the seeds of the matching S4 design-A cells, and the random choices of this variant (which dyads,
episode starts) come from a separate stream, so q = 1 and 100 % reproduce S4 design A exactly (checked against
Results/<estimator>/S_04/raw_results.csv when it exists).
Readouts: planted-edge power per edge (pairing permutation) and after max-T over the 8 cross-person edges, reverse-edge
rate, median delta of the planted edge, and for partial_dyads the dyad-level detection rate (empirical p < .05) in
coupled and in uncoupled dyads.

Usage: python S04b_partial_coupling.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_04/partial/
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
from src.sim_generators import coupling_window, simulate_dyads_partial, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import dyad_pvalues, real_matrices, score_study, surrogate_values, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 4
SETTINGS = {
    "quick": {"n_rep": 4, "n_dyads": 10, "seg_s": 60, "gains": [0.3], "levels": [1.0, 0.5]},
    "full": {"n_rep": 200, "n_dyads": 25, "seg_s": 60, "gains": [0.3, 0.4], "levels": [1.0, 0.75, 0.5, 0.25]},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
VARIANTS = ["partial_dyads", "partial_time"]
# Seeds of S4 (S04_power.py): base seed, its cell list (N, segment, gain) and replicates per cell, per mode
S04_BASE_SEED = 20260104
S04_SETTINGS = {"quick": {"n_rep": 6, "n_dyads": [10], "seg_s": [60], "gains": [0.0, 0.3, 0.6]},
                "full": {"n_rep": 200, "n_dyads": [10, 25, 50], "seg_s": [60, 120, 300], "gains": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}}[MODE]

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
TARGET_EDGE, REVERSE_EDGE = "cg:ROI->child:ROI", "child:ROI->cg:ROI"
EDGES = {TARGET_EDGE: (1, 0), REVERSE_EDGE: (0, 1)}
FAMILY = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
          "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1),
          "cg:ROI->child:HRV": (1, 2), "child:ROI->cg:HRV": (0, 3)}
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05
N_PERM = 1000
POWER_TARGET = 0.80
UNCOUPLED_DYAD_FPR_MAX = 0.10          # dyad-level false-positive rate of uncoupled dyads [proposed]


def s04_seed(n_dyads, seg_s, gain, rep):
    """Seed of the S4 study (N, segment, gain, replicate), as built in S04_power.main."""
    cells = [(n, s, g) for n in S04_SETTINGS["n_dyads"] for s in S04_SETTINGS["seg_s"] for g in S04_SETTINGS["gains"]]
    seeds = spawn_seeds(S04_BASE_SEED, len(cells) * S04_SETTINGS["n_rep"])
    return seeds[cells.index((n_dyads, seg_s, gain)) * S04_SETTINGS["n_rep"] + rep]


def run_task(task):
    """One study: S4 design A with a share of the dyads (partial_dyads) or of the time (partial_time) coupled."""
    rng = np.random.default_rng(task["seed"])
    variant_rng = np.random.default_rng([task["seed"], 2])      # which dyads / episode starts: separate stream
    n, n_samples, level = SETTINGS["n_dyads"], int(SETTINGS["seg_s"] * FS), task["level"]
    coupled, on_masks = np.ones(n, bool), None
    if task["variant"] == "partial_dyads" and level < 1:
        coupled = np.zeros(n, bool)
        coupled[variant_rng.choice(n, int(round(level * n)), replace=False)] = True
    if task["variant"] == "partial_time" and level < 1:
        on_masks = np.stack([coupling_window(variant_rng, n_samples, level) for _ in range(n)])
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    designs = zscore_designs(simulate_dyads_partial(rng, n, n_samples, FS, POLE_RADIUS_RANGES, F0_RANGES,
                                                    [(1, 0, 1, task["gain"])], N_LAGS, coupled, on_masks))
    # as src.sim_inference.run_study (same draws), plus the dyad-level p-values split by coupled / uncoupled dyads
    real, roots, _ = real_matrices(designs, NODES, est)
    values, _, _ = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng)
    keep = roots < STABILITY_MAX_ROOT
    real, values, coupled_kept = real[keep], values[keep][:, keep], coupled[keep]
    summaries = score_study(real, values, real, "pooled", None, EDGES, DYAD_ALPHA, ALPHA, rng, N_PERM, family=FAMILY,
                            multiplicity="maxT", multiplicity_rng=np.random.default_rng([task["seed"], 1]))
    p_target = dyad_pvalues(real, values, "pooled")[:, 0, 1]
    tag = {k: task[k] for k in ("variant", "level", "gain", "rep")}
    out = []
    for name in EDGES:
        row = dict(summaries[name], edge=name, **tag)
        if name == TARGET_EDGE:
            for label, mask in (("coupled", coupled_kept), ("uncoupled", ~coupled_kept)):
                row[f"dyad_rate_{label}"] = float(np.mean(p_target[mask] < DYAD_ALPHA)) if mask.any() else np.nan
                row[f"n_{label}"] = int(mask.sum())
        out.append(row)
    return out


def summarize(df):
    """Power, reverse-edge rate, median delta and dyad-level rates per variant, level and gain (Wilson 95% intervals)."""
    rows = []
    for (variant, level, gain), sub in df.groupby(["variant", "level", "gain"]):
        fwd, rev = sub[sub.edge == TARGET_EDGE], sub[sub.edge == REVERSE_EDGE]
        row = {"variant": variant, "level": level, "gain": gain, "n_rep": len(fwd)}
        for label, flags in (("power", fwd.study_reject), ("power_maxT", fwd.study_reject_adj_maxT),
                             ("reverse_rate", rev.study_reject), ("reverse_rate_maxT", rev.study_reject_adj_maxT)):
            k = int(flags.astype(bool).sum())
            row[label] = k / len(flags)
            row[f"{label}_ci_low"], row[f"{label}_ci_high"] = wilson_interval(k, len(flags))
        row["median_delta"] = float(fwd.median_delta.median())
        for label in ("coupled", "uncoupled"):
            hits = (fwd[f"dyad_rate_{label}"] * fwd[f"n_{label}"]).sum()
            total = int(fwd[f"n_{label}"].sum())
            row[f"dyad_rate_{label}"] = hits / total if total else np.nan
            if total:
                row[f"dyad_rate_{label}_ci_low"], row[f"dyad_rate_{label}_ci_high"] = wilson_interval(int(round(hits)), total)
        rows.append(row)
    return pd.DataFrame(rows)


def reproduces_design_a(df, s04_raw):
    """True if every level-1 study gives the same planted-edge p-values (per edge and max-T) as S4 design A."""
    ref = s04_raw[(s04_raw.design == "A") & (s04_raw.n_dyads == SETTINGS["n_dyads"]) & (s04_raw.seg_s == SETTINGS["seg_s"])]
    ours = df[df.level == 1.0]
    merged = ours.merge(ref, on=["gain", "rep", "edge"], suffixes=("", "_s04"))
    if len(merged) != len(ours) or merged.empty:
        return None
    return bool(np.allclose(merged.study_p, merged.study_p_s04) and np.allclose(merged.study_p_adj_maxT, merged.study_p_adj_maxT_s04))


def plot(summary, out_dir):
    """fig_partial_coupling: power (per edge and max-T) and reverse rate vs coupled share; dyad-level rates (partial_dyads)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, variant, xlabel in ((axes[0], "partial_dyads", "share of dyads coupled (q)"), (axes[1], "partial_time", "share of the 60 s coupled")):
        for i, (gain, sub) in enumerate(summary[summary.variant == variant].groupby("gain")):
            sub = sub.sort_values("level")
            ax.errorbar(sub.level, sub.power, yerr=[sub.power - sub.power_ci_low, sub.power_ci_high - sub.power], marker="o",
                        capsize=2, color=f"C{i}", label=f"gain {gain}, per edge")
            ax.plot(sub.level, sub.power_maxT, marker="s", ls="--", color=f"C{i}", label=f"gain {gain}, max-T")
            ax.plot(sub.level, sub.reverse_rate, marker="x", ls=":", color=f"C{i}", label=f"gain {gain}, reverse edge")
        ax.axhline(POWER_TARGET, color="k", lw=0.6, ls=":"); ax.axhline(ALPHA, color="grey", lw=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("study-level rate"); ax.set_ylim(0, 1.02); ax.set_title(variant, fontsize=10)
    axes[0].legend(fontsize=7)
    for i, (gain, sub) in enumerate(summary[summary.variant == "partial_dyads"].groupby("gain")):
        sub = sub.sort_values("level")
        axes[2].plot(sub.level, sub.dyad_rate_coupled, marker="o", color=f"C{i}", label=f"gain {gain}, coupled dyads")
        axes[2].plot(sub.level, sub.dyad_rate_uncoupled, marker="o", ls="--", color=f"C{i}", label=f"gain {gain}, uncoupled dyads")
    axes[2].axhline(DYAD_ALPHA, color="grey", lw=0.6); axes[2].axhline(UNCOUPLED_DYAD_FPR_MAX, color="r", lw=0.6, ls="--")
    axes[2].set_xlabel("share of dyads coupled (q)"); axes[2].set_ylabel(f"dyad-level rate (p < {DYAD_ALPHA})")
    axes[2].set_title("partial_dyads: dyad level", fontsize=10); axes[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_partial_coupling.{ext}", dpi=150)
    plt.close(fig)


def main():
    """Run S04b and write all outputs to Results/<estimator>/S_04/partial."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR) / "partial"
    out_dir.mkdir(exist_ok=True)
    tasks = [{"variant": v, "level": q, "gain": g, "rep": r, "seed": s04_seed(SETTINGS["n_dyads"], SETTINGS["seg_s"], g, r)}
             for v in VARIANTS for q in SETTINGS["levels"] for g in SETTINGS["gains"] for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot(summary, out_dir)

    s04_raw = out_dir.parent / "raw_results.csv"
    reproduced = reproduces_design_a(df, pd.read_csv(s04_raw)) if s04_raw.exists() else None
    uncoupled = summary[(summary.variant == "partial_dyads") & (summary.level < 1)]
    criteria = {
        f"[proposed] partial_dyads: dyad-level rate of uncoupled dyads <= {UNCOUPLED_DYAD_FPR_MAX} in every cell":
            bool((uncoupled.dyad_rate_uncoupled <= UNCOUPLED_DYAD_FPR_MAX).all()),
        "[proposed] q = 1 (and 100 % of the time) reproduces S4 design A (same p-values on the same seeds)": bool(reproduced),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "reproduces_s04_design_a": reproduced, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
