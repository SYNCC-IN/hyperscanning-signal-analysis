"""S12 - Mutual (bidirectional) coupling between partners (Table S1, S12).

Question: when influence runs both ways, are both directions detected, how often is a one-way link mistaken for a mutual
one, and can the dominant direction (or its absence) be read from the data?
Design: four nodes as in S1 (0 child:ROI, 1 cg:ROI, 2 child:HRV, 3 cg:HRV), heterogeneous persons, N = 25, 60 s.
One pair of nodes is coupled in both directions at lag 1 with gains (forward = caregiver -> child, reverse = child ->
caregiver); all other edges are absent.
  - brain-brain (cg:ROI <-> child:ROI): (0.3, 0), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3), (0.2, 0.2), (0.2, 0.1);
    (0.4/0.4 is not used: about 4% of the person draws give an unstable generator)
  - heart-heart (cg:HRV <-> child:HRV): (0.3, 0), (0.3, 0.15), (0.3, 0.3).
Readouts per condition (study level, pairing permutation test):
  - forward and reverse detection; 'mutual detected' = both edges significant in the same study;
  - direction: share of studies in which the median over dyads of (delta_fwd - delta_rev) is positive;
  - asymmetry test: sign-flip permutation test of the mean over dyads of d = delta_fwd - delta_rev (two-sided).
    delta = real value - median of the pseudo-dyad null of that edge, so each edge is corrected for its own baseline
    (signal smoothness, scale); under equal coupling in both directions d is symmetric around 0.
Criteria (fixed before the run of 2026-11-07): see `criteria` in main().

Usage: python S12_mutual_coupling.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_12/
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
from src.sim_inference import delta_and_z_all, real_matrices, score_study, surrogate_values, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 12
BASE_SEED = 20260112
CONDITIONS = {"brain-brain": [(0.3, 0.0), (0.3, 0.1), (0.3, 0.2), (0.3, 0.3), (0.2, 0.2), (0.2, 0.1)],
              "heart-heart": [(0.3, 0.0), (0.3, 0.15), (0.3, 0.3)]}
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 15, "seg_s": 60, "conditions": {"brain-brain": [(0.3, 0.0), (0.3, 0.1), (0.3, 0.3)], "heart-heart": [(0.3, 0.3)]}},
    "full": {"n_rep": 200, "n_dyads": 25, "seg_s": 60, "conditions": CONDITIONS},
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
PAIRS = {"brain-brain": (1, 0), "heart-heart": (3, 2)}          # (caregiver node, child node)
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05
N_PERM = 1000
FPR_MAX, POWER_MIN, DIRECTION_MIN = 0.10, 0.80, 0.95     # [proposed]
CROSS_PERSON = [(1, 0), (0, 1), (3, 2), (2, 3), (3, 0), (2, 1), (1, 2), (0, 3)]   # max-T family: 8 cross-person edges


def sign_flip_pvalue(d, rng, n_perm):
    """Two-sided sign-flip permutation test of mean(d) = 0."""
    d = d[np.isfinite(d)]
    observed = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (1 + n_perm))


def run_task(task):
    """One replicated study for (pair, forward gain, reverse gain)."""
    rng = np.random.default_rng(task["seed"])
    cg, child = PAIRS[task["pair"]]
    edges = [(cg, child, 1, task["g_fwd"])] + ([(child, cg, 1, task["g_rev"])] if task["g_rev"] > 0 else [])
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], int(SETTINGS["seg_s"] * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, edges, N_LAGS))
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    real, roots, _ = real_matrices(designs, NODES, est)
    values, _, n_unstable = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng)
    keep = roots < STABILITY_MAX_ROOT
    real, values = real[keep], values[keep][:, keep]
    edge_dict = {"forward": (cg, child), "reverse": (child, cg)}
    family = {name: e for e in CROSS_PERSON for name in [next((n for n, v in edge_dict.items() if v == e), str(e))]}
    summaries = score_study(real, values, real, "pooled", None, edge_dict, DYAD_ALPHA, ALPHA, rng, N_PERM, family=family,
                            multiplicity="maxT", multiplicity_rng=np.random.default_rng([task["seed"], 1]))
    delta, _ = delta_and_z_all(real, values, "pooled", None)
    d = delta[:, child, cg] - delta[:, cg, child]                  # [target, source]: forward minus reverse
    return {"pair": task["pair"], "g_fwd": task["g_fwd"], "g_rev": task["g_rev"], "rep": task["rep"],
            "fwd_reject": summaries["forward"]["study_reject"], "rev_reject": summaries["reverse"]["study_reject"],
            "fwd_reject_maxT": summaries["forward"]["study_reject_adj_maxT"], "rev_reject_maxT": summaries["reverse"]["study_reject_adj_maxT"],
            "other_reject_maxT": any(summaries[n]["study_reject_adj_maxT"] for n in family if n not in edge_dict),
            "fwd_dyad_rate": summaries["forward"]["dyad_rate"], "rev_dyad_rate": summaries["reverse"]["dyad_rate"],
            "dir_positive": float(np.nanmedian(d) > 0), "asym_p": sign_flip_pvalue(d, rng, N_PERM),
            "mean_d": float(np.nanmean(d)), "n_real_unstable": int((~keep).sum()), "n_pairs_unstable": n_unstable}


def main():
    """Run S12 and write all outputs to Results/<estimator>/S_12."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    sim_progress.begin(f"S12 {ESTIMATOR}", Path(__file__).resolve().parent / "Results", mode=MODE, replicates=SETTINGS["n_rep"])
    cells = [(pair, gf, gr) for pair, conds in SETTINGS["conditions"].items() for gf, gr in conds]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"pair": pair, "g_fwd": gf, "g_rev": gr, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (pair, gf, gr) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame(run_parallel(run_task, tasks, N_JOBS))
    df["mutual"] = df.fwd_reject & df.rev_reject
    df["mutual_maxT"] = df.fwd_reject_maxT & df.rev_reject_maxT
    df["asym_reject"] = df.asym_p < ALPHA
    df.to_csv(out_dir / "raw_results.csv", index=False)

    summary = df.groupby(["pair", "g_fwd", "g_rev"]).agg(
        forward_rate=("fwd_reject", "mean"), reverse_rate=("rev_reject", "mean"), mutual_rate=("mutual", "mean"),
        forward_dominant=("dir_positive", "mean"), asymmetry_rate=("asym_reject", "mean"), mean_d=("mean_d", "mean"),
        fwd_dyad_rate=("fwd_dyad_rate", "mean"), rev_dyad_rate=("rev_dyad_rate", "mean"), n_rep=("rep", "size"),
        forward_rate_maxT=("fwd_reject_maxT", "mean"), reverse_rate_maxT=("rev_reject_maxT", "mean"),
        mutual_rate_maxT=("mutual_maxT", "mean"), other_edges_any_maxT=("other_reject_maxT", "mean")).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.reverse_rate * r.n_rep)), int(r.n_rep)), axis=1)
    summary["reverse_ci_low"], summary["reverse_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, len(SETTINGS["conditions"]), figsize=(6 * len(SETTINGS["conditions"]), 4), sharey=True)
    for ax, (pair, sub) in zip(np.atleast_1d(axes), summary.groupby("pair")):
        labels = [f"{a:g}/{b:g}" for a, b in zip(sub.g_fwd, sub.g_rev)]
        x = np.arange(len(sub))
        for col, lab in [("forward_rate", "forward detected"), ("reverse_rate", "reverse detected"), ("mutual_rate", "both detected"),
                         ("asymmetry_rate", "asymmetry test"), ("forward_dominant", "forward dominant")]:
            ax.plot(x, sub[col], marker="o", label=lab)
        ax.axhline(ALPHA, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("gain forward / reverse"); ax.set_title(f"{pair} ({ESTIMATOR})")
    np.atleast_1d(axes)[0].set_ylabel("proportion of studies"); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_mutual_coupling.png", dpi=150); plt.close(fig)

    def at(pair, gf, gr, col):
        sel = summary[(summary.pair == pair) & np.isclose(summary.g_fwd, gf) & np.isclose(summary.g_rev, gr)][col]
        return float(sel.iloc[0]) if len(sel) else float("nan")

    one_way = summary[summary.g_rev == 0]
    symmetric = summary[np.isclose(summary.g_fwd, summary.g_rev)]
    asymmetric = summary[(summary.g_rev > 0) & (summary.g_fwd > summary.g_rev)]
    criteria = {
        f"one-way links: reverse edge (false mutual coupling) <= {FPR_MAX}": bool((one_way.reverse_rate <= FPR_MAX).all()),
        f"symmetric mutual coupling: both edges detected >= {POWER_MIN} (brain 0.3/0.3, heart 0.3/0.3)":
            bool(min(at("brain-brain", 0.3, 0.3, "mutual_rate"), at("heart-heart", 0.3, 0.3, "mutual_rate")) >= POWER_MIN),
        f"symmetric mutual coupling: asymmetry test false-positive rate <= {FPR_MAX}": bool((symmetric.asymmetry_rate <= FPR_MAX).all()),
        f"asymmetric mutual coupling: forward direction dominant in >= {DIRECTION_MIN} of studies": bool((asymmetric.forward_dominant >= DIRECTION_MIN).all()),
        f"brain 0.3/0.1: asymmetry detected >= {POWER_MIN}": bool(at("brain-brain", 0.3, 0.1, "asymmetry_rate") >= POWER_MIN),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": {k: v for k, v in SETTINGS.items() if k != "conditions"},
                   "conditions": {k: [list(c) for c in v] for k, v in SETTINGS["conditions"].items()}, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)
    sim_progress.finish(out_dir, {"forward rate": sim_progress.range_text(summary.forward_rate.tolist()), "reverse rate": sim_progress.range_text(summary.reverse_rate.tolist())}, criteria)


if __name__ == "__main__":
    main()
