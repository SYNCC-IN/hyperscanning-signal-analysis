"""S08 - Sensitivity to estimator, model order, window geometry, detrending, Box-Cox and band (Table 2, S8).

One-factor-at-a-time variations around the baseline (dDTF, p = 4, 10 s windows with 50% overlap, linear
detrending, Box-Cox 0.25, band 0.15-0.5 Hz). All variants analyse the same simulated studies (paired comparison).
Scenarios: 'null' (no coupling) and 'planted' (cg:ROI->child:ROI, cg:HRV->child:HRV, cg:HRV->child:ROI).
Readouts per variant: false-positive rate (null, mean over the six cross-person edges), power of the planted edges,
rate of detection of the three reverse edges in the planted scenario, direction accuracy, share of unstable fits.

Usage: python S08_sensitivity.py [quick|full]       Results: Results/S_08/
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.design import window_geometry
from src.sim_generators import simulate_dyads, zscore_designs
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import run_study

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
N_JOBS = 4
SIM_ID = 8
BASE_SEED = 20260108
SETTINGS = {
    "quick": {"n_rep": 4, "n_dyads": 12, "seg_s": 60, "variants": ["baseline", "estimator=ffDTF", "order=SC", "window=6s", "band=0.05-0.15"]},
    "full": {"n_rep": 200, "n_dyads": 25, "seg_s": 60, "variants": None},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

FS = 2.5
BASE = {"p": 4, "win_s": 10.0, "overlap": 0.5, "estimator": "dDTF", "detrend_type": "linear", "box_cox_lambda": 0.25,
        "band_hz": (0.15, 0.5), "p_max": 8, "crit_type": "SC"}
VARIANTS = {"baseline": {},
            "estimator=DTF": {"estimator": "DTF"}, "estimator=ffDTF": {"estimator": "ffDTF"}, "estimator=GPDC": {"estimator": "GPDC"},
            "order=2": {"p": 2}, "order=3": {"p": 3}, "order=6": {"p": 6}, "order=8": {"p": 8},
            "order=SC": {"p": None, "crit_type": "SC"}, "order=HQ": {"p": None, "crit_type": "HQ"}, "order=AIC": {"p": None, "crit_type": "AIC"},
            "window=6s": {"win_s": 6.0}, "window=15s": {"win_s": 15.0}, "window=20s": {"win_s": 20.0},
            "overlap=0.25": {"overlap": 0.25}, "overlap=0.75": {"overlap": 0.75}, "window=whole segment": {"win_s": None},
            "detrend=constant": {"detrend_type": "constant"}, "box-cox=none": {"box_cox_lambda": -1.0},
            "band=0.05-0.15": {"band_hz": (0.05, 0.15)}, "band=0.1-0.3": {"band_hz": (0.1, 0.3)}, "band=0.3-0.5": {"band_hz": (0.3, 0.5)}}
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
PLANTED = [(1, 0, 1, 0.4), (3, 2, 1, 0.3), (3, 0, 2, 0.3)]
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
         "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1)}
PLANTED_NAMES = ["cg:ROI->child:ROI", "cg:HRV->child:HRV", "cg:HRV->child:ROI"]
REVERSE_NAMES = ["child:ROI->cg:ROI", "child:HRV->cg:HRV", "child:HRV->cg:ROI"]
STABILITY_MAX_ROOT = 1.0
Z_CRIT, ALPHA = 1.96, 0.05
FPR_MAX, POWER_MIN, SHARE_OK = 0.10, 0.80, 0.80      # [proposed]


def variant_est(overrides):
    """Keyword arguments of `edge_matrix` for a variant."""
    cfg = BASE | overrides
    if cfg["win_s"] is None:
        win_len = step = None
    else:
        win_len, step = window_geometry(cfg["win_s"], cfg["overlap"], FS)
    return dict(p=cfg["p"], win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=cfg["estimator"], band_hz=cfg["band_hz"],
                box_cox_lambda=cfg["box_cox_lambda"], detrend_type=cfg["detrend_type"], p_max=cfg["p_max"], crit_type=cfg["crit_type"])


def active_variants():
    """Variants to run in this mode."""
    names = SETTINGS["variants"] or list(VARIANTS)
    return {k: VARIANTS[k] for k in names}


def run_task(task):
    """One replicated study (null or planted); every variant is analysed on the same designs."""
    rng = np.random.default_rng(task["seed"])
    edges = PLANTED if task["scenario"] == "planted" else []
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], int(SETTINGS["seg_s"] * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, edges, N_LAGS))
    out = []
    for name, overrides in active_variants().items():
        for r in run_study(designs, NODES, CHILD_NODES, variant_est(overrides), EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, Z_CRIT, ALPHA, 0):
            out.append(dict(r, variant=name, scenario=task["scenario"], rep=task["rep"]))
    return out


def main():
    """Run S08 and write all outputs to Results/S_08."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID)
    seeds = spawn_seeds(BASE_SEED, 2 * SETTINGS["n_rep"])
    tasks = [{"scenario": sc, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]} for i, sc in enumerate(["null", "planted"]) for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    rows = []
    for name in active_variants():
        null = df[(df.variant == name) & (df.scenario == "null")]
        pl = df[(df.variant == name) & (df.scenario == "planted")]
        forward, reverse = pl[pl.edge.isin(PLANTED_NAMES)], pl[pl.edge.isin(REVERSE_NAMES)]
        rows.append({"variant": name, "null_fpr": null.study_reject.mean(), "power_planted": forward.study_reject.mean(),
                     "reverse_detection": reverse.study_reject.mean(), "direction_accuracy": forward.dir_study.mean(),
                     "median_z_planted": forward.median_z.median(), "median_z_null": null.median_z.median(),
                     "unstable_real_share": (pl.n_real_unstable.mean() / SETTINGS["n_dyads"]), "unstable_pairs": pl.n_pairs_unstable.mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.28 * len(summary))), sharey=True)
    y = np.arange(len(summary))
    for ax, column, title in zip(axes, ["null_fpr", "power_planted", "direction_accuracy"], ["false-positive rate (null)", "power (planted edges)", "direction accuracy"]):
        ax.barh(y, summary[column]); ax.set_title(title); ax.invert_yaxis()
    axes[0].set_yticks(y); axes[0].set_yticklabels(summary.variant, fontsize=7)
    axes[0].axvline(ALPHA, color="k", lw=0.6); axes[1].axvline(POWER_MIN, color="k", lw=0.6, ls=":")
    fig.tight_layout(); fig.savefig(out_dir / "fig_variants.png", dpi=150); plt.close(fig)

    base = summary[summary.variant == "baseline"].iloc[0]
    ok = (summary.null_fpr <= FPR_MAX)
    criteria = {
        f"baseline: null FPR <= {FPR_MAX}": bool(base.null_fpr <= FPR_MAX),
        f"baseline: power of planted edges >= {POWER_MIN}": bool(base.power_planted >= POWER_MIN),
        f">= {SHARE_OK:.0%} of variants keep the null FPR <= {FPR_MAX}": bool(ok.mean() >= SHARE_OK),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "share_variants_fpr_ok": float(ok.mean()),
                   "variants_failing_fpr": summary[~ok].variant.tolist(), "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
