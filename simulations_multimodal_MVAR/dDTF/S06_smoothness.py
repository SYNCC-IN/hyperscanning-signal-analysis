"""S06 - Smoothness artefact: does a smooth channel masquerade as a source, and does the surrogate correction remove it? (Table 2, S6)

No coupling at all. The caregiver IBI-like node (cg:HRV, node 3) is made smoother and smoother (pole radius grid)
while the child ROI node (node 0) stays rough. Cross-person edges of interest: cg:HRV -> child:ROI (smooth to rough)
and child:ROI -> cg:HRV (rough to smooth).
Factors: smoothness of cg:HRV, z-scoring of designs (on/off; without z-scoring dDTF is not scale invariant).
Readouts: uncorrected asymmetry (real dDTF smooth->rough minus rough->smooth, Box-Cox scale), corrected asymmetry
(median delta difference), study-level false-positive rate of both edges after the surrogate correction.

Usage: python S06_smoothness.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_06/
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
SIM_ID = 6
BASE_SEED = 20260106
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 15, "radii": [0.5, 0.9], "seg_s": 60},
    "full": {"n_rep": 200, "n_dyads": 25, "radii": [0.5, 0.65, 0.8, 0.9, 0.95], "seg_s": 60},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

FS = 2.5
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
EDGES = {"smooth->rough (cg:HRV->child:ROI)": (3, 0), "rough->smooth (child:ROI->cg:HRV)": (0, 3)}
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
FPR_MAX = 0.10                        # [proposed]


def radius_ranges(r_smooth):
    """Pole-radius ranges: rough ROI nodes, rough child HRV, smooth caregiver HRV (node 3)."""
    return np.array([[0.45, 0.55], [0.45, 0.55], [0.45, 0.55], [max(r_smooth - 0.03, 0.4), min(r_smooth + 0.03, 0.97)]])


def run_task(task):
    """One replicated null study for (smoothness, z-scoring)."""
    rng = np.random.default_rng(task["seed"])
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    raw = simulate_dyads(rng, SETTINGS["n_dyads"], int(SETTINGS["seg_s"] * FS), FS, radius_ranges(task["radius"]), F0_RANGES, [], N_LAGS)
    designs = zscore_designs(raw) if task["zscore"] else raw
    out = []
    for r in run_study(designs, NODES, CHILD_NODES, est, EDGES, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM):
        out.append(dict(r, radius=task["radius"], zscore=task["zscore"], rep=task["rep"]))
    return out


def main():
    """Run S06 and write all outputs to Results/<estimator>/S_06."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [(r, z) for r in SETTINGS["radii"] for z in (True, False)]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"radius": r, "zscore": z, "rep": i, "seed": seeds[c * SETTINGS["n_rep"] + i]}
             for c, (r, z) in enumerate(cells) for i in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    fwd, rev = list(EDGES)
    rows = []
    for (radius, z), sub in df.groupby(["radius", "zscore"]):
        a = sub[sub.edge == fwd].set_index("rep"); b = sub[sub.edge == rev].set_index("rep")
        n = len(a)
        rows.append({"radius": radius, "zscore": z,
                     "uncorrected_asymmetry": float((a.real_mean - b.real_mean).mean()),
                     "uncorrected_asymmetry_sd": float((a.real_mean - b.real_mean).std()),
                     "corrected_asymmetry": float((a.median_delta - b.median_delta).mean()),
                     "fpr_smooth_to_rough": float(a.study_reject.mean()), "fpr_rough_to_smooth": float(b.study_reject.mean()),
                     "fpr_ci_high": max(wilson_interval(int(a.study_reject.sum()), n)[1], wilson_interval(int(b.study_reject.sum()), n)[1]),
                     "median_z_smooth_to_rough": float(a.median_z.median()), "n_rep": n})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for z, sub in summary.groupby("zscore"):
        axes[0].plot(sub.radius, sub.uncorrected_asymmetry, marker="o", label=f"uncorrected, z-score={z}")
        axes[0].plot(sub.radius, sub.corrected_asymmetry, marker="s", ls="--", label=f"surrogate-corrected, z-score={z}")
        axes[1].plot(sub.radius, sub[["fpr_smooth_to_rough", "fpr_rough_to_smooth"]].max(axis=1), marker="o", label=f"worst direction, z-score={z}")
    axes[0].axhline(0, color="k", lw=0.6); axes[0].set_xlabel("pole radius of cg:HRV (smoothness)"); axes[0].set_ylabel(f"{ESTIMATOR} asymmetry (smooth->rough minus reverse)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(ALPHA, color="k", lw=0.6); axes[1].set_xlabel("pole radius of cg:HRV (smoothness)"); axes[1].set_ylabel("false-positive rate after correction")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_smoothness_artefact.png", dpi=150); plt.close(fig)

    zs = summary[summary.zscore]
    smooth, rough = zs.sort_values("radius").iloc[-1], zs.sort_values("radius").iloc[0]
    criteria = {
        "artefact present: uncorrected asymmetry > 0 at the smoothest level (z-scored)": bool(smooth.uncorrected_asymmetry > 0),
        "artefact grows with smoothness (smoothest > roughest, z-scored)": bool(smooth.uncorrected_asymmetry > rough.uncorrected_asymmetry),
        "|corrected asymmetry| < |uncorrected| at the smoothest level (z-scored)": bool(abs(smooth.corrected_asymmetry) < abs(smooth.uncorrected_asymmetry)),
        f"false-positive rate after correction <= {FPR_MAX} in every cell (z-scored)": bool((zs[["fpr_smooth_to_rough", "fpr_rough_to_smooth"]].max(axis=1) <= FPR_MAX).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
