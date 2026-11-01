"""S05 - Conditioning on the other channels: indirect paths and common drivers (Table 2, S5).

Nodes: 0 child:ROI, 1 cg:ROI, 2 child:HRV, 3 cg:HRV.
Scenario 'cascade': planted cg:ROI -> child:ROI -> child:HRV. The edge cg:ROI -> child:HRV is absent
  (indirect only).
Scenario 'common_driver': planted child:ROI -> cg:ROI and child:ROI -> cg:HRV. The edge cg:ROI -> cg:HRV is absent
  (shared source only).
Analyses compared: 4-node dDTF, 4-node ffDTF (no partial-coherence factor), 4-node DTF (normalized per frequency),
and bivariate dDTF on the pair that contains the absent edge (the other channels are ignored).
Readouts: study-level detection (signed-rank) of the absent edge (false positive) and of the planted edges (power).

Usage: python S05_conditioning.py [quick|full] [dDTF|GPDC]       Results: Results/<estimator>/S_05/
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
SIM_ID = 5
BASE_SEED = 20260105
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 15, "gains": [0.4], "seg_s": 60},
    "full": {"n_rep": 200, "n_dyads": 25, "gains": [0.2, 0.4, 0.6], "seg_s": 60},
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
CHILD_NODES = [0, 2]
STABILITY_MAX_ROOT = 1.0
Z_CRIT, ALPHA = 1.96, 0.05
FPR_MAX = 0.10                        # accepted false-positive rate of an absent edge [proposed]
POWER_MIN = 0.80                      # [proposed]

# scenario -> (planted edges as (source, target) , absent edge (source, target), bivariate node pair)
SCENARIOS = {
    "cascade": {"planted": [(1, 0), (0, 2)], "absent": (1, 2), "pair": [1, 2]},
    "common_driver": {"planted": [(0, 1), (0, 3)], "absent": (1, 3), "pair": [1, 3]},
}
NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV"]
ANALYSES = {f"4-node {ESTIMATOR}": (ESTIMATOR, None), "4-node ffDTF": ("ffDTF", None), "4-node DTF": ("DTF", None),
            f"bivariate {ESTIMATOR}": (ESTIMATOR, "pair")}


def edge_name(edge):
    """Readable edge label."""
    return f"{NODE_NAMES[edge[0]]}->{NODE_NAMES[edge[1]]}"


def run_task(task):
    """One replicated study for (scenario, gain): all analyses on the same simulated dyads."""
    rng = np.random.default_rng(task["seed"])
    sc = SCENARIOS[task["scenario"]]
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    edges = [(s, t, 1, task["gain"]) for s, t in sc["planted"]]
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], int(SETTINGS["seg_s"] * FS), FS, POLE_RADIUS_RANGES, F0_RANGES, edges, N_LAGS))
    out = []
    for label, (estimator, subset) in ANALYSES.items():
        est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=estimator, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
        nodes = [0, 1, 2, 3] if subset is None else sc["pair"]
        pos = {n: i for i, n in enumerate(nodes)}
        wanted = {"absent": sc["absent"]} | {f"planted {i}": e for i, e in enumerate(sc["planted"])}
        edge_dict = {k: (pos[s], pos[t]) for k, (s, t) in wanted.items() if s in pos and t in pos}
        for r in run_study(designs, nodes, CHILD_NODES, est, edge_dict, ["pooled"], None, rng, STABILITY_MAX_ROOT, Z_CRIT, ALPHA, 0):
            role = "absent" if r["edge"] == "absent" else "planted"
            out.append(dict(r, analysis=label, role=role, edge_label=edge_name(wanted[r["edge"]]),
                            scenario=task["scenario"], gain=task["gain"], rep=task["rep"]))
    return out


def main():
    """Run S05 and write all outputs to Results/<estimator>/S_05."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    cells = [(s, g) for s in SCENARIOS for g in SETTINGS["gains"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"scenario": s, "gain": g, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (s, g) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    keys = ["scenario", "gain", "analysis", "role", "edge_label"]
    summary = df.groupby(keys).agg(study_rate=("study_reject", "mean"), dyad_rate=("dyad_rate", "mean"),
                                   median_z=("median_z", "median"), n_rep=("rep", "size")).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.study_rate * r.n_rep)), int(r.n_rep)), axis=1)
    summary["ci_low"], summary["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary.to_csv(out_dir / "summary.csv", index=False)

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6 * len(SCENARIOS), 4), sharey=True)
    for ax, (scenario, sub) in zip(np.atleast_1d(axes), summary[summary.role == "absent"].groupby("scenario")):
        for analysis, s2 in sub.groupby("analysis"):
            ax.plot(s2.gain, s2.study_rate, marker="o", label=analysis)
        ax.axhline(ALPHA, color="k", lw=0.6); ax.set_title(f"{scenario}: absent edge {sub.edge_label.iloc[0]}")
        ax.set_xlabel("planted gain"); ax.set_ylabel("false-positive rate (study level)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_absent_edge_false_positives.png", dpi=150); plt.close(fig)

    ref_gain = 0.4 if 0.4 in SETTINGS["gains"] else SETTINGS["gains"][0]
    at = summary[summary.gain == ref_gain]
    absent4 = at[(at.role == "absent") & (at.analysis == f"4-node {ESTIMATOR}")].study_rate
    absent_biv = at[(at.role == "absent") & (at.analysis == f"bivariate {ESTIMATOR}")].study_rate
    planted4 = at[(at.role == "planted") & (at.analysis == f"4-node {ESTIMATOR}")].study_rate
    criteria = {
        f"4-node {ESTIMATOR}: absent-edge false-positive rate <= {FPR_MAX} (gain {ref_gain})": bool((absent4 <= FPR_MAX).all()),
        f"4-node {ESTIMATOR}: planted-edge power >= {POWER_MIN} (gain {ref_gain})": bool((planted4 >= POWER_MIN).all()),
        f"bivariate {ESTIMATOR} produces more false positives than 4-node {ESTIMATOR} in every scenario":
            bool((absent_biv.values > absent4.values).all()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)


if __name__ == "__main__":
    main()
