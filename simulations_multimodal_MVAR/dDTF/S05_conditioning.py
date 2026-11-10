"""S05 - Conditioning on the other channels: indirect paths and common drivers (Table S1, S5).

Nodes: 0 child:ROI, 1 cg:ROI, 2 child:HRV, 3 cg:HRV (and 4 = Z, an extra dyad-level driver, in 'omitted_driver').
Scenario 'cascade': planted cg:ROI -> child:ROI (lag 1) -> child:HRV (lag 1). The cross-person edge cg:ROI -> child:HRV
  is absent (indirect only).
Scenario 'common_driver' (cross-person, observed driver): planted cg:ROI -> cg:HRV (lag 1) and cg:ROI -> child:ROI (lag 2).
  The cross-person edge cg:HRV -> child:ROI is absent; both ends share the parent cg:ROI, and the heart receives the drive
  earlier, so a bivariate analysis sees a false cg:HRV -> child:ROI edge. In pseudo-dyads the cg:ROI -> child:ROI link is
  broken, so the artefact exists only in real dyads: the scenario tests conditioning under the pseudo-dyad null.
Scenario 'omitted_driver': a dyad-level process Z (node 4; smooth, breathing-like, not stimulus-locked) drives cg:HRV (lag 1)
  and child:HRV (lag 2). There is no edge between the partners. Analysed (a) without Z (4-node model: Z omitted) and
  (b) with Z as a fifth node. In pseudo-dyads Z goes with the caregiver role. Expected: (a) a dyad-level heart-heart edge
  is detected (a limitation, quantified), (b) it is removed by conditioning on Z.
(The former 'common_driver' scenario with a within-person absent edge was removed: such edges cannot be tested against
pseudo-dyads, so it said nothing about conditioning.)
Analyses compared: 4-node <estimator>, 4-node ffDTF (no partial-coherence factor), 4-node DTF (normalized per frequency),
bivariate <estimator> on the pair that contains the absent edge, and in 'omitted_driver' the 5-node <estimator> with Z.
Readouts: study-level detection (pairing permutation test) of the absent edge(s) and of the planted cross-person edges.
Collinearity diagnostics of the 4-node and 5-node <estimator> fits (median over real dyads and over pseudo-dyads):
correlation of Z with the caregiver heart, condition number of the stacked-window lag regressors, and the largest VIF of
the caregiver-heart lags (`src.sim_estimation.regressor_diagnostics`).
Sweep (omitted driver, 4-node without Z and 5-node with Z, gains 0.4 and 0.6, one factor at a time): segment length
60 / 120 / 300 s at p = 4, and model order p = 2 / 4 / 6 at 60 s; separate seeds (`raw_results_sweep.csv`,
`summary_sweep.csv`, `fig_collinearity_sweep`).
Criteria (fixed before the run of 2026-11-07): see `criteria` in main().

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
from src.sim_estimation import regressor_diagnostics
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src import sim_progress
from src.sim_inference import surrogate_design
from src.sim_inference import run_study, wilson_interval

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
N_JOBS = 4
SIM_ID = 5
BASE_SEED = 20260105
SETTINGS = {
    "quick": {"n_rep": 6, "n_dyads": 15, "gains": [0.4], "seg_s": 60,
              "sweep_gains": [0.6], "sweep_cells": [(60, 4), (120, 4), (60, 2)], "sweep_n_rep": 4},
    "full": {"n_rep": 200, "n_dyads": 25, "gains": [0.2, 0.4, 0.6], "seg_s": 60,
             "sweep_gains": [0.4, 0.6], "sweep_cells": [(60, 4), (120, 4), (300, 4), (60, 2), (60, 6)], "sweep_n_rep": 200},
}[MODE]                                # sweep_cells: (segment s, model order p), one factor varied at a time
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])

FS = 2.5
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92], [0.80, 0.92]])   # node 4 = Z
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
CHILD_NODES = [0, 2]                  # node 4 (Z) is not a child node, so pseudo-dyads take it from the caregiver's dyad
STABILITY_MAX_ROOT = 1.0
DYAD_ALPHA, ALPHA = 0.05, 0.05         # one-sided dyad level (empirical p); study and group level
N_PERM = 1000                         # permutations of the study-level (pairing) and group-level tests
FPR_MAX = 0.10                        # accepted false-positive rate of an absent edge [proposed]
POWER_MIN = 0.80                      # [proposed]
OMITTED_MIN = 0.50                    # omitted driver: the dyad-level edge is expected in >= 50% of studies [expected limitation]

# planted: (source, target, lag); absent: name -> (source, target); pair: nodes of the bivariate analysis;
# power: planted cross-person edges whose detection is reported; k: number of simulated nodes
SCENARIOS = {
    "cascade": {"k": 4, "planted": [(1, 0, 1), (0, 2, 1)], "absent": {"absent": (1, 2)}, "pair": [1, 2], "power": [(1, 0)]},
    "common_driver": {"k": 4, "planted": [(1, 3, 1), (1, 0, 2)], "absent": {"absent": (3, 0)}, "pair": [3, 0], "power": [(1, 0)]},
    "omitted_driver": {"k": 5, "planted": [(4, 3, 1), (4, 2, 2)], "absent": {"absent": (3, 2), "absent reverse": (2, 3)},
                       "pair": [3, 2], "power": []},
}
NODE_NAMES = ["child:ROI", "cg:ROI", "child:HRV", "cg:HRV", "Z"]
CROSS_PERSON = [(1, 0), (0, 1), (3, 2), (2, 3), (3, 0), (2, 1), (1, 2), (0, 3)]   # max-T family of the 4-node analysis
CG_HEART, Z_NODE = 3, 4               # nodes of the collinearity diagnostics
SWEEP_SCENARIO = "omitted_driver"


def analyses(scenario):
    """Analysis label -> (estimator, nodes) for a scenario."""
    out = {f"4-node {ESTIMATOR}": (ESTIMATOR, [0, 1, 2, 3]), "4-node ffDTF": ("ffDTF", [0, 1, 2, 3]),
           "4-node DTF": ("DTF", [0, 1, 2, 3]), f"bivariate {ESTIMATOR}": (ESTIMATOR, SCENARIOS[scenario]["pair"])}
    if scenario == "omitted_driver":
        out[f"5-node {ESTIMATOR} (Z included)"] = (ESTIMATOR, [0, 1, 2, 3, 4])
    return out


def edge_name(edge):
    """Readable edge label."""
    return f"{NODE_NAMES[edge[0]]}->{NODE_NAMES[edge[1]]}"


def collinearity(designs, nodes, p, win_len, step):
    """Median over real dyads and over pseudo-dyads of the regressor diagnostics of one analysis (node set)."""
    n = len(designs)
    sets = {"real": [designs[d][nodes] for d in range(n)],
            "pseudo": [surrogate_design(designs, b, c, nodes, CHILD_NODES) for b in range(n) for c in range(n) if b != c]}
    out = {}
    for kind, items in sets.items():
        diag = [regressor_diagnostics(x, p, win_len, step, list(nodes).index(CG_HEART)) for x in items]
        out[f"cond_{kind}"] = float(np.median([d["condition_number"] for d in diag]))
        out[f"vif_cg_heart_{kind}"] = float(np.median([d["vif_max"] for d in diag]))
        if Z_NODE < designs.shape[1]:          # Z goes with the caregiver role, so this is the same for pseudo-dyads
            corr = [np.corrcoef(designs[d][Z_NODE], designs[d][CG_HEART])[0, 1] for d in range(n)]
            out[f"corr_z_cg_heart_{kind}"] = float(np.median(corr))
    return out


def run_task(task):
    """One replicated study for (scenario, gain): all analyses on the same simulated dyads.

    Sweep tasks (``task["sweep"]``) set their own segment length and model order and run only the 4-node and 5-node
    <estimator> analyses.
    """
    rng = np.random.default_rng(task["seed"])
    sc = SCENARIOS[task["scenario"]]
    k = sc["k"]
    seg_s, p = task.get("seg_s", SETTINGS["seg_s"]), task.get("p", MODEL_ORDER)
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    edges = [(s, t, lag, task["gain"]) for s, t, lag in sc["planted"]]
    designs = zscore_designs(simulate_dyads(rng, SETTINGS["n_dyads"], int(seg_s * FS), FS,
                                            POLE_RADIUS_RANGES[:k], F0_RANGES[:k], edges, N_LAGS))
    wanted = dict(sc["absent"]) | {f"planted {i}": e for i, e in enumerate(sc["power"])}
    out = []
    for label, (estimator, nodes) in analyses(task["scenario"]).items():
        if task.get("sweep") and label not in (f"4-node {ESTIMATOR}", f"5-node {ESTIMATOR} (Z included)"):
            continue
        est = dict(p=p, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=estimator, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
        diag = collinearity(designs, nodes, p, win_len, step) if estimator == ESTIMATOR and len(nodes) > 2 else {}
        pos = {n: i for i, n in enumerate(nodes)}
        edge_dict = {name: (pos[s], pos[t]) for name, (s, t) in wanted.items() if s in pos and t in pos}
        mult = {}
        if label == f"4-node {ESTIMATOR}":           # max-T over the 8 cross-person edges (separate random stream)
            named = {e: n for n, e in wanted.items()}
            mult = dict(family={named.get(e, edge_name(e)): e for e in CROSS_PERSON}, multiplicity="maxT",
                        multiplicity_rng=np.random.default_rng([task["seed"], 1]))
        for r in run_study(designs, nodes, CHILD_NODES, est, edge_dict, ["pooled"], None, rng, STABILITY_MAX_ROOT, DYAD_ALPHA, ALPHA, N_PERM, **mult):
            tag = dict(diag, analysis=label, scenario=task["scenario"], gain=task["gain"], rep=task["rep"], seg_s=seg_s, p=p)
            if r["edge"] not in wanted:
                out.append(dict(r, role="other", edge_label=r["edge"], **tag))
                continue
            role = "planted" if r["edge"].startswith("planted") else r["edge"]
            out.append(dict(r, role=role, edge_label=edge_name(wanted[r["edge"]]), **tag))
    return out


DIAG_COLS = ["corr_z_cg_heart_real", "cond_real", "cond_pseudo", "vif_cg_heart_real", "vif_cg_heart_pseudo"]


def run_sweep(out_dir):
    """Omitted-driver sweep over segment length and model order (separate seeds); writes raw, summary and figure."""
    cells = [(g, seg, p) for g in SETTINGS["sweep_gains"] for seg, p in SETTINGS["sweep_cells"]]
    n_rep = SETTINGS["sweep_n_rep"]
    seeds = spawn_seeds([BASE_SEED, 1], len(cells) * n_rep)
    tasks = [{"sweep": True, "scenario": SWEEP_SCENARIO, "gain": g, "seg_s": seg, "p": p, "rep": r, "seed": seeds[i * n_rep + r]}
             for i, (g, seg, p) in enumerate(cells) for r in range(n_rep)]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results_sweep.csv", index=False)
    scored = df[df.role.isin(["absent", "absent reverse"])]
    summary = scored.groupby(["gain", "seg_s", "p", "analysis", "role", "edge_label"]).agg(
        study_rate=("study_reject", "mean"), n_rep=("rep", "size"), **{c: (c, "median") for c in DIAG_COLS}).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.study_rate * r.n_rep)), int(r.n_rep)), axis=1)
    summary["ci_low"], summary["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary.to_csv(out_dir / "summary_sweep.csv", index=False)

    five = summary[summary.analysis.str.startswith("5-node")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (factor, fixed, label) in enumerate([("seg_s", ("p", MODEL_ORDER), "segment length (s), p = 4"),
                                                   ("p", ("seg_s", 60), "model order p, 60 s")]):
        for (analysis, role, gain), sub in summary[summary[fixed[0]] == fixed[1]].groupby(["analysis", "role", "gain"]):
            sub = sub.sort_values(factor)
            axes[0, col].errorbar(sub[factor], sub.study_rate, yerr=[sub.study_rate - sub.ci_low, sub.ci_high - sub.study_rate],
                                  marker="o", ms=3, capsize=2, ls="-" if analysis.startswith("5-node") else ":",
                                  label=f"{analysis.split(' (')[0]}, {role}, gain {gain}")
        axes[0, col].axhline(ALPHA, color="k", lw=0.6); axes[0, col].axhline(FPR_MAX, color="r", lw=0.6, ls="--")
        axes[0, col].set_xlabel(label); axes[0, col].set_ylabel("study-level rate (heart-heart edge, no true edge)")
        for gain, sub in five[(five[fixed[0]] == fixed[1]) & (five.role == "absent")].groupby("gain"):
            sub = sub.sort_values(factor)
            axes[1, col].plot(sub[factor], sub.vif_cg_heart_real, marker="o", label=f"VIF cg heart, real, gain {gain}")
            axes[1, col].plot(sub[factor], sub.cond_real, marker="s", ls="--", label=f"condition number, real, gain {gain}")
        axes[1, col].set_xlabel(label); axes[1, col].set_ylabel("5-node fit (median over dyads)")
    axes[0, 0].legend(fontsize=6); axes[1, 0].legend(fontsize=6)
    fig.suptitle("Omitted driver: collinearity of Z and the caregiver heart", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_collinearity_sweep.{ext}", dpi=150)
    plt.close(fig)
    return summary


def main():
    """Run S05 and write all outputs to Results/<estimator>/S_05."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    sim_progress.begin(f"S05 {ESTIMATOR}", Path(__file__).resolve().parent / "Results", mode=MODE, replicates=SETTINGS["n_rep"])
    cells = [(s, g) for s in SCENARIOS for g in SETTINGS["gains"]]
    seeds = spawn_seeds(BASE_SEED, len(cells) * SETTINGS["n_rep"])
    tasks = [{"scenario": s, "gain": g, "rep": r, "seed": seeds[i * SETTINGS["n_rep"] + r]}
             for i, (s, g) in enumerate(cells) for r in range(SETTINGS["n_rep"])]
    df = pd.DataFrame([r for recs in run_parallel(run_task, tasks, N_JOBS) for r in recs])
    df.to_csv(out_dir / "raw_results.csv", index=False)

    keys = ["scenario", "gain", "analysis", "role", "edge_label"]
    scored = df[df.role != "other"].copy()
    scored["reject_maxT"] = scored.study_reject_adj_maxT.map({True: 1.0, False: 0.0})
    summary = scored.groupby(keys).agg(study_rate=("study_reject", "mean"), dyad_rate=("dyad_rate", "mean"),
                                       median_z=("median_z", "median"), n_rep=("rep", "size"),
                                       study_rate_maxT=("reject_maxT", "mean")).reset_index()
    ci = summary.apply(lambda r: wilson_interval(int(round(r.study_rate * r.n_rep)), int(r.n_rep)), axis=1)
    summary["ci_low"], summary["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    summary.to_csv(out_dir / "summary.csv", index=False)
    four_node = df[df.analysis == f"4-node {ESTIMATOR}"].copy()
    four_node["reject_maxT"] = four_node.study_reject_adj_maxT.astype(bool)
    four_node.groupby(["scenario", "gain", "role", "edge_label"]).reject_maxT.mean().reset_index().rename(
        columns={"reject_maxT": "study_rate_maxT"}).to_csv(out_dir / "summary_maxT_all_cross_edges.csv", index=False)
    diag_cols = [c for c in df.columns if c.startswith(("corr_z", "cond_", "vif_"))]
    df.dropna(subset=["cond_real"]).groupby(["scenario", "gain", "analysis", "rep"])[diag_cols].first().groupby(
        ["scenario", "gain", "analysis"]).median().reset_index().to_csv(out_dir / "summary_collinearity.csv", index=False)
    sweep = run_sweep(out_dir)

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

    def rate(scenario, analysis, role):
        sel = at[(at.scenario == scenario) & (at.analysis == analysis) & (at.role == role)].study_rate
        return float(sel.iloc[0]) if len(sel) else float("nan")

    four, biv, five = f"4-node {ESTIMATOR}", f"bivariate {ESTIMATOR}", f"5-node {ESTIMATOR} (Z included)"
    criteria = {
        f"cascade: 4-node {ESTIMATOR} absent indirect edge <= {FPR_MAX} (gain {ref_gain})": rate("cascade", four, "absent") <= FPR_MAX,
        f"cascade: 4-node {ESTIMATOR} planted cross-person edge power >= {POWER_MIN} (gain {ref_gain})": rate("cascade", four, "planted") >= POWER_MIN,
        "cascade: bivariate has more false positives than the 4-node model": rate("cascade", biv, "absent") > rate("cascade", four, "absent"),
        f"common driver: 4-node {ESTIMATOR} absent cross-person edge <= {FPR_MAX} (gain {ref_gain})": rate("common_driver", four, "absent") <= FPR_MAX,
        "common driver: bivariate has more false positives than the 4-node model": rate("common_driver", biv, "absent") > rate("common_driver", four, "absent"),
        f"omitted driver: 5-node {ESTIMATOR} with Z, both heart-heart edges <= {FPR_MAX} (gain {ref_gain})":
            max(rate("omitted_driver", five, "absent"), rate("omitted_driver", five, "absent reverse")) <= FPR_MAX,
        f"omitted driver: 4-node {ESTIMATOR} without Z detects the dyad-level edge >= {OMITTED_MIN} (expected limitation)":
            rate("omitted_driver", four, "absent") >= OMITTED_MIN,
    }
    criteria = {k_: bool(v) for k_, v in criteria.items()}
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": SETTINGS, "criteria": criteria,
                   "sweep_5node_absent_rate": sweep[sweep.analysis.str.startswith("5-node") & (sweep.role == "absent")][
                       ["gain", "seg_s", "p", "study_rate", "vif_cg_heart_real", "cond_real"]].round(4).to_dict(orient="records")}, f, indent=2)
    print(summary.round(3).to_string()); print(criteria)
    sim_progress.finish(out_dir, {"absent-edge rate (4-node)": sim_progress.range_text(summary[(summary.role == "absent") & (summary.analysis == four)].study_rate.tolist()), "planted power (4-node)": sim_progress.range_text(summary[(summary.role == "planted") & (summary.analysis == four)].study_rate.tolist())}, criteria)


if __name__ == "__main__":
    main()
