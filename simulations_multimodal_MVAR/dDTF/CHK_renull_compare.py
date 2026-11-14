"""Fixed-delta group contrast (before 2026-11-14) vs renull=True (new default): S3, S4 design B, S7 and S13.

Reads Results/<estimator>/S_NN_pre_renull/ (no per-window detrending, renull off) and Results/<estimator>/S_NN/ (the
rerun with renull on, same seeds). Only tests against the within-group null can change; every other cell is checked to
be identical. All rates come with 95% Wilson intervals; rates pooled over edges use n = studies x edges (intervals
somewhat too narrow).
Outputs (simulations_multimodal_MVAR/checks_2026-11/renull/):
  S03_within_group.csv   group contrast, within-group null: per-edge rate and max-T family-wise rate per cell;
  S04_design_B.csv       design B power (uncorrected, max-T) per N, segment and gain (gain 0 = false-positive rate),
                         and the 5 vs 5 cells;
  S07_within_group.csv   group-contrast false-positive rate with the within-group null per cell;
  S13.csv                scenario a family-wise rates (group contrast per film x study, over 24, on the film mean,
                         interaction) and power on planted edges (scenarios c-f), delta DV;
  criteria_changes.csv   pass/fail criteria whose outcome changed;
  unchanged_check.csv    largest difference in the cells that renull must not change (other nulls, study level).

Usage: python CHK_renull_compare.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim_inference import wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS = Path(__file__).resolve().parent / "Results"
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "renull"
ESTIMATORS = ["dDTF", "GPDC"]
VERSIONS = {"fixed delta (old)": "_pre_renull", "renull (new)": ""}
S3_N_EDGES = 8
S13_PLANTED = {"c": "group", "d": "group", "e": "interaction", "f": "film"}   # scenario -> effect tested on planted edges
S13_METHODS = ["perm_film_maxT", "perm_24_maxT", "perm_mean_maxT", "perm_interaction_maxT", "perm_contrast_maxT"]
SIMS = {3: "S_03", 4: "S_04", 7: "S_07", 13: "S_13"}


def folder(estimator, sim, suffix):
    """Results folder of one simulation and version."""
    return RESULTS / estimator / f"{SIMS[sim]}{suffix}"


def with_ci(k, n):
    """'rate (low-high)' from successes and trials, 95% Wilson interval."""
    low, high = wilson_interval(int(round(k)), int(n))
    return f"{k / n:.3f} ({low:.3f}-{high:.3f})"


def pivot(rows, keys):
    """One column per version."""
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index=keys, columns="version", values="value", aggfunc="first").reset_index()[keys + list(VERSIONS)]


def s3():
    """Group contrast with the within-group null: per-edge and max-T family-wise rates per cell."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            fwer = pd.read_csv(folder(est, 3, suffix) / "summary_fwer.csv")
            sub = fwer[(fwer["mode"] == "within_group") & (fwer.level == "group")]
            for r in sub[sub.method == "none"].itertuples():
                rows.append({"estimator": est, "measure": "per-edge", "n_dyads": r.n_dyads, "seg_s": r.seg_s, "version": version,
                             "value": with_ci(r.per_edge_rate * r.n_rep * S3_N_EDGES, r.n_rep * S3_N_EDGES)})
            for r in sub[sub.method == "maxT"].itertuples():
                rows.append({"estimator": est, "measure": "FWER max-T", "n_dyads": r.n_dyads, "seg_s": r.seg_s, "version": version,
                             "value": with_ci(r.fwer * r.n_rep, r.n_rep)})
    return pivot(rows, ["estimator", "measure", "n_dyads", "seg_s"])


def s4():
    """Design B (group contrast, within-group null): power per cell, uncorrected and after max-T; the 5 vs 5 cells."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            raw = pd.read_csv(folder(est, 4, suffix) / "raw_results.csv", usecols=["design", "n_dyads", "seg_s", "gain", "rep", "edge",
                                                                               "group_reject", "group_reject_adj_maxT"], low_memory=False)
            planted = raw[(raw.design == "B") & (raw.edge == "cg:ROI->child:ROI")]
            for (n, seg, gain), sub in planted.groupby(["n_dyads", "seg_s", "gain"]):
                for measure, column in (("uncorrected", "group_reject"), ("max-T", "group_reject_adj_maxT")):
                    hits = sub[column].astype(bool)
                    rows.append({"estimator": est, "measure": measure, "n_dyads": n, "seg_s": seg, "gain": gain, "n": len(hits),
                                 "version": version, "value": with_ci(hits.sum(), len(hits))})
    return pivot(rows, ["estimator", "measure", "n_dyads", "seg_s", "gain", "n"])


def s7():
    """Group-contrast false-positive rate with the within-group null, per cell."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            summary = pd.read_csv(folder(est, 7, suffix) / "summary.csv")
            for r in summary[summary["mode"] == "within_group"].itertuples():
                rows.append({"estimator": est, "null_source": r.null_source, "gain": r.gain, "sd_log": r.sd_log, "ratio": r.ratio,
                             "version": version, "value": with_ci(r.group_fpr * r.n_rep, r.n_rep)})
    return pivot(rows, ["estimator", "null_source", "gain", "sd_log", "ratio"])


def s13_raw(est, suffix):
    """All S13 permutation records of the delta DV."""
    files = glob.glob(str(folder(est, 13, suffix) / "raw_results_full_reps_*.csv"))
    raw = pd.concat([pd.read_csv(f, usecols=["scenario", "rep", "method", "dv", "film", "edge", "effect", "detected"]) for f in files])
    return raw[(raw.dv == "delta") & raw.method.isin(S13_METHODS)]


def s13():
    """Scenario a family-wise rates and planted-edge power of the permutation tests (delta DV)."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            raw = s13_raw(est, suffix)
            null = raw[raw.scenario == "a"]
            for method, unit in (("perm_film_maxT", ["rep", "film"]), ("perm_24_maxT", ["rep"]), ("perm_mean_maxT", ["rep"]),
                                 ("perm_interaction_maxT", ["rep"]), ("perm_contrast_maxT", ["rep"])):
                any_hit = null[null.method == method].groupby(unit).detected.any()
                rows.append({"estimator": est, "scenario": "a", "measure": f"family-wise, {method} (per {' x '.join(unit)})",
                             "version": version, "value": with_ci(any_hit.sum(), len(any_hit))})
            for scenario, effect in S13_PLANTED.items():
                sub = raw[(raw.scenario == scenario) & (raw.effect == effect) & (raw.edge == "cg:ROI->child:ROI")]
                for method, s2 in sub.groupby("method"):
                    rows.append({"estimator": est, "scenario": scenario, "measure": f"power, {method}, cg:ROI->child:ROI ({effect})",
                                 "version": version, "value": with_ci(s2.detected.sum(), len(s2))})
    return pivot(rows, ["estimator", "scenario", "measure"])


def criteria_changes():
    """Criteria of S3, S4, S7, S13 whose outcome changed."""
    rows = []
    for est in ESTIMATORS:
        for sim in SIMS:
            old = json.loads((folder(est, sim, "_pre_renull") / "summary.json").read_text())["criteria"]
            new = json.loads((folder(est, sim, "") / "summary.json").read_text())["criteria"]
            for name in sorted(set(old) | set(new)):
                if old.get(name) != new.get(name):
                    rows.append({"sim": SIMS[sim], "estimator": est, "criterion": name, "old": old.get(name), "new": new.get(name)})
    return pd.DataFrame(rows, columns=["sim", "estimator", "criterion", "old", "new"])


def unchanged_check():
    """Largest |new - old| in cells that renull must not change: S3 other nulls and study level, S4 design A, S7 other nulls."""
    rows = []
    for est in ESTIMATORS:
        old, new = (pd.read_csv(folder(est, 3, s) / "summary_fwer.csv") for s in ("_pre_renull", ""))
        keep = ~((old["mode"] == "within_group") & (old.level == "group"))
        rows.append({"estimator": est, "cells": "S3 fwer/per-edge, not (within_group, group)",
                     "max_abs_diff": float(np.max(np.abs(new[keep][["fwer", "per_edge_rate"]].to_numpy() - old[keep][["fwer", "per_edge_rate"]].to_numpy())))})
        old, new = (pd.read_csv(folder(est, 4, s) / "summary.csv") for s in ("_pre_renull", ""))
        a = old.design == "A"
        rows.append({"estimator": est, "cells": "S4 design A power / power_maxT",
                     "max_abs_diff": float(np.max(np.abs(new[a][["power", "power_maxT"]].to_numpy() - old[a][["power", "power_maxT"]].to_numpy())))})
        old, new = (pd.read_csv(folder(est, 7, s) / "summary.csv") for s in ("_pre_renull", ""))
        other = old["mode"] != "within_group"
        rows.append({"estimator": est, "cells": "S7 study/group/dyad FPR, other nulls; study FPR within_group",
                     "max_abs_diff": float(max(np.max(np.abs(new[other][["study_fpr", "group_fpr", "dyad_fpr"]].to_numpy()
                                                             - old[other][["study_fpr", "group_fpr", "dyad_fpr"]].to_numpy())),
                                               np.max(np.abs(new[~other].study_fpr.to_numpy() - old[~other].study_fpr.to_numpy()))))})
    return pd.DataFrame(rows)


def main():
    """Write the comparison tables and print them."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("S03_within_group", s3), ("S04_design_B", s4), ("S07_within_group", s7), ("S13", s13),
                        ("criteria_changes", criteria_changes), ("unchanged_check", unchanged_check)):
        table = build()
        table.to_csv(OUT / f"{name}.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 80):
            print(f"\n=== {name} ===\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
