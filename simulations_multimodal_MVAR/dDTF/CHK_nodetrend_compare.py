"""Old (per-window linear detrending) vs new default (no per-window detrending): key numbers of S1, S3, S5 and S8.

Reads Results/<estimator>/S_NN_linear_detrend/ (the full run of 2026-11-10/11) and Results/<estimator>/S_NN/ (the
rerun with the new default, same seeds) and writes comparison tables with 95% Wilson intervals. Rates that pool edges
(S3 per-edge, S8) use n = studies x edges, which ignores the correlation between edges of one study; their intervals
are somewhat too narrow.

Usage: python CHK_nodetrend_compare.py
Results: simulations_multimodal_MVAR/checks_2026-11/nodetrend/
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sim_inference import wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS = Path(__file__).resolve().parent / "Results"
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "nodetrend"
ESTIMATORS = ["dDTF", "GPDC"]
VERSIONS = {"linear (old)": "_linear_detrend", "none (new)": ""}
S1_N_REP = 200
S3_N_EDGES = 8
S8_NULL_EDGES, S8_PLANTED_EDGES = 6, 3
S5_SWEEP_SCENARIO_ROLE = "absent"


def folder(estimator, sim_id, suffix):
    """Results folder of one simulation and version."""
    return RESULTS / estimator / f"S_{sim_id:02d}{suffix}"


def with_ci(rate, n):
    """'rate (low-high)' with a 95% Wilson interval."""
    low, high = wilson_interval(int(round(rate * n)), int(n))
    return f"{rate:.3f} ({low:.3f}-{high:.3f})"


def side_by_side(rows, keys):
    """Pivot long rows (keys + version + value) to one column per version."""
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index=keys, columns="version", values="value", aggfunc="first").reset_index()[keys + list(VERSIONS)]


def s1():
    """S1 part B: detection of every directed edge (planted, reverse, absent), uncorrected and after max-T."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            table = pd.read_csv(folder(est, 1, suffix) / "partB_detection_rate_maxT.csv")
            for r in table.itertuples():
                rows.append({"estimator": est, "edge": r.edge, "planted": r.planted, "measure": "uncorrected",
                             "version": version, "value": with_ci(r.rate_uncorrected, S1_N_REP)})
                rows.append({"estimator": est, "edge": r.edge, "planted": r.planted, "measure": "max-T",
                             "version": version, "value": with_ci(r.rate_maxT, S1_N_REP)})
            median_z = pd.read_csv(folder(est, 1, suffix) / "partB_summary.csv")
            for r in median_z.itertuples():
                rows.append({"estimator": est, "edge": r.edge, "planted": True, "measure": "median z",
                             "version": version, "value": f"{r.median_z:.2f}"})
    return side_by_side(rows, ["estimator", "planted", "edge", "measure"])


def s3():
    """S3: per-edge (mean over 8 edges) and max-T family-wise false-positive rates, study level and group contrast."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            fwer = pd.read_csv(folder(est, 3, suffix) / "summary_fwer.csv")
            for r in fwer[fwer.method.isin(["none", "maxT"])].itertuples():
                keys = {"estimator": est, "n_dyads": r.n_dyads, "seg_s": r.seg_s, "mode": r.mode, "level": r.level}
                if r.method == "none":
                    rows.append(dict(keys, measure="per-edge", version=version, value=with_ci(r.per_edge_rate, r.n_rep * S3_N_EDGES)))
                else:
                    rows.append(dict(keys, measure="FWER max-T", version=version, value=with_ci(r.fwer, r.n_rep)))
    return side_by_side(rows, ["estimator", "level", "mode", "measure", "n_dyads", "seg_s"])


def s5():
    """S5: omitted-driver sweep (heart-heart edge without a true link) and the absent / planted edges of the main run."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            sweep = pd.read_csv(folder(est, 5, suffix) / "summary_sweep.csv")
            for r in sweep[sweep.role == S5_SWEEP_SCENARIO_ROLE].itertuples():
                rows.append({"estimator": est, "table": "omitted-driver sweep", "analysis": r.analysis.split(" (")[0],
                             "edge": r.edge_label, "gain": r.gain, "seg_s": r.seg_s, "p": r.p, "version": version,
                             "value": with_ci(r.study_rate, r.n_rep)})
            main = pd.read_csv(folder(est, 5, suffix) / "summary.csv")
            for r in main[main.role != "absent reverse"].itertuples():
                rows.append({"estimator": est, "table": f"{r.scenario}, {r.role}", "analysis": r.analysis.split(" (")[0],
                             "edge": r.edge_label, "gain": r.gain, "seg_s": 60, "p": 4, "version": version,
                             "value": with_ci(r.study_rate, r.n_rep)})
    return side_by_side(rows, ["estimator", "table", "analysis", "edge", "gain", "seg_s", "p"])


def s8():
    """S8: null false-positive rate, power and direction accuracy of every variant."""
    rows = []
    for est in ESTIMATORS:
        for version, suffix in VERSIONS.items():
            n_rep = 200
            for r in pd.read_csv(folder(est, 8, suffix) / "summary.csv").itertuples():
                keys = {"estimator": est, "variant": r.variant}
                rows.append(dict(keys, measure="null FPR", version=version, value=with_ci(r.null_fpr, n_rep * S8_NULL_EDGES)))
                rows.append(dict(keys, measure="power", version=version, value=with_ci(r.power_planted, n_rep * S8_PLANTED_EDGES)))
                rows.append(dict(keys, measure="direction accuracy", version=version, value=f"{r.direction_accuracy:.3f}"))
    return side_by_side(rows, ["estimator", "variant", "measure"])


def main():
    """Write the four comparison tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("S01", s1), ("S03", s3), ("S05", s5), ("S08", s8)):
        table = build()
        table.to_csv(OUT / f"compare_{name}.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 40):
            print(f"\n=== {name} ===\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
