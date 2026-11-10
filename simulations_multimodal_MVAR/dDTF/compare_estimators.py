"""Side-by-side comparison of the dDTF and GPDC runs of S01-S13 and the variants S04b, S06b, for the manuscript.

Both estimators analyse the same simulated studies (same seeds), so every row is a paired comparison.
Reads Results/<estimator>/S_xx/ and writes Results/comparison/:
  criteria.csv         - every acceptance criterion, met (True/False) per estimator, and `status`: 'pass' (met by
                         every estimator) or 'fail', prefixed 'proposed: ' for criteria marked [proposed] (thresholds
                         not yet fixed);
  runs.csv             - mode and replication count of each run (check both estimators come from the same protocol);
  S_xx_<table>.csv     - the main readouts of each summary table, one column per estimator;
  comparison.md        - all of the above as Markdown tables.
Labels that name the estimator (e.g. '4-node dDTF', 'dDTF (4-node)', criteria texts) are written with '[est]'
so that the rows of both estimators line up.

Usage: python compare_estimators.py      (run_all.py calls it at the end)
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "Results"
OUT_DIR = RESULTS / "comparison"
ESTIMATORS = ["dDTF", "GPDC"]
PLACEHOLDER = "[est]"
DECIMALS = 3
VARIANT_DIRS = {"S_04b": "S_04/partial", "S_06b": "S_06/noise"}   # variants: label -> folder under Results/<estimator>
# sim label -> list of (summary table, key columns, readout columns)
TABLES = {
    "S_01": [("partB_summary.csv", ["edge", "gain"], ["detect_rate", "beats_reverse_rate", "beats_absent_rate", "reverse_detect_rate", "median_z"])],
    "S_02": [("summary.csv", ["truth", "path", "gap"], ["fwd", "rev", "direction_index", "relative_fwd"])],
    "S_03": [("summary.csv", ["n_dyads", "seg_s", "mode", "edge"], ["dyad_fpr", "study_fpr", "group_fpr", "median_z", "sd_z"]),
             ("summary_fwer.csv", ["n_dyads", "seg_s", "mode", "level", "method"], ["fwer", "per_edge_rate"]),
             ("summary_small_n.csv", ["n_dyads", "seg_s", "mode", "edge"], ["study_fpr", "group_fpr", "group_fpr_ci_low", "group_fpr_ci_high", "n_rep"])],
    "S_04": [("summary.csv", ["design", "n_dyads", "seg_s", "gain"], ["power", "power_maxT", "power_minP", "dyad_detection", "direction_accuracy"]),
             ("summary_nonplanted_maxT.csv", ["design", "n_dyads", "seg_s", "gain", "edge"], ["rate_maxT"]),
             ("summary_small_n_group.csv", ["design", "n_dyads", "seg_s", "gain"], ["power", "power_ci_low", "power_ci_high", "power_maxT", "n_rep"])],
    "S_04b": [("summary.csv", ["variant", "level", "gain"], ["power", "power_maxT", "reverse_rate", "median_delta",
                                                           "dyad_rate_coupled", "dyad_rate_uncoupled"])],
    "S_05": [("summary.csv", ["scenario", "gain", "analysis", "role", "edge_label"], ["study_rate", "dyad_rate"]),
             ("summary_collinearity.csv", ["scenario", "gain", "analysis"], ["cond_real", "vif_cg_heart_real", "vif_cg_heart_pseudo"]),
             ("summary_sweep.csv", ["gain", "seg_s", "p", "analysis", "role"], ["study_rate", "cond_real", "vif_cg_heart_real"])],
    "S_06": [("summary.csv", ["radius", "zscore"], ["uncorrected_asymmetry", "corrected_asymmetry", "fpr_smooth_to_rough", "fpr_rough_to_smooth"])],
    "S_06b": [("summary_no_coupling.csv", ["noise_set", "noise_sd"], ["uncorrected_asymmetry", "corrected_asymmetry",
                                                                      "fpr_smooth_to_rough", "fpr_rough_to_smooth"]),
              ("summary_coupled.csv", ["noise_sd", "edge"], ["power", "power_maxT", "direction_accuracy", "reverse_rate"])],
    "S_07": [("summary.csv", ["null_source", "mode", "gain", "sd_log", "ratio"], ["study_fpr", "group_fpr", "dyad_fpr"])],
    "S_08": [("summary.csv", ["variant"], ["null_fpr", "power_planted", "reverse_detection", "direction_accuracy"])],
    "S_09": [("summary.csv", ["truth", "gap", "band"], ["forward_rate", "reverse_rate", "dyad_rate"])],
    "S_10": [("summary.csv", ["option", "truth", "strength"], ["forward_rate", "reverse_rate", "dyad_rate"])],
    "S_11": [("summary_partA.csv", ["scenario", "role", "method"], ["study_rate", "dyad_rate", "direction_accuracy"]),
             ("summary_partB.csv", ["truth", "gap", "method"], ["study_rate", "dyad_rate"])],
    "S_12": [("summary.csv", ["pair", "g_fwd", "g_rev"], ["forward_rate", "reverse_rate", "mutual_rate", "forward_dominant", "asymmetry_rate"])],
    "S_13": [("summary_detection.csv", ["scenario", "method", "dv", "edge", "effect"], ["rate"]),
             ("summary_familywise_null.csv", ["scenario", "method", "dv", "effect", "scope"], ["rate"]),
             ("summary_shrinkage.csv", ["scenario", "method", "edge"], ["reference", "mean_estimate", "bias", "rmse"]),
             ("summary_gate.csv", ["scenario", "model"], ["fail_share", "mean_seconds"])],
}


def neutral(text, estimator):
    """Replace the estimator's name in a label by the placeholder, so labels of both runs match."""
    return text.replace(estimator, PLACEHOLDER)


def sim_dir(estimator, sim):
    """Results folder of a simulation label (S_xx or a variant of VARIANT_DIRS)."""
    return RESULTS / estimator / VARIANT_DIRS.get(sim, sim)


def summary_files(estimator):
    """(label, summary.json path) of every run of one estimator, variants after their simulation."""
    labels = sorted({p.parent.name for p in RESULTS.glob(f"{estimator}/S_*/summary.json")} | set(VARIANT_DIRS))
    return [(sim, sim_dir(estimator, sim) / "summary.json") for sim in labels if (sim_dir(estimator, sim) / "summary.json").exists()]


def criterion_status(row):
    """'pass' if every estimator met the criterion, else 'fail'; prefixed 'proposed: ' for criteria marked [proposed]."""
    verdict = "pass" if all(row[e] is True for e in ESTIMATORS) else "fail"
    return f"proposed: {verdict}" if "[proposed]" in row["criterion"] else verdict


def criteria_table():
    """Acceptance criteria of every simulation, one boolean column per estimator (missing run -> empty), and `status`."""
    rows = []
    for estimator in ESTIMATORS:
        for sim, path in summary_files(estimator):
            for criterion, met in json.loads(path.read_text())["criteria"].items():
                rows.append({"sim": sim, "criterion": neutral(criterion, estimator), "estimator": estimator, "met": met})
    table = pd.DataFrame(rows).pivot_table(index=["sim", "criterion"], columns="estimator", values="met", aggfunc="first", sort=False)
    table = table.reindex(columns=ESTIMATORS).reset_index()
    table["status"] = table.apply(criterion_status, axis=1)
    return table


def runs_table():
    """Mode and number of replications of every run, one column pair per estimator."""
    rows = []
    for estimator in ESTIMATORS:
        for sim, path in summary_files(estimator):
            summary = json.loads(path.read_text())
            rows.append({"sim": sim, "estimator": estimator,
                         "run": f"{summary['mode']}, n_rep {summary['settings'].get('n_rep', '?')}"})
    table = pd.DataFrame(rows).pivot(index="sim", columns="estimator", values="run")
    return table.reindex(columns=ESTIMATORS).reset_index()


def readout_table(sim, table_name, keys, readouts):
    """One summary table of one simulation: key columns, then each readout once per estimator."""
    merged = None
    for estimator in ESTIMATORS:
        path = sim_dir(estimator, sim) / table_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for key in keys:
            if pd.api.types.is_string_dtype(df[key]):
                df[key] = df[key].map(lambda v: neutral(v, estimator))
        df = df[keys + readouts].rename(columns={r: f"{r} {estimator}" for r in readouts})
        merged = df if merged is None else merged.merge(df, on=keys, how="outer", sort=False)
    if merged is None:
        return None
    ordered = [f"{r} {e}" for r in readouts for e in ESTIMATORS if f"{r} {e}" in merged]
    return merged[keys + ordered]


def to_markdown(df):
    """Plain Markdown table (numbers rounded to DECIMALS, missing values as '-')."""
    def cell(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if pd.isna(v):
            return "-"
        return f"{v:.{DECIMALS}f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    """Write the criteria, run and readout comparisons as CSV files and one Markdown file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    criteria, runs = criteria_table(), runs_table()
    criteria.to_csv(OUT_DIR / "criteria.csv", index=False)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    md = [f"# Estimator comparison: {' vs '.join(ESTIMATORS)}", "",
          f"Same simulated studies for every estimator (paired). `{PLACEHOLDER}` stands for the estimator of the column.", "",
          "## Runs", "", to_markdown(runs), "", "## Acceptance criteria", "", to_markdown(criteria), ""]
    for sim, tables in TABLES.items():
        for table_name, keys, readouts in tables:
            table = readout_table(sim, table_name, keys, readouts)
            if table is None:
                continue
            stem = Path(table_name).stem
            table.to_csv(OUT_DIR / f"{sim}_{stem}.csv", index=False)
            md += [f"## {sim} - {stem}", "", to_markdown(table), ""]
    (OUT_DIR / "comparison.md").write_text("\n".join(md))
    print(f"Estimator comparison written to {OUT_DIR}")


if __name__ == "__main__":
    main()
