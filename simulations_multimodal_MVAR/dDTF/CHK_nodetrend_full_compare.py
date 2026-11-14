"""Old (per-window linear detrending) vs new default (no detrending): all estimator-comparison tables of S1-S13.

Reads Results/comparison_linear_detrend/ (from the full run of 2026-11-10/11) and Results/comparison/ (rebuilt by
compare_estimators.py after the rerun on the same seeds). Every table there has key columns and one column per
measure and estimator ("<measure> dDTF", "<measure> GPDC").
Outputs (simulations_multimodal_MVAR/checks_2026-11/nodetrend/):
  full_criteria_changes.csv   criteria whose outcome changed (per estimator);
  full_changes.csv            every cell of a rate measure whose change exceeds the Monte Carlo noise
                              (|new - old| > 1.96 * sqrt(2 p (1 - p) / n), p = mean of old and new, at least
                              MIN_RATE_CHANGE), and every non-rate measure whose change exceeds MIN_RELATIVE_CHANGE of
                              the old value (and MIN_ABS_CHANGE); with old/new, Wilson intervals for rates;
  full_change_counts.csv      per table and measure: cells compared, cells flagged, largest change.
n is the table's n_rep column when present, else N_REP[table prefix]; rates pooled over edges therefore get intervals
that are somewhat too narrow, and old and new share their simulated data, so this test is conservative for
differences that come only from the estimation.

Usage: python CHK_nodetrend_full_compare.py
"""

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
OLD, NEW = RESULTS / "comparison_linear_detrend", RESULTS / "comparison"
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "nodetrend"
ESTIMATORS = ["dDTF", "GPDC"]
RATE_WORDS = ("rate", "fpr", "power", "fwer", "detection", "accuracy", "share", "dominant", "detect")
N_REP = {"S_13": 100}                      # default 200 studies per cell
DEFAULT_N_REP = 200
MIN_RATE_CHANGE = 0.03
MIN_RELATIVE_CHANGE, MIN_ABS_CHANGE = 0.20, 0.02
SKIP = {"criteria.csv", "runs.csv"}


def split_columns(frame):
    """Key columns and measure names of a comparison table."""
    measures = sorted({c.rsplit(" ", 1)[0] for c in frame.columns if c.rsplit(" ", 1)[-1] in ESTIMATORS})
    keys = [c for c in frame.columns if c.rsplit(" ", 1)[-1] not in ESTIMATORS]
    return keys, measures


def is_rate(measure):
    """True for measures that are proportions of studies (rates, power, accuracy)."""
    return any(word in measure.lower() for word in RATE_WORDS) and "ci_" not in measure


def rate_text(p, n):
    """'p (low-high)' with a 95% Wilson interval."""
    low, high = wilson_interval(int(round(p * n)), int(n))
    return f"{p:.3f} ({low:.3f}-{high:.3f})"


def table_changes(name):
    """Flagged changes and counts of one comparison table."""
    old, new = pd.read_csv(OLD / name), pd.read_csv(NEW / name)
    keys, measures = split_columns(old)
    merged = old.merge(new, on=keys, suffixes=(" OLD", " NEW"))
    flagged, counts = [], []
    for measure in measures:
        if measure.startswith("n_rep") or "_ci_" in measure:
            continue
        for est in ESTIMATORS:
            column = f"{measure} {est}"
            if f"{column} OLD" not in merged:
                continue
            a, b = merged[f"{column} OLD"].astype(float), merged[f"{column} NEW"].astype(float)
            n = merged[f"n_rep {est} OLD"].astype(float) if f"n_rep {est} OLD" in merged else pd.Series(N_REP.get(name[:4], DEFAULT_N_REP), index=merged.index)
            if is_rate(measure):
                p = ((a + b) / 2).clip(0, 1)
                limit = np.maximum(1.96 * np.sqrt(2 * p * (1 - p) / n), MIN_RATE_CHANGE)
            else:
                limit = np.maximum(MIN_RELATIVE_CHANGE * a.abs(), MIN_ABS_CHANGE)
            hit = (b - a).abs() > limit
            counts.append({"table": name, "measure": measure, "estimator": est, "cells": int((a.notna() & b.notna()).sum()),
                           "flagged": int(hit.sum()), "largest_change": float((b - a).abs().max())})
            for i in np.flatnonzero(hit.to_numpy()):
                row = {"table": name, "measure": measure, "estimator": est,
                       "cell": ", ".join(f"{k}={merged[k].iloc[i]}" for k in keys), "old": a.iloc[i], "new": b.iloc[i],
                       "change": b.iloc[i] - a.iloc[i]}
                if is_rate(measure):
                    row["old_ci"], row["new_ci"] = rate_text(a.iloc[i], n.iloc[i]), rate_text(b.iloc[i], n.iloc[i])
                flagged.append(row)
    return flagged, counts


def criteria_changes():
    """Criteria whose outcome differs between the old and the new run."""
    old, new = pd.read_csv(OLD / "criteria.csv"), pd.read_csv(NEW / "criteria.csv")
    merged = old.merge(new, on=["sim", "criterion"], how="outer", suffixes=(" old", " new"))
    rows = []
    for r in merged.itertuples(index=False):
        values = dict(zip(merged.columns, r))
        for est in ESTIMATORS:
            if str(values[f"{est} old"]) != str(values[f"{est} new"]):
                rows.append({"sim": values["sim"], "criterion": values["criterion"], "estimator": est,
                             "old": values[f"{est} old"], "new": values[f"{est} new"]})
    return pd.DataFrame(rows)


def main():
    """Write the criteria changes, the flagged cell changes and the per-measure counts."""
    OUT.mkdir(parents=True, exist_ok=True)
    flagged, counts = [], []
    for path in sorted(OLD.glob("*.csv")):
        if path.name in SKIP or not (NEW / path.name).exists():
            continue
        f, c = table_changes(path.name)
        flagged += f
        counts += c
    criteria = criteria_changes()
    criteria.to_csv(OUT / "full_criteria_changes.csv", index=False)
    pd.DataFrame(flagged).to_csv(OUT / "full_changes.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "full_change_counts.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_colwidth", 90):
        print(criteria.to_string(index=False))
        frame = pd.DataFrame(counts)
        print(frame[frame.flagged > 0].to_string(index=False))


if __name__ == "__main__":
    main()
