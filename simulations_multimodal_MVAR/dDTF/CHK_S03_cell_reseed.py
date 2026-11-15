"""Check - rerun of the outlying S3 cell (N = 25 dyads, 120 s) with new seeds and 1000 studies.

In S03 (200 studies per cell) the study-level false-positive rate of cg:ROI->child:ROI in the cell N = 25, 120 s
reached 12.5% (Wilson 95% CI 8.6-17.8%) for both measures. This check repeats that cell with a new seed stream and
1000 studies (Monte Carlo SE about 0.7% at a rate of 5%), using S03's `run_task` unchanged (same generator, all
8 cross-person edges, all three nulls, the same multiplicity corrections). Studies are saved in chunks, so a run
can be resumed.

Usage: python CHK_S03_cell_reseed.py full dDTF|GPDC|summarize   (S03 reads the same two arguments on import)
Results: simulations_multimodal_MVAR/checks_2026-11/S3_cell_reseed/
"""

import os
import sys
from pathlib import Path

import pandas as pd

import S03_null_calibration as s03                                            # reads MODE, ESTIMATOR from argv

ESTIMATOR = sys.argv[2]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sim_harness import run_parallel, spawn_seeds                         # noqa: E402
from src.sim_inference import wilson_interval                                 # noqa: E402
from src import sim_progress                                                  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_JOBS = int(os.environ.get("SIM_NJOBS", 5))
N_DYADS, SEG_S = 25, 120
N_STUDIES = 1000
CHUNK = 200                                    # studies per saved chunk
BASE_SEED = 20261115                           # new stream (S03 used 20260103)
TARGET_EDGE = "cg:ROI->child:ROI"
OUT = Path(__file__).resolve().parents[1] / "checks_2026-11" / "S3_cell_reseed"


def run_estimator(estimator):
    """Run the 1000 studies of the cell for one estimator, chunk by chunk; write raw_<estimator>_<chunk>.csv."""
    s03.ESTIMATOR = estimator
    seeds = spawn_seeds(BASE_SEED, N_STUDIES)
    sim_progress.begin(f"CHK S3 cell reseed {estimator}", OUT, mode="full", replicates=N_STUDIES)
    for start in range(0, N_STUDIES, CHUNK):
        path = OUT / f"raw_{estimator}_{start:04d}.csv"
        if path.exists():
            continue
        tasks = [{"n_dyads": N_DYADS, "seg_s": SEG_S, "rep": r, "seed": seeds[r]} for r in range(start, start + CHUNK)]
        results = run_parallel(s03.run_task, tasks, N_JOBS)
        pd.DataFrame([rec for recs, *_ in results for rec in recs]).to_csv(path, index=False)


def summarize():
    """Per-edge study-level rates with Wilson intervals and max-T family-wise rates, both estimators."""
    rows, fwer = [], []
    for estimator in ("dDTF", "GPDC"):
        files = sorted(OUT.glob(f"raw_{estimator}_*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files])
        for (mode, edge), sub in df.groupby(["mode", "edge"]):
            hits, n = int(sub.study_reject.sum()), len(sub)
            low, high = wilson_interval(hits, n)
            rows.append({"estimator": estimator, "mode": mode, "edge": edge, "study_fpr": hits / n,
                         "ci_low": low, "ci_high": high, "group_fpr": sub.group_reject.mean(), "n_rep": n})
        fwer.append(s03.fwer_table(df).assign(estimator=estimator))
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "summary_rates.csv", index=False)
    fwer = pd.concat(fwer)
    fwer.to_csv(OUT / "summary_fwer.csv", index=False)
    print(rates[rates.edge == TARGET_EDGE].round(4).to_string(index=False))
    print(rates.groupby(["estimator", "mode"]).study_fpr.agg(["mean", "min", "max"]).round(4).to_string())
    print(fwer[fwer.method == "maxT"].round(4).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if ESTIMATOR == "summarize":
        summarize()
    else:
        run_estimator(ESTIMATOR)
