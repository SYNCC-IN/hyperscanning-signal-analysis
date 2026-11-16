"""Stage 0c - synthetic end-to-end check of Stages 5 and 7 (the real pipeline scripts, unchanged).

Simulated studies (`src.sim_generators`, settings of S1/S3: 25 dyads, 60 s at 2.5 Hz, pole radii and resonance
frequencies of the simulations, two groups of 12 and 13 dyads, three films with new innovations and the same persons)
are written as Stage-2-style design files into a temporary analysis root; `stage05_surrogate.py` and
`stage07_permutation_tests.py` then run on them exactly as on the real data (`SYNCC_ANALYSIS_ROOT`).

- 'planted' (S1 settings): cg:ROI->child:ROI (lag 1, 0.4), cg:HRV->child:HRV (lag 1, 0.3), cg:HRV->child:ROI (lag 2,
  0.3) in every dyad and film. Expected: the planted edges are significant (test 1); with GPDC, reverse edges rarely.
- 'no_coupling' (S3 settings): no coupling. Expected: family-wise rates near 5 %, per-edge rates 3-7 %.

Every film of a study is one replicate of the per-film tests; every study is one replicate of the cross-film tests.
The rates are compared with the stored simulation results where the settings match (`Results/<estimator>/S_01`,
`S_03`). Writes `00c_end_to_end/end_to_end_rates.csv`, `end_to_end_comparison.csv` and the raw Stage 7 rows.

Usage: python stage00c_end_to_end.py [n_planted n_no_coupling]   (default 20 and 50 studies)
       python stage00c_end_to_end.py summary                   (rebuild the summaries from the saved raw rows)
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.design import node_names
from src.io_utils import ensure_dir
from src.pipeline_config import load_stage_config
from src.sim_generators import sample_node_params, simulate_dyads_fixed_params, zscore_designs
from src.sim_inference import wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage07_permutation_tests")
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"] / "00c_end_to_end")
SIM_RESULTS = PROJECT_ROOT / "simulations_multimodal_MVAR" / "dDTF" / "Results"

FILMS = CFG["FILMS"]
ESTIMATORS = CFG["ESTIMATORS"]
NODE_NAMES = node_names(CFG["nodes"])           # child:ROI, cg:ROI, child:HRV, cg:HRV (the simulations' node order)
ALPHA = CFG["ALPHA"]
FS = CFG["TARGET_SFREQ"]
N_DYADS, SEG_S = 25, 60                         # S1 / S3 cell
GROUP_SIZES = {"TD": 12, "ASD": 13}
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
PLANTED = {"cg:ROI->child:ROI": (1, 0, 1, 0.4), "cg:HRV->child:HRV": (3, 2, 1, 0.3), "cg:HRV->child:ROI": (3, 0, 2, 0.3)}
SCENARIOS = {"planted": list(PLANTED.values()), "no_coupling": []}
SUMMARY_ONLY = len(sys.argv) > 1 and sys.argv[1] == "summary"
N_STUDIES = {"planted": 20, "no_coupling": 50} if SUMMARY_ONLY or len(sys.argv) < 3 else \
    {"planted": int(sys.argv[1]), "no_coupling": int(sys.argv[2])}
BASE_SEED = 20261116
N_WORKERS = 6


def write_study(root, scenario, seed):
    """Simulate one study and write it as Stage-2 design files `<dyad>_<film>.nc` under `root/02_envelopes`."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(list(GROUP_SIZES), list(GROUP_SIZES.values()))
    radii, f0s = sample_node_params(rng, N_DYADS, POLE_RADIUS_RANGES), sample_node_params(rng, N_DYADS, F0_RANGES)
    out = ensure_dir(root / "02_envelopes")
    for film in FILMS:
        designs = zscore_designs(simulate_dyads_fixed_params(rng, radii, f0s, int(SEG_S * FS), FS,
                                                             [SCENARIOS[scenario]] * N_DYADS, N_LAGS))
        for d in range(N_DYADS):
            xr.DataArray(designs[d], dims=("variable", "time"),
                         coords={"variable": NODE_NAMES, "time": np.arange(designs.shape[2]) / FS},
                         attrs={"fs": FS, "group": groups[d], "age_months": 48.0, "film": film, "dyad_id": f"S_{d:03d}"},
                         ).to_netcdf(out / f"S_{d:03d}_{film}.nc")


def run_study(task):
    """Write one study, run Stages 5 and 7 on it, return its Stage 7 rows."""
    root = Path(tempfile.mkdtemp(prefix=f"e2e_{task['scenario']}_{task['rep']:03d}_"))
    write_study(root, task["scenario"], task["seed"])
    env = dict(os.environ, SYNCC_ANALYSIS_ROOT=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    for script in ("stage05_surrogate.py", "stage07_permutation_tests.py"):
        subprocess.run([sys.executable, str(HERE / script)], env=env, check=True, capture_output=True)
    rows = pd.concat([pd.read_csv(root / "07_permutation" / est / "stage07_tests.csv") for est in ESTIMATORS])
    shutil.rmtree(root)
    print(f"  {task['scenario']} {task['rep'] + 1}/{N_STUDIES[task['scenario']]} done", flush=True)
    return rows.assign(scenario=task["scenario"], rep=task["rep"])


def rate(flags):
    """Rate, Wilson 95 % interval and n of a boolean series."""
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = wilson_interval(k, n)
    return {"rate": k / n, "ci_low": lo, "ci_high": hi, "n": n}


def summarize(raw):
    """Rejection rates: per edge (raw and max-T) and family-wise (any max-T rejection per replicate)."""
    raw = raw[raw["p_raw"].notna()].assign(reject_raw=lambda d: d["p_raw"] < ALPHA, reject_maxT=lambda d: d["p_maxT"] < ALPHA)
    out = []
    for (scenario, est, test, scope), sub in raw.groupby(["scenario", "estimator", "test", "scope"]):
        unit = ["rep", "film"] if scope.startswith("per film") else ["rep"]
        if not test.startswith("7") and not test.startswith("8"):
            fwer = sub.groupby(unit)["reject_maxT"].any()          # one family per film (per-film tests) or per study
            out.append(dict(scenario=scenario, estimator=est, test=test, scope=scope, edge="any (family-wise, max-T)", **rate(fwer)))
        for edge, e_sub in sub.groupby("edge"):
            out.append(dict(scenario=scenario, estimator=est, test=test, scope=scope, edge=edge, kind="per-edge raw",
                            **rate(e_sub["reject_raw"])))
            if not test.startswith("7") and not test.startswith("8"):
                out.append(dict(scenario=scenario, estimator=est, test=test, scope=scope, edge=edge, kind="per-edge max-T",
                                **rate(e_sub["reject_maxT"])))
    return pd.DataFrame(out)


def comparison(rates):
    """Pipeline rates next to the stored simulation results with matching settings (25 dyads, 60 s)."""
    rows = []
    for est in ESTIMATORS:
        s1 = pd.read_csv(SIM_RESULTS / est / "S_01" / "partB_detection_rate_maxT.csv").set_index("edge")
        e1 = rates[(rates.scenario == "planted") & (rates.estimator == est) & (rates.test == "1 edge existence")
                   & (rates.kind == "per-edge max-T")].set_index("edge")
        for edge in e1.index:
            rows.append({"estimator": est, "check": "S1 planted: test 1 detection (max-T)", "edge": edge,
                         "pipeline": e1.loc[edge, "rate"], "simulation": s1.loc[edge, "rate_maxT"] if edge in s1.index else np.nan,
                         "note": "simulation: pooled null, one group; pipeline: within-group re-pairing, two groups"})
        fwer = pd.read_csv(SIM_RESULTS / est / "S_03" / "summary_fwer.csv")
        cell = fwer[(fwer.n_dyads == N_DYADS) & (fwer.seg_s == SEG_S) & (fwer["mode"] == "within_group")]
        for level, test, scope in (("study", "1 edge existence", "per film"), ("group", "4 group contrast", "per film, max-T within film")):
            pipe = rates[(rates.scenario == "no_coupling") & (rates.estimator == est) & (rates.test == test) & (rates.scope == scope)
                         & (rates.edge == "any (family-wise, max-T)")]
            sim = cell[(cell.level == level) & (cell.method == "maxT")] if {"level", "method"} <= set(cell.columns) else cell.iloc[:0]
            rows.append({"estimator": est, "check": f"S3 no coupling: FWER max-T, {test}", "edge": "family of 8",
                         "pipeline": pipe["rate"].iloc[0] if len(pipe) else np.nan,
                         "simulation": sim["fwer"].iloc[0] if len(sim) and "fwer" in sim else np.nan,
                         "note": "within-group null, 25 dyads, 60 s"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seeds = np.random.SeedSequence(BASE_SEED).generate_state(sum(N_STUDIES.values()))
    tasks = [{"scenario": s, "rep": r} for s, n in N_STUDIES.items() for r in range(n)]
    for task, seed in zip(tasks, seeds):
        task["seed"] = int(seed)
    if SUMMARY_ONLY:
        raw = pd.read_csv(OUTPUT_DIR / "end_to_end_stage07_rows.csv", keep_default_na=False, na_values=[""])
        raw["scenario"] = raw["scenario"].replace({"null": "no_coupling"})     # label of the first run
    else:
        with ThreadPoolExecutor(N_WORKERS) as pool:
            raw = pd.concat(list(pool.map(run_study, tasks)), ignore_index=True)
        raw.to_csv(OUTPUT_DIR / "end_to_end_stage07_rows.csv", index=False)
    rates = summarize(raw)
    rates.to_csv(OUTPUT_DIR / "end_to_end_rates.csv", index=False)
    comp = comparison(rates)
    comp.to_csv(OUTPUT_DIR / "end_to_end_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(rates[rates.edge.str.startswith("any")].to_string(index=False))
    print(comp.to_string(index=False))
