"""S13 factorial-ANOVA diagnostics on the studies of the `fl` run (no new studies).

Every study of scenarios a, h, i is re-simulated from its seed and its delta is asserted to equal, exactly, the delta
saved by the `fl` run (S_13_delta_*.parquet), so the analyses below use the same studies with float64 surrogate tables
(the saved T_bc tables are float32; the float64 tables are written as T_bc64_<scenario>_*.npz). The permutation
matrix is redrawn from the study's generator (seed, 2) as in the `fl` run, and the F p-values of the `fl` run are
asserted to be reproduced exactly.

Step 2  F_obs of the fixed variant equals F_obs of the renull variant at the observed labels (every group term).
Step 3  group terms (omnibus G, GxF, GxE, Gxdirection, Gxsource, Gxtarget, GxFxE; per-edge max-F G, GxF) in a and h:
        variants fixed (Freedman-Lane, Z = 1) and renull, statistics F and ss_hyp; scenario i: G main effect, fixed
        Freedman-Lane with c in Z and the hybrid "renull + covariate" (delta re-nulled for every relabelling,
        Y ~ group + c), statistics F, ss_hyp and beta (squared group coefficient).
Step 4  G main effect in a and h: SS_hyp and SS_err at the observed labels and over the rearrangements.

Usage: python S13_fl_diagnostics.py full [dDTF|GPDC]
  SIM_NJOBS=<n>  parallel processes (default 8)
Writes Results/<estimator>/S_13/: S_13_diag_statistic.csv, S_13_diag_ss.csv, S_13_diag_covariate.csv,
S_13_diag_raw.csv (one row per study and test), S_13_diag_identity.csv (Step 2, per study), T_bc64_<scenario>_*.npz.
"""

import functools
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import S13_group_film_bayes as s13      # reads MODE and ESTIMATOR from sys.argv[1:3], the positions used here
from src import sim_progress
from src.design import window_geometry
from src.perm_anova import (freedman_lane, labels_from_permutations, rearrangement_matrix, relabel_renull_test, ss_components,
                            term_statistic)
from src.sim_harness import run_parallel, spawn_seeds
from src.sim_inference import family_pvalues, renull_delta, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE, ESTIMATOR = sys.argv[1], sys.argv[2]
assert MODE == "full", "the diagnostics read the full `fl` run"
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
OUT_DIR = HERE / "Results" / ESTIMATOR / "S_13"
RUN_TAG = "full_flrun_reps_0-199"                     # files of the `fl` run
GRID_SCENARIOS = ["a", "h"]                           # Steps 2-4
COVARIATE_SCENARIO = "i"                              # Step 3, covariate
STATISTICS = ["F", "ss_hyp"]
COVARIATE_STATISTICS = ["F", "ss_hyp", "beta"]
IDENTITY_RTOL = 1e-10
STORED_P_ATOL = 1e-12        # the fl CSV keeps 16 significant digits; p-values are multiples of 1 / (1 + N_PERM)
PERCENTILES = [5, 95]
ALPHA = s13.ALPHA
FWER_BAND = s13.FWER_BAND
GROUP_TERMS = [t for t, (_, kind) in s13.FL_OMNIBUS.items() if kind == "between"]
EDGE_TERMS = [t for t, (_, kind) in s13.FL_EDGE.items() if kind == "between"]
EDGES = list(s13.EDGES)


@functools.lru_cache(maxsize=1)
def saved_run():
    """Delta table and factorial-ANOVA rows of the `fl` run (read once per process)."""
    delta = pd.read_parquet(OUT_DIR / f"S_13_delta_{RUN_TAG}.parquet").set_index(["scenario", "rep"]).sort_index()
    fl = pd.read_csv(OUT_DIR / f"fl_results_{RUN_TAG}.csv").set_index(["scenario", "rep"]).sort_index()
    return delta, fl


def check_same_study(df, scenario, rep):
    """Assert that the re-simulated delta equals the saved delta exactly (same dyads, films, edges, values)."""
    saved = saved_run()[0].loc[(scenario, rep)]
    key = ["dyad", "film", "edge"]
    merged = df.merge(saved[key + ["delta"]], on=key, suffixes=("", "_saved"), how="outer", validate="one_to_one")
    assert len(merged) == len(saved) == len(df), f"study {scenario}/{rep}: different cells"
    assert (merged.delta == merged.delta_saved).all(), f"study {scenario}/{rep}: re-simulated delta differs from the saved delta"


def stored_p(scenario, rep, method, term, edge="all"):
    """p-value of one test in the `fl` run."""
    fl = saved_run()[1].loc[(scenario, rep)]
    row = fl[(fl.method == method) & (fl.effect == term) & (fl.edge == edge)]
    assert len(row) == 1, (scenario, rep, method, term, edge)
    return float(row.p.iloc[0])


def group_tests(cells, groups, perms, renulled, statistic):
    """Fixed (Freedman-Lane, Z = 1) and renull distributions of every group test (columns as `s13.fl_group_term_F`).

    Returns
    -------
    dict of {variant: (observed (m,), null (B, m))}
    """
    observed = groups == 1
    X, Z = s13.fl_design(groups, None)["between"]
    per_edge = cells.reshape(len(cells), len(s13.FILMS), len(EDGES))
    fixed = [freedman_lane(cells @ s13.FL_OMNIBUS[t][0], X, Z, perms, "permute", statistic) for t in GROUP_TERMS]
    fixed += [freedman_lane(per_edge[:, :, e] @ s13.FL_EDGE[t][0], X, Z, perms, "permute", statistic)
              for t in EDGE_TERMS for e in range(len(EDGES))]
    return {"fixed": (np.array([f[0] for f in fixed]), np.stack([f[1] for f in fixed], axis=1)),
            "renull": relabel_renull_test(renulled, observed, perms, s13.fl_group_term_F, statistic)}


def test_rows(obs, null):
    """(scope, term, edge, p) of every group test: omnibus unadjusted, per-edge max-F (standardized, step-down)."""
    k = len(GROUP_TERMS)
    rows = [("omnibus", t, "all", float(p)) for t, p in zip(GROUP_TERMS, family_pvalues(obs[:k], null[:, :k])["unadjusted"])]
    for j, term in enumerate(EDGE_TERMS):
        block = slice(k + j * len(EDGES), k + (j + 1) * len(EDGES))
        rows += [("per-edge max-F", term, edge, float(p)) for edge, p in zip(EDGES, family_pvalues(obs[block], null[:, block])["maxT"])]
    return rows


def ss_rows(cells, groups, perms, renulled):
    """Step 4: SS_hyp and SS_err of the G main effect at the observed labels and over the rearrangements, per variant."""
    observed = groups == 1
    C = s13.FL_OMNIBUS["G"][0]
    X, Z = s13.fl_design(groups, None)["between"]
    Y = cells @ C
    fitted = Z @ (np.linalg.pinv(Z) @ Y)
    labels = np.vstack([observed[None], labels_from_permutations(observed, perms)])
    variants = {"fixed": ss_components(np.concatenate([Y[None], fitted[None] + (Y - fitted)[perms]]), X, Z),
                "renull": ss_components(renulled(labels) @ C, np.where(labels, 1.0, -1.0)[:, :, None], Z)}
    rows = []
    for variant, (hyp, err) in variants.items():
        row = {"variant": variant, "ss_hyp_obs": hyp[0], "ss_err_obs": err[0], "ss_hyp_perm_mean": hyp[1:].mean(),
               "ss_err_perm_mean": err[1:].mean()}
        for q in PERCENTILES:
            row[f"ss_hyp_perm_p{q}"], row[f"ss_err_perm_p{q}"] = np.percentile(hyp[1:], q), np.percentile(err[1:], q)
        rows.append(row)
    return rows


def covariate_rows(cells, groups, covariate, perms, renulled):
    """Scenario i, G main effect: fixed Freedman-Lane with c in Z and the hybrid renull + covariate, every statistic."""
    observed = groups == 1
    C = s13.FL_OMNIBUS["G"][0]
    X, Z = s13.fl_design(groups, covariate)["between"]
    rows, identity = [], []

    def hybrid(y, lab, statistic):
        return term_statistic(y @ C, np.where(lab, 1.0, -1.0)[:, :, None], Z, statistic)[:, None]

    for statistic in COVARIATE_STATISTICS:
        f_obs, f_null = freedman_lane(cells @ C, X, Z, perms, "permute", statistic)
        h_obs, h_null = relabel_renull_test(renulled, observed, perms, hybrid, statistic)
        for method, obs, null in (("fixed_FL_cov", np.array([f_obs]), f_null[:, None]), ("renull_cov_hybrid", h_obs, h_null)):
            rows.append({"method": method, "statistic": statistic, "term": "G", "p": float(family_pvalues(obs, null)["unadjusted"][0])})
        identity.append(abs(f_obs - h_obs[0]) / abs(f_obs))
    return rows, float(max(identity))


def study(task):
    """Re-simulate one study, check it against the `fl` run, and run the Step 2-4 analyses."""
    start = time.time()
    scenario, rep = task["scenario"], task["rep"]
    win_len, step = window_geometry(s13.WIN_LEN_S, s13.OVERLAP, s13.FS)
    est = dict(p=s13.MODEL_ORDER, win_len=win_len, step=step, freqs=s13.FREQS, fs=s13.FS, estimator=ESTIMATOR,
               band_hz=s13.BAND_HZ, box_cox_lambda=s13.BOX_COX)
    df, groups, tables, covariate = s13.simulate_delta_table(np.random.default_rng(task["seed"]), scenario, s13.SETTINGS["n_per_group"], est)
    check_same_study(df, scenario, rep)
    used, _, cells = s13.fl_cells(df, len(groups))
    assert used.all(), "every dyad has all films in the fl run"
    perms = rearrangement_matrix(len(groups), s13.N_PERM, "permute", np.random.default_rng([task["seed"], 2]))
    real = np.hstack([tables[f][0] for f in s13.FILMS])
    pair_values = np.concatenate([tables[f][1] for f in s13.FILMS], axis=2)

    def renulled(labels):
        return renull_delta(real, pair_values, labels)

    tag = {"scenario": scenario, "rep": rep}
    out = {"raw": [], "identity": [], "ss": [], "tables": tables, "groups": groups, "covariate": covariate}
    if scenario == COVARIATE_SCENARIO:
        rows, identity = covariate_rows(cells, groups, covariate, perms, renulled)
        f_row = [r for r in rows if r["method"] == "fixed_FL_cov" and r["statistic"] == "F"][0]
        assert abs(f_row["p"] - stored_p(scenario, rep, "fl_omnibus_fixed", "G")) <= STORED_P_ATOL, "fixed FL (c in Z) does not reproduce the fl run"
        out["raw"] = [dict(r, scope="omnibus", edge="all", **tag) for r in rows]
        out["identity"] = [dict(tag, variant_pair="fixed_FL_cov vs renull_cov_hybrid (all statistics)", max_rel_diff=identity,
                                n_mismatch=int(identity > IDENTITY_RTOL), n_tests=len(COVARIATE_STATISTICS))]
    else:
        f_obs = {}
        for statistic in STATISTICS:
            for variant, (obs, null) in group_tests(cells, groups, perms, renulled, statistic).items():
                f_obs[(variant, statistic)] = obs
                for scope, term, edge, p in test_rows(obs, null):
                    out["raw"].append({"method": variant, "statistic": statistic, "scope": scope, "term": term, "edge": edge, "p": p, **tag})
                    if statistic == "F":
                        method = f"fl_{'omnibus' if scope == 'omnibus' else 'edge'}_{variant}{'_maxF' if scope != 'omnibus' else ''}"
                        assert abs(p - stored_p(scenario, rep, method, term, edge)) <= STORED_P_ATOL, f"{method} {term} {edge} does not reproduce the fl run"
        rel = np.abs(f_obs[("fixed", "F")] - f_obs[("renull", "F")]) / np.abs(f_obs[("fixed", "F")])
        out["identity"] = [dict(tag, variant_pair="fixed vs renull (F)", max_rel_diff=float(rel.max()),
                                n_mismatch=int((rel > IDENTITY_RTOL).sum()), n_tests=len(rel))]
        out["ss"] = [dict(r, **tag) for r in ss_rows(cells, groups, perms, renulled)]
    out["seconds"] = time.time() - start
    return out


def rate_row(flags):
    """Rate, Wilson 95% interval and n of a boolean series."""
    k, n = int(np.sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return {"rate": k / n, "ci_low": low, "ci_high": high, "n": n, "in_band": bool(FWER_BAND[0] <= k / n <= FWER_BAND[1])}


def summarize(raw, ss):
    """Step 3 rate tables (omnibus rejection, per-edge max-F family-wise rate) and Step 4 ratio table."""
    raw = raw.assign(detected=raw.p < ALPHA)
    grid = raw[raw.scenario.isin(GRID_SCENARIOS)]
    rows = []
    for (scenario, variant, statistic, scope, term), sub in grid.groupby(["scenario", "method", "statistic", "scope", "term"]):
        rows.append({"estimator": ESTIMATOR, "scenario": scenario, "variant": variant, "statistic": statistic, "term": term,
                     "scope": scope, **rate_row(sub.groupby("rep").detected.any())})
    cov = [{"estimator": ESTIMATOR, "scenario": COVARIATE_SCENARIO, "method": m, "statistic": s, **rate_row(sub.detected)}
           for (m, s), sub in raw[raw.scenario == COVARIATE_SCENARIO].groupby(["method", "statistic"])]
    ratios = []
    for (scenario, variant), sub in ss.groupby(["scenario", "variant"]):
        r = {"estimator": ESTIMATOR, "scenario": scenario, "variant": variant, "n": len(sub),
             "median_ss_err_perm_over_obs": float(np.median(sub.ss_err_perm_mean / sub.ss_err_obs)),
             "median_ss_hyp_perm_over_obs": float(np.median(sub.ss_hyp_perm_mean / sub.ss_hyp_obs))}
        for q in PERCENTILES:
            r[f"median_ss_err_perm_p{q}_over_obs"] = float(np.median(sub[f"ss_err_perm_p{q}"] / sub.ss_err_obs))
        ratios.append(r)
    return pd.DataFrame(rows), pd.DataFrame(cov), pd.DataFrame(ratios)


def main():
    """Run the diagnostics for one estimator and write the tables."""
    sim_progress.begin(f"S13 fl diagnostics {ESTIMATOR}", HERE / "Results", mode=MODE)
    delta = saved_run()[0]
    seeds = spawn_seeds(s13.BASE_SEED, len(s13.SCENARIOS) * s13.N_REP_MAX)
    tasks = [{"scenario": s, "rep": int(r), "seed": seeds[list(s13.SCENARIOS).index(s) * s13.N_REP_MAX + int(r)]}
             for s in GRID_SCENARIOS + [COVARIATE_SCENARIO] for r in sorted(delta.loc[s].index.unique())]
    start = time.time()
    results = run_parallel(study, tasks, N_JOBS)
    raw = pd.DataFrame([r for res in results for r in res["raw"]])
    identity = pd.DataFrame([r for res in results for r in res["identity"]])
    ss = pd.DataFrame([r for res in results for r in res["ss"]])
    raw.to_csv(OUT_DIR / "S_13_diag_raw.csv", index=False)
    identity.to_csv(OUT_DIR / "S_13_diag_identity.csv", index=False)
    for scenario in GRID_SCENARIOS + [COVARIATE_SCENARIO]:
        arrays = {}
        for task, res in zip(tasks, results):
            if task["scenario"] != scenario:
                continue
            key = f"rep{task['rep']:04d}"
            arrays[f"{key}_groups"] = res["groups"]
            if res["covariate"] is not None:
                arrays[f"{key}_covariate"] = res["covariate"]
            for film, (real, values) in res["tables"].items():
                arrays[f"{key}_{film}_real"], arrays[f"{key}_{film}_values"] = real, values
        np.savez_compressed(OUT_DIR / f"T_bc64_{scenario}_{RUN_TAG}.npz", edges=np.array(EDGES), **arrays)
    statistic, covariate, ratios = summarize(raw, ss)
    statistic.to_csv(OUT_DIR / "S_13_diag_statistic.csv", index=False)
    covariate.to_csv(OUT_DIR / "S_13_diag_covariate.csv", index=False)
    ratios.to_csv(OUT_DIR / "S_13_diag_ss.csv", index=False)
    seconds = np.array([res["seconds"] for res in results])
    print(f"{len(tasks)} studies in {time.time() - start:.0f} s with {N_JOBS} processes; {seconds.mean():.1f} s per study")
    print(f"Step 2: {int(identity.n_mismatch.sum())} mismatches; max rel diff {identity.max_rel_diff.max():.2e}")
    sim_progress.finish(OUT_DIR, {"studies": str(len(tasks))}, {})


if __name__ == "__main__":
    main()
