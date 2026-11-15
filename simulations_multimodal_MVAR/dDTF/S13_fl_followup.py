"""S13 factorial-ANOVA follow-up: power, precision, covariate under unequal reactivity, origin of the F x E inflation.

All studies are simulated from the S13 seed scheme (seed index = scenario index * N_REP_MAX + replicate, from
`spawn_seeds(BASE_SEED, len(SCENARIOS) * N_REP_MAX)`): replicates of the `fl` run reproduce it exactly (asserted on the
delta and on the stored p-values), new replicates (a, h: 200-999; scenario j) get their own seeds of the same scheme.
Float64 surrogate tables are written as T_bc64_<scenario>_follow.npz. Within a study, every factorial-ANOVA method uses
the permutation and sign-flip matrices of the `fl` run (generator (seed, 2)).

Step 1 (power)      c, e, f, g x 100: group terms with renull + ss_hyp, renull + F, fixed + F, and the existing S13 tests
                    (perm_film_maxT, perm_mean_maxT, perm_interaction_maxT; generator (seed, 1), reproduced exactly).
Step 2 (precision)  a, h x 1000: group terms with renull + ss_hyp, renull + F, fixed + F.
Step 3 (scenario j) j x 200: G main effect and per-edge max-F G: hybrid renull + covariate (ss_hyp, F, beta), fixed
                    Freedman-Lane with c in Z (F, ss_hyp), renull + ss_hyp without c; SS_err ratios.
Step 4 (within)     a, h x 1000 (the Step 2 studies): F, E, FxE (sign flip, F and ss_hyp) and the per-edge film contrast
                    with max-T (`signflip_maxT`) on delta, T_real and T_real - pooled null median; across-replicate
                    variances of the intercept contrasts.

Usage: python S13_fl_followup.py full [dDTF|GPDC] [power|precision|j|summary|all]
  SIM_NREP=<n>       replicates per step (smoke tests); SIM_NJOBS=<n> processes (default 8)
  SIM_OUT_DIR=<path> write here instead of Results/<estimator>/S_13 (the `fl` run is always read from Results)
Writes S_13_follow_<step>_raw.csv per step and, with `summary`, S_13_follow_power.csv, S_13_follow_precision.csv,
S_13_follow_j.csv, S_13_follow_j_ssratio.csv, S_13_follow_within.csv, S_13_follow_within_var.csv and the [proposed] power
criteria (merged into summary.json).
"""

import json
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

import S13_group_film_bayes as s13      # reads MODE and ESTIMATOR from sys.argv[1:3]
import S13_fl_diagnostics as diag       # same argv layout; provides the reproduction checks against the fl run
from src import sim_progress
from src.design import window_geometry
from src.perm_anova import (freedman_lane, labels_from_permutations, rearrangement_matrix, relabel_renull_test, ss_components,
                            term_statistic)
from src.sim_harness import run_parallel, spawn_seeds
from src.sim_inference import family_pvalues, group_null_medians, renull_delta, signflip_maxT, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE, ESTIMATOR = sys.argv[1], sys.argv[2]
STEP = sys.argv[3] if len(sys.argv) > 3 else "all"
assert MODE == "full" and STEP in ("power", "precision", "j", "summary", "all")
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
OUT_DIR = Path(os.environ["SIM_OUT_DIR"]) if os.environ.get("SIM_OUT_DIR") else HERE / "Results" / ESTIMATOR / "S_13"
FL_REPS = {"a": 200, "h": 200, "c": 100, "e": 100, "f": 100, "g": 100}          # replicates in the fl run
STEPS = {"power": {"c": 100, "e": 100, "f": 100, "g": 100}, "precision": {"a": 1000, "h": 1000}, "j": {"j": 200}}
if os.environ.get("SIM_NREP"):
    STEPS = {k: {s: int(os.environ["SIM_NREP"]) for s in v} for k, v in STEPS.items()}
BAYES_REPS = 100                                     # replicates of the stored max-T / Bayes run (raw_results_full_*)
EXISTING = {"perm_film_maxT": "group", "perm_mean_maxT": "group", "perm_interaction_maxT": "interaction"}
METHODS = [("renull", "ss_hyp"), ("renull", "F"), ("fixed", "F")]
J_STATISTICS = ["ss_hyp", "F", "beta"]
WITHIN_TERMS = ["F", "E", "FxE"]
WITHIN_STATISTICS = ["F", "ss_hyp"]
DVS = ["delta", "T_real", "T_real_minus_pooled"]
P_ATOL = diag.STORED_P_ATOL
ALPHA, FWER_BAND, POWER_TARGET, LEAKAGE_MAX = s13.ALPHA, s13.FWER_BAND, s13.POWER_TARGET, s13.LEAKAGE_MAX
EDGES = list(s13.EDGES)
GROUP_TERMS, EDGE_TERMS = diag.GROUP_TERMS, diag.EDGE_TERMS


# ---------------------------------------------------------------------------
# Per-study analyses
# ---------------------------------------------------------------------------
def simulate(task):
    """Simulate one study; check it against the fl run when the replicate belongs to it.

    Dyads with an unstable real fit in a film are left out of the factorial analyses (`used`), as in `s13.fl_records`.
    """
    win_len, step = window_geometry(s13.WIN_LEN_S, s13.OVERLAP, s13.FS)
    est = dict(p=s13.MODEL_ORDER, win_len=win_len, step=step, freqs=s13.FREQS, fs=s13.FS, estimator=ESTIMATOR,
               band_hz=s13.BAND_HZ, box_cox_lambda=s13.BOX_COX)
    df, groups, tables, covariate = s13.simulate_delta_table(np.random.default_rng(task["seed"]), task["scenario"],
                                                             s13.SETTINGS["n_per_group"], est)
    if task["rep"] < FL_REPS.get(task["scenario"], 0):
        diag.check_same_study(df, task["scenario"], task["rep"])
    used, _, cells = s13.fl_cells(df, len(groups))
    return df, groups, tables, covariate, cells, used


def rearrangements(seed, n):
    """Permutation and sign-flip matrices of the fl run (generator (seed, 2), permutations drawn first)."""
    rng = np.random.default_rng([seed, 2])
    return rearrangement_matrix(n, s13.N_PERM, "permute", rng), rearrangement_matrix(n, s13.N_PERM, "signflip", rng)


def flip_generator(seed, n):
    """Generator positioned so that its next `sign_flips` draw equals the fl sign-flip matrix."""
    rng = np.random.default_rng([seed, 2])
    rearrangement_matrix(n, s13.N_PERM, "permute", rng)
    return rng


def stacked(tables):
    """Real values (n, 24) and surrogate tables (n, n, 24), film-major."""
    return (np.hstack([tables[f][0] for f in s13.FILMS]), np.concatenate([tables[f][1] for f in s13.FILMS], axis=2))


def group_term_rows(cells, groups, perms, renulled, task, check_stored):
    """Group terms for METHODS (`diag.group_tests`, `diag.test_rows`); F p-values checked against the fl run."""
    rows = []
    for statistic in sorted({s for _, s in METHODS}):
        tests = diag.group_tests(cells, groups, perms, renulled, statistic)
        for variant in sorted({v for v, s in METHODS if s == statistic}):
            obs, null = tests[variant]
            for scope, term, edge, p in diag.test_rows(obs, null):
                rows.append({"method": variant, "statistic": statistic, "scope": scope, "term": term, "edge": edge, "p": p})
                if check_stored and statistic == "F":
                    method = f"fl_{'omnibus' if scope == 'omnibus' else 'edge'}_{variant}{'' if scope == 'omnibus' else '_maxF'}"
                    assert abs(p - diag.stored_p(task["scenario"], task["rep"], method, term, edge)) <= P_ATOL, (task, method, term, edge)
    return rows


def existing_rows(df, groups, tables, task):
    """Existing S13 max-T tests of delta (generator (seed, 1), drawn in the order of `s13.run_task`)."""
    rng = np.random.default_rng([task["seed"], 1])
    records = s13.permutation_records(df, groups, rng, "delta", tables)
    s13.permutation_records(df, groups, rng, "z", tables)
    records += s13.contrast_records(df, groups, rng, "delta", tables)
    return [{"method": r["method"], "statistic": "maxT", "scope": r["film"], "term": r["effect"], "edge": r["edge"], "p": r["p"]}
            for r in records if r["method"] in EXISTING]


def check_existing(rows, task):
    """Assert that the existing tests reproduce the stored max-T run (scenarios and replicates it contains)."""
    stored = stored_existing()
    sub = stored[(stored.scenario == task["scenario"]) & (stored.rep == task["rep"])]
    if sub.empty:
        return
    merged = pd.DataFrame(rows).merge(sub, left_on=["method", "scope", "edge"], right_on=["method", "film", "edge"], suffixes=("", "_stored"))
    assert len(merged) == len(rows) and (np.abs(merged.p - merged.p_stored) <= P_ATOL).all(), f"existing tests differ: {task}"


_STORED = {}


def stored_existing():
    """Stored delta rows of the existing max-T tests (read once per process)."""
    if "existing" not in _STORED:
        frames = [pd.read_csv(p) for p in sorted((HERE / "Results" / ESTIMATOR / "S_13").glob("raw_results_full_reps_*.csv"))]
        raw = pd.concat(frames, ignore_index=True)
        _STORED["existing"] = raw[raw.method.isin(list(EXISTING)) & (raw.dv == "delta")][["scenario", "rep", "method", "film", "edge", "p"]]
    return _STORED["existing"]


def within_rows(cells, real, pair_values, groups, used, flips, seed):
    """Step 4: within-dyad terms and the per-edge film contrast on three DVs, and the intercept contrasts per term.

    The nulls (within-group and pooled medians) use every dyad's valid surrogate pairs; the tests use the dyads `used`.
    """
    pooled = np.nanmedian(pair_values.reshape(-1, pair_values.shape[2]), axis=0)
    medians = group_null_medians(pair_values, (groups == 1)[None])[0]                 # (group, 24)
    real, groups = real[used], groups[used]
    dvs = {"delta": cells, "T_real": real, "T_real_minus_pooled": real - pooled[None]}
    assert np.allclose(dvs["delta"], real - medians[groups]), "delta must be T_real minus the within-group null median"
    X, Z = s13.fl_design(groups, None)["within"]
    rows, vectors = [], []
    for dv, Y in dvs.items():
        for term in WITHIN_TERMS:
            C = s13.FL_OMNIBUS[term][0]
            for statistic in WITHIN_STATISTICS:
                obs, null = freedman_lane(Y @ C, X, Z, flips, "signflip", statistic)
                rows.append({"dv": dv, "scope": "omnibus", "term": term, "statistic": statistic,
                             "p": float(family_pvalues(np.array([obs]), null[:, None])["unadjusted"][0])})
            means = np.stack([(Y @ C)[groups == g].mean(axis=0) for g in (0, 1)]).mean(axis=0)
            vectors += [{"dv": dv, "term": term, "column": j, "value": float(v)} for j, v in enumerate(means)]
        wide = Y.reshape(len(Y), len(s13.FILMS), len(EDGES))
        contrast = wide[:, 0] - wide[:, 1:].mean(axis=1)
        res = signflip_maxT(contrast, flip_generator(seed, len(Y)), s13.N_PERM, groups)
        rows += [{"dv": dv, "scope": "per-edge max-T", "term": "film contrast", "statistic": "maxT", "edge": e, "p": float(p)}
                 for e, p in zip(EDGES, res["maxT"])]
    for term in WITHIN_TERMS:
        offset = (medians @ s13.FL_OMNIBUS[term][0]).mean(axis=0)                       # mean over groups of C' med_g
        vectors += [{"dv": "null_median_offset", "term": term, "column": j, "value": float(v)} for j, v in enumerate(offset)]
    return rows, vectors


def j_rows(cells, groups, covariate, perms, renulled):
    """Step 3: G main effect and per-edge max-F G with the hybrid, fixed FL (c in Z) and renull without c; SS_err ratios."""
    observed = groups == 1
    labels = np.vstack([observed[None], labels_from_permutations(observed, perms)])
    X, Z_cov = s13.fl_design(groups, covariate)["between"]
    Z_one = np.ones((len(groups), 1))
    C_G, C_edge = s13.FL_OMNIBUS["G"][0], s13.FL_EDGE["G"][0]
    per_edge = cells.reshape(len(cells), len(s13.FILMS), len(EDGES))

    def renull_stats(Z):
        def term_fn(y, lab, statistic):
            Xb = np.where(lab, 1.0, -1.0)[:, :, None]
            pe = y.reshape(y.shape[:2] + (len(s13.FILMS), len(EDGES)))
            cols = [term_statistic(y @ C_G, Xb, Z, statistic)] + [term_statistic(pe[..., e] @ C_edge, Xb, Z, statistic) for e in range(len(EDGES))]
            return np.stack(cols, axis=1)
        return term_fn

    def fixed_stats(statistic):
        res = [freedman_lane(cells @ C_G, X, Z_cov, perms, "permute", statistic)]
        res += [freedman_lane(per_edge[:, :, e] @ C_edge, X, Z_cov, perms, "permute", statistic) for e in range(len(EDGES))]
        return np.array([r[0] for r in res]), np.stack([r[1] for r in res], axis=1)

    runs = [("hybrid_renull_cov", s, relabel_renull_test(renulled, observed, perms, renull_stats(Z_cov), s)) for s in J_STATISTICS]
    runs += [("fixed_FL_cov", s, fixed_stats(s)) for s in ("F", "ss_hyp")]
    runs.append(("renull_no_cov", "ss_hyp", relabel_renull_test(renulled, observed, perms, renull_stats(Z_one), "ss_hyp")))
    rows = []
    for method, statistic, (obs, null) in runs:
        rows.append({"method": method, "statistic": statistic, "scope": "omnibus", "term": "G", "edge": "all",
                     "p": float(family_pvalues(obs[:1], null[:, :1])["unadjusted"][0])})
        rows += [{"method": method, "statistic": statistic, "scope": "per-edge max-F", "term": "G", "edge": e, "p": float(p)}
                 for e, p in zip(EDGES, family_pvalues(obs[1:], null[:, 1:])["maxT"])]
    Y_G = renulled(labels) @ C_G
    fitted = Z_cov @ (np.linalg.pinv(Z_cov) @ (cells @ C_G))
    err = {"hybrid_renull_cov": ss_components(Y_G, np.where(labels, 1.0, -1.0)[:, :, None], Z_cov)[1],
           "renull_no_cov": ss_components(Y_G, np.where(labels, 1.0, -1.0)[:, :, None], Z_one)[1],
           "fixed_FL_cov": ss_components(np.concatenate([(cells @ C_G)[None], fitted[None] + ((cells @ C_G) - fitted)[perms]]), X, Z_cov)[1]}
    ss = [{"method": m, "ss_err_obs": e[0], "ss_err_perm_mean": e[1:].mean()} for m, e in err.items()]
    return rows, ss


def study(task):
    """One study of one step."""
    start = time.time()
    df, groups, tables, covariate, cells, used = simulate(task)
    n_used = int(used.sum())
    perms, flips = rearrangements(task["seed"], n_used)
    real, pair_values = stacked(tables)
    groups_used = groups[used]
    covariate_used = None if covariate is None else covariate[used]

    def renulled(labels):
        """Re-nulled delta of the used dyads; left-out dyads keep their observed group in every labelling."""
        full = np.tile(groups == 1, (len(labels), 1))
        full[:, used] = labels
        return renull_delta(real, pair_values, full)[:, used, :]

    tag = {"scenario": task["scenario"], "rep": task["rep"], "seed": task["seed"], "n_dyads": n_used}
    in_fl = task["rep"] < FL_REPS.get(task["scenario"], 0)
    out = {"tables": tables, "groups": groups, "covariate": covariate, "rows": [], "vectors": [], "ss": []}
    if task["step"] in ("power", "precision"):
        out["rows"] = group_term_rows(cells, groups_used, perms, renulled, task, in_fl)
    if task["step"] == "power":
        existing = existing_rows(df, groups, tables, task)
        check_existing(existing, task)
        out["rows"] += existing
    if task["step"] == "precision":
        rows, vectors = within_rows(cells, real, pair_values, groups, used, flips, task["seed"])
        out["rows"] += [dict(r, method="within") for r in rows]
        out["vectors"] = vectors
    if task["step"] == "j":
        out["rows"], out["ss"] = j_rows(cells, groups_used, covariate_used, perms, renulled)
    for key in ("rows", "vectors", "ss"):
        out[key] = [dict(r, step=task["step"], **tag) for r in out[key]]
    out["seconds"] = time.time() - start
    return out


def write_step(step, tasks, results):
    """Raw rows, intercept vectors, SS rows and float64 tables of one step."""
    for key in ("rows", "vectors", "ss"):
        rows = [r for res in results for r in res[key]]
        if rows:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"S_13_follow_{step}_{key}.csv", index=False)
    for scenario in sorted({t["scenario"] for t in tasks}):
        arrays = {}
        for task, res in zip(tasks, results):
            if task["scenario"] != scenario:
                continue
            key = f"rep{task['rep']:04d}"
            arrays[f"{key}_groups"] = res["groups"]
            if res["covariate"] is not None:
                arrays[f"{key}_covariate"] = res["covariate"]
            for film, (r, v) in res["tables"].items():
                arrays[f"{key}_{film}_real"], arrays[f"{key}_{film}_values"] = r, v
        np.savez_compressed(OUT_DIR / f"T_bc64_{scenario}_follow.npz", edges=np.array(EDGES), **arrays)


def run_step(step):
    """Simulate and analyse every study of one step."""
    seeds = spawn_seeds(s13.BASE_SEED, len(s13.SCENARIOS) * s13.N_REP_MAX)
    tasks = [{"step": step, "scenario": s, "rep": r, "seed": seeds[list(s13.SCENARIOS).index(s) * s13.N_REP_MAX + r]}
             for s, n in STEPS[step].items() for r in range(n)]
    assert all(t["rep"] < s13.N_REP_MAX for t in tasks)
    start = time.time()
    results = run_parallel(study, tasks, N_JOBS)
    write_step(step, tasks, results)
    seconds = np.array([r["seconds"] for r in results])
    print(f"step {step}: {len(tasks)} studies in {time.time() - start:.0f} s with {N_JOBS} processes, {seconds.mean():.1f} s per study")
    return {"step": step, "studies": len(tasks), "wall_seconds": time.time() - start, "seconds_per_study": float(seconds.mean()),
            "seed_index": {s: [list(s13.SCENARIOS).index(s) * s13.N_REP_MAX, list(s13.SCENARIOS).index(s) * s13.N_REP_MAX + n - 1]
                           for s, n in STEPS[step].items()}}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def rate_row(flags):
    """Rate, Wilson 95% interval, n, old band verdict and whether the interval contains alpha."""
    k, n = int(np.sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return {"rate": k / n, "ci_low": low, "ci_high": high, "n": n, "in_band": bool(FWER_BAND[0] <= k / n <= FWER_BAND[1]),
            "ci_contains_alpha": bool(low <= ALPHA <= high)}


def reverse_of(edges):
    """Reverse edges of a list of 'a->b' names."""
    return [f"{b}->{a}" for a, b in (e.split("->") for e in edges)]


def power_table(raw):
    """Step 1: detection of planted terms / edges, family-wise rate of non-planted terms, leakage into null edges."""
    raw = raw.assign(detected=raw.p < ALPHA)
    rows = []
    for (scenario, method, statistic), sub in raw.groupby(["scenario", "method", "statistic"]):
        label = method if method in EXISTING else f"{method}+{statistic}"
        base = {"estimator": ESTIMATOR, "scenario": scenario, "method": label}
        if method not in EXISTING:
            planted = s13.planted_terms(scenario)
            omni = sub[sub.scope == "omnibus"]
            for term, t in omni.groupby("term"):
                rows.append(dict(base, kind="omnibus term", scope="omnibus", term=term, edge="all", null_cell=term not in planted,
                                 **rate_row(t.detected)))
            null_terms = omni[~omni.term.isin(planted)]
            if not null_terms.empty:
                rows.append(dict(base, kind="non-planted omnibus family", scope="omnibus", term="+".join(sorted(null_terms.term.unique())),
                                 edge="all", null_cell=False,           # union of uncorrected omnibus tests: not a nominal-alpha cell
                                 **rate_row(null_terms.groupby("rep").detected.any())))
            edge_tests = [(t, s13.FL_PLANTED_KEY[t], sub[(sub.scope == "per-edge max-F") & (sub.term == t)]) for t in EDGE_TERMS]
        else:
            edge_df = sub.groupby(["rep", "edge"], as_index=False).detected.any()                # per-film tests: any film
            edge_tests = [(method, EXISTING[method], edge_df)]
        for term, key, e in edge_tests:
            planted = s13.planted_effects(scenario)[key]
            for edge in planted:
                rows.append(dict(base, kind="planted edge", scope="per-edge", term=term, edge=edge, null_cell=False,
                                 **rate_row(e[e.edge == edge].detected)))
            scopes = [("leakage, all null edges", planted)]
            if ESTIMATOR == "dDTF" and planted:
                scopes.append(("leakage, null edges except reverse of planted", planted + reverse_of(planted)))
            for kind, excluded in scopes:
                inherited = ESTIMATOR == "dDTF" and planted and kind == "leakage, all null edges"   # D6: reverse edges not null
                rows.append(dict(base, kind=kind, scope="per-edge", term=term, edge="family", null_cell=not inherited,
                                 **rate_row(e[~e.edge.isin(excluded)].groupby("rep").detected.any())))
    return pd.DataFrame(rows)


def grid_table(raw, method_cols):
    """Rejection rate per scenario, method columns, scope and term (per-edge max-F: family-wise), all null cells."""
    raw = raw.assign(detected=raw.p < ALPHA)
    rows = []
    for keys, sub in raw.groupby(["scenario"] + method_cols + ["scope", "term"]):
        rows.append({"estimator": ESTIMATOR, **dict(zip(["scenario"] + method_cols + ["scope", "term"], keys)), "null_cell": True,
                     **rate_row(sub.groupby("rep").detected.any())})
    return pd.DataFrame(rows)


def within_tables(raw, vectors):
    """Step 4: within-term and film-contrast rates per DV; across-replicate variance ratios of the intercept contrasts."""
    raw = raw.assign(detected=raw.p < ALPHA)
    rows = []
    for (scenario, dv, scope, term, statistic), sub in raw.groupby(["scenario", "dv", "scope", "term", "statistic"]):
        rows.append({"estimator": ESTIMATOR, "scenario": scenario, "dv": dv, "scope": scope, "term": term, "statistic": statistic,
                     "null_cell": not (term == "E" and dv == "T_real"), **rate_row(sub.groupby("rep").detected.any())})
    var = vectors.groupby(["scenario", "dv", "term", "column"]).value.var().groupby(["scenario", "dv", "term"]).sum().unstack("dv")
    ratios = pd.DataFrame({"var_delta_over_T_real": var["delta"] / var["T_real"],
                           "var_null_offset_over_T_real": var["null_median_offset"] / var["T_real"],
                           "var_pooled_dv_over_T_real": var["T_real_minus_pooled"] / var["T_real"],
                           "var_T_real": var["T_real"]}).reset_index()
    ratios.insert(0, "estimator", ESTIMATOR)
    return pd.DataFrame(rows), ratios


def power_criteria(power):
    """[proposed] power criteria of renull + ss_hyp (fl spec targets); omnibus GxF in e reported only."""
    criteria, values = {}, {}
    descriptive = s13.DESCRIPTIVE_ONLY.get(ESTIMATOR, [])
    p = power[power.method == "renull+ss_hyp"]

    def get(scenario, kind, term, edge=None):
        sub = p[(p.scenario == scenario) & (p.kind == kind) & (p.term == term)]
        sub = sub if edge is None else sub[sub.edge == edge]
        assert len(sub) == 1, (scenario, kind, term, edge)
        return sub.iloc[0]

    def put(key, text, row, label="[proposed]", descriptive_key=None):
        values[key] = {k: row[k] for k in ("rate", "ci_low", "ci_high", "n")}
        criteria[f"{label} {text}"] = "descriptive" if descriptive_key in descriptive else bool(row["rate"] >= POWER_TARGET)

    present = set(p.scenario)
    if "c" in present:
        put("follow_c_GxE", f"follow-up scenario c: omnibus GxE detected >= {POWER_TARGET:.0%} (renull + ss_hyp)", get("c", "omnibus term", "GxE"))
        put("follow_c_edge_G", f"follow-up scenario c: per-edge max-F G detects the planted edge >= {POWER_TARGET:.0%} (renull + ss_hyp)",
            get("c", "planted edge", "G", s13.planted_effects("c")["group"][0]))
    if "e" in present:
        put("follow_e_edge_GxF", f"follow-up scenario e: per-edge max-F GxF detects the planted edge >= {POWER_TARGET:.0%} (renull + ss_hyp)",
            get("e", "planted edge", "GxF", s13.planted_effects("e")["interaction"][0]))
        put("follow_e_GxF", f"follow-up scenario e: omnibus GxF detected >= {POWER_TARGET:.0%} (renull + ss_hyp)", get("e", "omnibus term", "GxF"),
            label=s13.REPORT_ONLY)
    if "g" in present:
        put("follow_g_Gxdirection", f"follow-up scenario g: planned 1-df G x direction detected >= {POWER_TARGET:.0%} (renull + ss_hyp; "
            "descriptive only for the measure whose reverse edges inherit the forward coupling, D6)", get("g", "omnibus term", "Gxdirection"),
            descriptive_key="g")
    return criteria, values


def summarize():
    """Build every follow-up table from the raw files and merge the power criteria into summary.json."""
    def read(step, key):
        path = OUT_DIR / f"S_13_follow_{step}_{key}.csv"
        return pd.read_csv(path) if path.exists() else None

    power_raw, precision_raw, j_raw = read("power", "rows"), read("precision", "rows"), read("j", "rows")
    if power_raw is not None:
        power = power_table(power_raw)
        power.to_csv(OUT_DIR / "S_13_follow_power.csv", index=False)
        criteria, values = power_criteria(power)
        path = OUT_DIR / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else {"mode": MODE, "settings": {"n_rep": None}, "criteria": {}, "criterion_values": {}}
        summary["criteria"].update(criteria)
        summary["criterion_values"].update(values)
        path.write_text(json.dumps(summary, indent=2, default=float))
        print(criteria)
    if precision_raw is not None:
        groups = precision_raw[precision_raw.method != "within"]
        grid_table(groups.rename(columns={"method": "variant"}), ["variant", "statistic"]).to_csv(OUT_DIR / "S_13_follow_precision.csv", index=False)
        within, ratios = within_tables(precision_raw[precision_raw.method == "within"], read("precision", "vectors"))
        within.to_csv(OUT_DIR / "S_13_follow_within.csv", index=False)
        ratios.to_csv(OUT_DIR / "S_13_follow_within_var.csv", index=False)
    if j_raw is not None:
        grid_table(j_raw, ["method", "statistic"]).to_csv(OUT_DIR / "S_13_follow_j.csv", index=False)
        ss = read("j", "ss").assign(ratio=lambda d: d.ss_err_perm_mean / d.ss_err_obs)
        table = ss.groupby(["scenario", "method"]).ratio.agg(median_ss_err_perm_over_obs="median", n="size").reset_index()
        table.insert(0, "estimator", ESTIMATOR)
        table.to_csv(OUT_DIR / "S_13_follow_j_ssratio.csv", index=False)


def main():
    """Run the requested step(s) and the summaries."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_progress.begin(f"S13 fl follow-up {ESTIMATOR} {STEP}", HERE / "Results", mode=MODE)
    runtime = []
    for step in (["power", "precision", "j"] if STEP == "all" else [STEP] if STEP != "summary" else []):
        runtime.append(run_step(step))
    if runtime:
        path = OUT_DIR / "S_13_follow_runtime.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update({r["step"]: r for r in runtime})
        path.write_text(json.dumps(previous, indent=2))
    summarize()
    sim_progress.finish(OUT_DIR, {"steps": STEP}, {})


if __name__ == "__main__":
    main()
