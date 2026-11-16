"""S13 film-contrast tests without the shared-offset problem (A/B/C), and group terms with coupling present.

Step 1 (film tests). Per-edge film contrast (film 1 minus the mean of films 2-3; max-T over the 8 edges, standardized,
step-down) and the omnibus F and F x E terms (unstudentized: tr SS_hyp), all with the S5.3 group-level statistic (the
unweighted mean of the two group means):
  A  sign flip of delta (delta = T_real - within-group null median; the current test). Reference. Omnibus terms by
     Freedman-Lane with the group in Z (`freedman_lane`, ss_hyp); the per-edge contrast by plain sign flips
     (`balanced_mean_maxT`, = `signflip_maxT`). Both use the sign-flip matrix of the fl run (generator (seed, 2)).
  B  partner re-pairing of the raw pairing table T (diagonal = real dyads): children re-paired with the caregivers of
     their own group (`caregiver_permutations`, generator (seed, 3)); the statistic is the group-balanced mean of the
     re-paired pairs minus the group-balanced mean of all within-group pairs (`all_pairs_mean`, invariant under
     re-pairing, so it only centres the two-sided test). Omnibus terms with `intercept_ss`.
  C  as A, with the within-group null median of every rearrangement replaced by a person-level bootstrap of the group
     (`bootstrap_null_medians`, generator (seed, 4)): the rearranged data get the shift -(median_boot - median_obs).
     Approximate (the bootstrap ignores the dependence between a dyad's delta and its group median).
  Scenarios: a, h (null; float64 tables T_bc64_<s>_follow.npz, no re-simulation), b, k (null for film terms;
  simulated), f (power, simulated; replicates 0-99 reproduce the fl run), e (power for a one-group film effect, 100).
Step 2 (group terms with coupling present). On the b and f studies: group terms with renull + ss_hyp, renull + F and
  fixed + F (`S13_fl_followup.group_term_rows`), the existing S13 tests (`S13_fl_followup.existing_rows`), and for every
  group term the SS_hyp and SS_err of the renull test at the observed labels and over the relabellings.

Seeds: the S13 scheme, seed index = scenario index * N_REP_MAX + replicate (scenario k has index 10, appended after j, so
no existing seed changes). Reproduction checks: delta from the a/h tables equals the fl parquet (replicates 0-199) and
A equals the stored follow-up Step 4 p-values (all 1000); simulated fl replicates (e, f < 100) are checked by
`S13_fl_followup.simulate`, their group-term F p-values against the fl run and the existing tests against the stored
max-T run (b, f < 100).

Usage: python S13_film_tests.py full [dDTF|GPDC] [all|summary]
  SIM_NREP=<n>         replicates per scenario (smoke test); SIM_NJOBS=<n> processes (default 8)
  SIM_SCENARIOS=a,b    only these scenarios
  SIM_OUT_DIR=<path>   write here instead of Results/<estimator>/S_13 (stored runs are always read from Results)
Writes S_13_film_rows.csv, S_13_group_rows.csv, S_13_group_ss.csv (per study) and, with the summary,
S_13_film_tests.csv, S_13_group_bf.csv, S_13_film_runtime.json and the [proposed] criteria (merged into summary.json).
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
import S13_fl_diagnostics as diag
import S13_fl_followup as fu
from src.perm_anova import (all_pairs_mean, balanced_mean_maxT, bootstrap_null_medians, freedman_lane, intercept_ss,
                            labels_from_permutations, repaired_values, ss_components)
from src.sim_harness import run_parallel, spawn_seeds
from src.sim_inference import caregiver_permutations, family_pvalues, group_null_medians, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE, ESTIMATOR = sys.argv[1], sys.argv[2]
STEP = sys.argv[3] if len(sys.argv) > 3 else "all"
assert MODE == "full" and STEP in ("all", "summary")
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
STORED_DIR = HERE / "Results" / ESTIMATOR / "S_13"
OUT_DIR = Path(os.environ["SIM_OUT_DIR"]) if os.environ.get("SIM_OUT_DIR") else STORED_DIR
N_REP = {"a": 1000, "h": 1000, "b": 1000, "k": 1000, "f": 1000, "e": 100}
if os.environ.get("SIM_NREP"):
    N_REP = {s: min(n, int(os.environ["SIM_NREP"])) for s, n in N_REP.items()}
if os.environ.get("SIM_SCENARIOS"):
    N_REP = {s: n for s, n in N_REP.items() if s in os.environ["SIM_SCENARIOS"].split(",")}
FROM_TABLES = {"a", "h"}                    # read from T_bc64_<s>_follow.npz
FL_DELTA_REPS = 200                         # a / h replicates whose delta is in the fl parquet
GROUP_SCENARIOS = {"b", "f"}                # Step 2
FILM_NULL = ["a", "h", "b", "k"]
FILM_POWER = ["f", "e"]
FILM_TERMS = ["F", "FxE"]
METHODS = ["A", "B", "C"]
METHOD_LABELS = {"A": "A: sign flip of delta", "B": "B: partner re-pairing of T", "C": "C: sign flip, bootstrap offset"}
PLANTED_EDGE = "cg:ROI->child:ROI"          # film effect of e and f
REVERSE_EDGE = "child:ROI->cg:ROI"          # inherits the forward coupling in dDTF (D6)
GROUP_METHODS = [("renull", "ss_hyp"), ("renull", "F")]
EXISTING = list(fu.EXISTING)
N_PERM = s13.N_PERM                         # sign flips (A, C), re-pairings (B), bootstrap replicates (C)
ALPHA, FWER_BAND, POWER_TARGET = s13.ALPHA, s13.FWER_BAND, s13.POWER_TARGET
POWER_MARGIN = 0.05                         # decision rule: power of a candidate at most this far below A
P_ATOL = diag.STORED_P_ATOL
EDGES = list(s13.EDGES)
NF, NE = len(s13.FILMS), len(EDGES)


# ---------------------------------------------------------------------------
# Per-study analyses
# ---------------------------------------------------------------------------
_CACHE = {}


def stored(name):
    """Stored inputs read once per process: the a/h float64 tables and the follow-up Step 4 rows."""
    if name not in _CACHE:
        if name == "precision":
            raw = pd.read_csv(STORED_DIR / "S_13_follow_precision_rows.csv")
            _CACHE[name] = raw[(raw.method == "within") & (raw.dv == "delta")].set_index(["scenario", "rep"]).sort_index()
        else:
            _CACHE[name] = np.load(STORED_DIR / f"T_bc64_{name}_follow.npz")
    return _CACHE[name]


def tables_study(task):
    """Tables of a stored a/h study; delta (complete-case dyads) checked against the fl parquet."""
    npz = stored(task["scenario"])
    key = f"rep{task['rep']:04d}"
    groups = npz[f"{key}_groups"]
    tables = {film: (npz[f"{key}_{film}_real"], npz[f"{key}_{film}_values"]) for film in s13.FILMS}
    real, pair_values = fu.stacked(tables)
    used = np.isfinite(real).all(axis=1)
    medians = group_null_medians(pair_values, (groups == 1)[None])[0]
    cells = (real - medians[groups])[used]
    if task["rep"] < FL_DELTA_REPS:
        saved = diag.saved_run()[0].loc[(task["scenario"], task["rep"])]
        wide = np.full((len(groups), NF, NE), np.nan)
        wide[saved.dyad.to_numpy(), [s13.FILMS.index(f) for f in saved.film], [EDGES.index(e) for e in saved.edge]] = saved.delta.to_numpy()
        assert np.array_equal(wide.reshape(len(groups), -1)[used], cells), f"delta of the tables differs from the fl parquet: {task}"
    return None, groups, tables, cells, used


def film_contrast(Y):
    """Film 1 minus the mean of films 2-3 per edge, from film-major cells (..., 24) -> (..., 8)."""
    wide = Y.reshape(Y.shape[:-1] + (NF, NE))
    return wide[..., 0, :] - wide[..., 1:, :].mean(axis=-2)


def film_rows(cells, real, pair_values, groups, flips, seed):
    """Step 1: per-edge film contrast (max-T) and omnibus F, FxE (tr SS_hyp) with methods A, B, C."""
    X, Z = s13.fl_design(groups, None)["within"]
    terms = {t: s13.FL_OMNIBUS[t][0] for t in FILM_TERMS}
    medians = group_null_medians(pair_values, (groups == 1)[None])[0]
    assert np.allclose(cells, real - medians[groups], rtol=0, atol=1e-12), "delta must be T_real minus the within-group null median"
    out = {}
    # A: sign flip of delta
    contrast = film_contrast(cells)
    out["A"] = (balanced_mean_maxT(contrast, flips[:, :, None] * contrast[None], groups)["maxT"],
                {t: freedman_lane(cells @ C, X, Z, flips, "signflip", "ss_hyp") for t, C in terms.items()})
    # B: partner re-pairing of T
    table = pair_values.copy()
    table[np.arange(len(groups)), np.arange(len(groups))] = real
    perms = caregiver_permutations(len(groups), np.random.default_rng([seed, 3]), N_PERM, groups)
    centre = all_pairs_mean(table, groups)[groups]
    observed, null = real - centre, repaired_values(table, perms) - centre[None]
    out["B"] = (balanced_mean_maxT(film_contrast(observed), film_contrast(null), groups)["maxT"],
                {t: (float(intercept_ss(observed @ C, groups)), intercept_ss(null @ C, groups)) for t, C in terms.items()})
    # C: sign flip with the null median resampled by a person-level bootstrap
    boot = bootstrap_null_medians(pair_values, groups, np.random.default_rng([seed, 4]), N_PERM)
    shifts = -(boot - medians[None])[:, groups, :]                                       # (B, n, 24)
    out["C"] = (balanced_mean_maxT(contrast, flips[:, :, None] * contrast[None] + film_contrast(shifts), groups)["maxT"],
                {t: freedman_lane(cells @ C, X, Z, flips, "signflip", "ss_hyp", shifts=shifts @ C) for t, C in terms.items()})
    rows = []
    for method, (edge_p, omnibus) in out.items():
        rows += [{"method": method, "scope": "per-edge max-T", "term": "film contrast", "edge": e, "p": float(p)}
                 for e, p in zip(EDGES, edge_p)]
        rows += [{"method": method, "scope": "omnibus", "term": t, "edge": "all",
                  "p": float(family_pvalues(np.array([obs]), null[:, None])["unadjusted"][0])} for t, (obs, null) in omnibus.items()]
    return rows


def check_A(rows, task):
    """Assert that method A reproduces the stored follow-up Step 4 p-values (delta; film contrast, F and FxE ss_hyp)."""
    sub = stored("precision").loc[(task["scenario"], task["rep"])]
    for r in rows:
        if r["method"] != "A":
            continue
        if r["scope"] == "omnibus":
            hit = sub[(sub.scope == "omnibus") & (sub.term == r["term"]) & (sub.statistic == "ss_hyp")]
        else:
            hit = sub[(sub.scope == "per-edge max-T") & (sub.edge == r["edge"])]
        assert len(hit) == 1 and abs(hit.p.iloc[0] - r["p"]) <= P_ATOL, f"A differs from the stored Step 4 p-value: {task}, {r}"


def group_ss_rows(cells, groups, perms, renulled):
    """Step 2: SS_hyp and SS_err of every renull group test at the observed labels and averaged over the relabellings."""
    observed = groups == 1
    labels = np.vstack([observed[None], labels_from_permutations(observed, perms)])
    Y = renulled(labels)
    Z = np.ones((len(groups), 1))
    Xb = np.where(labels, 1.0, -1.0)[:, :, None]
    per_edge = Y.reshape(Y.shape[:2] + (NF, NE))
    tests = [("omnibus", t, "all", Y @ s13.FL_OMNIBUS[t][0]) for t in diag.GROUP_TERMS]
    tests += [("per-edge max-F", t, e, per_edge[..., j] @ s13.FL_EDGE[t][0]) for t in diag.EDGE_TERMS for j, e in enumerate(EDGES)]
    rows = []
    for scope, term, edge, y in tests:
        hyp, err = ss_components(y, Xb, Z)
        rows.append({"scope": scope, "term": term, "edge": edge, "ss_hyp_obs": hyp[0], "ss_hyp_perm_mean": hyp[1:].mean(),
                     "ss_err_obs": err[0], "ss_err_perm_mean": err[1:].mean()})
    return rows


def study(task):
    """One study: film tests (all scenarios) and group terms (b, f)."""
    start = time.time()
    if task["scenario"] in FROM_TABLES:
        df, groups, tables, cells, used = tables_study(task)
    else:
        df, groups, tables, _, cells, used = fu.simulate(task)
    n_used = int(used.sum())
    real, pair_values = fu.stacked(tables)
    perms, flips = fu.rearrangements(task["seed"], n_used)
    groups_used = groups[used]
    film = film_rows(cells, real[used], pair_values[np.ix_(used, used)], groups_used, flips, task["seed"])
    if task["scenario"] in FROM_TABLES:
        check_A(film, task)
    group, ss = [], []
    if task["scenario"] in GROUP_SCENARIOS:

        def renulled(labels):
            """Re-nulled delta of the used dyads; left-out dyads keep their observed group in every labelling."""
            full = np.tile(groups == 1, (len(labels), 1))
            full[:, used] = labels
            return fu.renull_delta(real, pair_values, full)[:, used, :]

        in_fl = task["rep"] < fu.FL_REPS.get(task["scenario"], 0)
        group = fu.group_term_rows(cells, groups_used, perms, renulled, task, in_fl)
        existing = fu.existing_rows(df, groups, tables, task)
        fu.check_existing(existing, task)
        group += existing
        ss = group_ss_rows(cells, groups_used, perms, renulled)
    tag = {"scenario": task["scenario"], "rep": task["rep"], "seed": task["seed"], "n_dyads": n_used}
    return {"film": [dict(r, **tag) for r in film], "group": [dict(r, **tag) for r in group],
            "ss": [dict(r, **tag) for r in ss], "seconds": time.time() - start}


def run():
    """Analyse every study and write the per-study rows and the runtime."""
    seeds = spawn_seeds(s13.BASE_SEED, len(s13.SCENARIOS) * s13.N_REP_MAX)
    index = {s: list(s13.SCENARIOS).index(s) for s in N_REP}
    tasks = [{"scenario": s, "rep": r, "seed": seeds[index[s] * s13.N_REP_MAX + r]} for s, n in N_REP.items() for r in range(n)]
    assert all(t["rep"] < s13.N_REP_MAX for t in tasks)
    start = time.time()
    results = run_parallel(study, tasks, N_JOBS)
    wall = time.time() - start
    for key, name in (("film", "S_13_film_rows"), ("group", "S_13_group_rows"), ("ss", "S_13_group_ss")):
        rows = [r for res in results for r in res[key]]
        if rows:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{name}.csv", index=False)
    seconds = pd.DataFrame({"scenario": [t["scenario"] for t in tasks], "seconds": [r["seconds"] for r in results]})
    runtime = {"wall_seconds": wall, "processes": N_JOBS, "studies": len(tasks),
               "seconds_per_study": seconds.groupby("scenario").seconds.mean().round(2).to_dict(),
               "seed_index": {s: [index[s] * s13.N_REP_MAX, index[s] * s13.N_REP_MAX + n - 1] for s, n in N_REP.items()}}
    (OUT_DIR / "S_13_film_runtime.json").write_text(json.dumps(runtime, indent=2))
    print(f"{len(tasks)} studies in {wall:.0f} s with {N_JOBS} processes")


# ---------------------------------------------------------------------------
# Summaries and [proposed] criteria
# ---------------------------------------------------------------------------
def classify(k, n):
    """Wilson 95% interval and calibration class of a null rejection rate (liberal: lower bound above alpha)."""
    lo, hi = wilson_interval(k, n)
    return lo, hi, "liberal" if lo > ALPHA else "conservative" if hi < ALPHA else "calibrated"


def rate_row(keys, detected, null_cell):
    """One summary row: rate, Wilson interval and (null cells only) class."""
    k, n = int(np.sum(detected)), len(detected)
    lo, hi, cls = classify(k, n)
    return dict(keys, n=n, rate=k / n, ci_low=lo, ci_high=hi, null_cell=null_cell, cls=cls if null_cell else "")


def film_table(raw):
    """Step 1 grid: null rates (a, h, b, k) and power / leakage (f, e) per scenario, method and term."""
    raw = raw.assign(detected=raw.p < ALPHA)
    rows = []
    for (scenario, method), sub in raw.groupby(["scenario", "method"]):
        keys = {"scenario": scenario, "method": method}
        null = scenario in FILM_NULL
        edge = sub[sub.scope == "per-edge max-T"]
        if null:
            fwer = edge.groupby("rep").detected.any()
            rows.append(rate_row(dict(keys, scope="per-edge max-T", term="film contrast", edge="any (FWER)"), fwer, True))
        else:
            planted = edge[edge.edge == PLANTED_EDGE].set_index("rep").detected
            leak = edge[edge.edge != PLANTED_EDGE].groupby("rep").detected.any()
            rows.append(rate_row(dict(keys, scope="per-edge max-T", term="film contrast", edge=PLANTED_EDGE), planted, False))
            rows.append(rate_row(dict(keys, scope="per-edge max-T", term="film contrast", edge="leakage, null edges"), leak, False))
            leak = edge[~edge.edge.isin([PLANTED_EDGE, REVERSE_EDGE])].groupby("rep").detected.any()
            rows.append(rate_row(dict(keys, scope="per-edge max-T", term="film contrast", edge="leakage, null edges except reverse"), leak, False))
        for term in FILM_TERMS:
            rows.append(rate_row(dict(keys, scope="omnibus", term=term, edge="all"),
                                 sub[(sub.scope == "omnibus") & (sub.term == term)].set_index("rep").detected, null))
    return pd.DataFrame(rows)


def group_table(raw, ss):
    """Step 2 grid: null rates of every group term (b, f) per method; SS ratios (renull) for liberal cells."""
    raw = raw.assign(detected=raw.p < ALPHA,
                     method_label=np.where(raw.statistic == "maxT", raw.method, raw.method + " + " + raw.statistic))
    ss = ss.assign(hyp_ratio=ss.ss_hyp_perm_mean / ss.ss_hyp_obs, err_ratio=ss.ss_err_perm_mean / ss.ss_err_obs)
    rows = []
    for (scenario, method), sub in raw.groupby(["scenario", "method_label"]):
        keys = {"scenario": scenario, "method": method}
        for (scope, term), cell in sub.groupby(["scope", "term"]):
            if scope == "omnibus":
                detected, edge = cell.set_index("rep").detected, "all"
            else:
                detected, edge = cell.groupby("rep").detected.any(), "any (FWER)"
            row = dict(rate_row(dict(keys, scope=scope, term=term, edge=edge), detected, True),
                       ss_hyp_ratio_median=np.nan, ss_err_ratio_median=np.nan)
            if row["cls"] == "liberal" and method.startswith("renull"):
                s = ss[(ss.scenario == scenario) & (ss.scope == scope) & (ss.term == term)]
                row["ss_hyp_ratio_median"], row["ss_err_ratio_median"] = s.hyp_ratio.median(), s.err_ratio.median()
            rows.append(row)
    return pd.DataFrame(rows)


def criteria(film, group):
    """[proposed] decision-rule criteria of this step: (criteria, criterion_values), merged into summary.json."""
    crit, values = {}, {}
    nulls = film[film.null_cell]
    power = {s: film[(film.scenario == s) & (film.edge == PLANTED_EDGE)].set_index("method").rate for s in FILM_POWER}
    for method in METHODS:
        liberal = nulls[(nulls.method == method) & (nulls.cls == "liberal")]
        key = f"[proposed] film tests, method {method}: no liberal null cell in a, h, b, k (per-edge max-T, F, FxE; Wilson 95% CI)"
        crit[key] = bool(liberal.empty)
        values[key] = ", ".join(f"{r.scenario} {r.term}" for r in liberal.itertuples()) or "none"
        f, a = power["f"][method], power["f"]["A"]
        key = f"[proposed] film tests, method {method}: power on the planted edge in f >= {POWER_TARGET} and >= A - {POWER_MARGIN}"
        crit[key], values[key] = bool(f >= POWER_TARGET and f >= a - POWER_MARGIN), {"rate": f, "rate_A": a}
        key = f"[proposed: report only] film tests, method {method}: power on the planted edge in e (one-group film effect)"
        crit[key], values[key] = float(power["e"][method]), {"rate": power["e"][method]}
    renull = group[group.method == "renull + ss_hyp"]
    liberal = renull[renull.cls == "liberal"]
    key = "[proposed] group terms with coupling present, renull + ss_hyp: no liberal null cell in b, f (Wilson 95% CI)"
    crit[key] = bool(liberal.empty)
    values[key] = ", ".join(f"{r.scenario} {r.term} {r.scope}" for r in liberal.itertuples()) or "none"
    return crit, values


def summary():
    """Write S_13_film_tests.csv, S_13_group_bf.csv and merge the [proposed] criteria into summary.json."""
    film = film_table(pd.read_csv(OUT_DIR / "S_13_film_rows.csv"))
    film.to_csv(OUT_DIR / "S_13_film_tests.csv", index=False)
    group = group_table(pd.read_csv(OUT_DIR / "S_13_group_rows.csv"), pd.read_csv(OUT_DIR / "S_13_group_ss.csv"))
    group.to_csv(OUT_DIR / "S_13_group_bf.csv", index=False)
    if set(FILM_POWER) <= set(film.scenario) and set(GROUP_SCENARIOS) <= set(group.scenario):
        path = OUT_DIR / "summary.json"
        data = json.loads(path.read_text()) if path.exists() else {"mode": MODE, "settings": {"n_rep": None}, "criteria": {}, "criterion_values": {}}
        crit, values = criteria(film, group)
        data["criteria"].update(crit)
        data.setdefault("criterion_values", {}).update(values)
        path.write_text(json.dumps(data, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(film.to_string(index=False))
    print(group.to_string(index=False))


def main():
    """Run the studies (`all`) and/or rebuild the summaries."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if STEP == "all":
        run()
    summary()


if __name__ == "__main__":
    main()
