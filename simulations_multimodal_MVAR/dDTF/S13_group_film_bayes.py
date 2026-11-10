"""S13 - Group and film effects across edges: permutation max-T vs Bayesian partial pooling (Table 2, S13).

Question: how do the per-edge permutation tests with max-T and Bayesian hierarchical models compare in estimating and
detecting group and film effects, and do partial-pooling models move an effect from one edge to the others?
Design: 2 groups x 15 dyads, 3 films x 60 s. Per-person node parameters are drawn once and kept across films; every
film has new innovations. 4-node Level-1 model; the family is the 8 cross-person edges. Scenarios (VAR gain, brain->brain
unless stated): a none; b cg brain->child brain 0.3 in all cells; c as b, ASD 0.1; d c + cg heart->child heart 0.3 TD /
0.1 ASD; e group difference (0.3 TD vs 0.1 ASD) only in film 1 (films 2, 3: 0.3 in both groups).
Dependent variable: delta against the within-group null of each film (surrogates built from dyads of the same group and
film), standardized within edge for the Bayesian models (`src.group_model.standardize_within_edge`); the robust z
(delta / scaled MAD of the null) is a sensitivity DV for the permutation tests.
Analyses per study:
  1. permutation: group contrast per edge and film, max-T over the 8 edges within each film; max-T over the 24
     edge x film tests; max-T over the 8 edges on the film-averaged delta;
  2. Bayesian per edge:            dv ~ film * group + (1|dyad)
  3. Bayesian, pooled intercepts:  dv ~ film * group + (1|edge) + (1|dyad)
  4. Bayesian, pooled effects:     dv ~ film * group + (film * group | edge) + (1|dyad); if a fit fails the convergence
     gate, the fallback (group | edge) model is fitted as well and reported separately (no silent refits).
Contrasts (95% HDI excludes 0 = detected): group effect (ASD - TD, mean over films), film contrast (film 1 - mean of
films 2 and 3, mean over groups), interaction (film contrast ASD - TD).
Convergence gate (`src.group_model.convergence_row`): R-hat < 1.01, bulk/tail ESS > 400, no divergences.

Usage: python S13_group_film_bayes.py [quick|full] [dDTF|GPDC] [summarize]
  SIM_NREP=<n>        replicates per scenario (default: SETTINGS)
  SIM_REPS=<a>-<b>    run only replicates a..b (inclusive) and write raw_results_reps_<a>-<b>.csv; the summary is built
                      from every raw_results_reps_*.csv in the folder ('summarize' rebuilds it without simulating)
  SIM_NJOBS=<n>       parallel processes (default 8)
Results: Results/<estimator>/S_13/
"""

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")            # one thread per worker; studies run in parallel
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import bambi as bmb

from src.design import window_geometry
from src.group_model import convergence_row, fit_model_with_priors, standardize_within_edge
from src.sim_generators import sample_node_params, simulate_dyads_fixed_params, zscore_designs
from src.sim_group_bayes import edge_effects, excludes_zero, population_effects
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import delta_and_z_all, group_permutation_maxT, real_matrices, surrogate_values, wilson_interval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
SUMMARIZE_ONLY = len(sys.argv) > 3 and sys.argv[3] == "summarize"
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
SIM_ID = 13
BASE_SEED = 20260113
N_REP_MAX = 1000                      # seeds are indexed (scenario, replicate) up to this many replicates
MCMC_FULL = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.95, "cores": 1, "sd_prior_edge": "halfstudentt"}
SETTINGS = {
    "quick": {"n_rep": 1, "n_per_group": 6, "scenarios": ["a", "d"],
              "mcmc": dict(MCMC_FULL, draws=150, tune=150, chains=2)},
    "full": {"n_rep": 100, "n_per_group": 15, "scenarios": ["a", "b", "c", "d", "e"], "mcmc": MCMC_FULL},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
REPS = range(SETTINGS["n_rep"])
if os.environ.get("SIM_REPS"):
    first, last = (int(x) for x in os.environ["SIM_REPS"].split("-"))
    REPS = range(first, last + 1)

FS = 2.5
SEG_S = 60
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)                 # [TBD: confirm]
BOX_COX = 0.25
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
POLE_RADIUS_RANGES = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0_RANGES = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
N_LAGS = 3
NODES, CHILD_NODES = [0, 1, 2, 3], [0, 2]
STABILITY_MAX_ROOT = 1.0
FILMS = ["F1", "F2", "F3"]            # F1 is the contrasted film ("film 1")
GROUPS = ["TD", "ASD"]                # group effect = ASD - TD
EDGES = {"cg:ROI->child:ROI": (1, 0), "child:ROI->cg:ROI": (0, 1), "cg:HRV->child:HRV": (3, 2),
         "child:HRV->cg:HRV": (2, 3), "cg:HRV->child:ROI": (3, 0), "child:HRV->cg:ROI": (2, 1),
         "cg:ROI->child:HRV": (1, 2), "child:ROI->cg:HRV": (0, 3)}   # the family: all 8 cross-person edges
# scenario -> {edge name: gains[group][film]} (group order GROUPS, film order FILMS; lag 1)
SCENARIOS = {
    "a": {},
    "b": {"cg:ROI->child:ROI": [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]},
    "c": {"cg:ROI->child:ROI": [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]},
    "d": {"cg:ROI->child:ROI": [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]], "cg:HRV->child:HRV": [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]]},
    "e": {"cg:ROI->child:ROI": [[0.3, 0.3, 0.3], [0.1, 0.3, 0.3]]},
}
ALPHA = 0.05
N_PERM = 1000
HDI_PROB = 0.95
GATE = {"rhat_max": 1.01, "ess_min": 400, "pareto_k_max": np.inf}   # Pareto k is not part of this gate
FIXED_PRIOR = bmb.Prior("Normal", mu=0, sigma=1)
SD_PRIOR = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))
PRIORS = {"Intercept": FIXED_PRIOR, "C(film, Sum)": FIXED_PRIOR, "C(group, Sum)": FIXED_PRIOR,
          "C(film, Sum):C(group, Sum)": FIXED_PRIOR, "1|dyad_id": SD_PRIOR, "1|edge": SD_PRIOR,
          "C(film, Sum)|edge": SD_PRIOR, "C(group, Sum)|edge": SD_PRIOR, "C(film, Sum):C(group, Sum)|edge": SD_PRIOR,
          "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1)}
FIXED = "dv ~ C(film, Sum) * C(group, Sum)"
MODELS = {   # label -> (formula, required group-specific prior keys)
    "bayes_edge": (f"{FIXED} + (1|dyad_id)", []),
    "bayes_pooled_int": (f"{FIXED} + (1|edge) + (1|dyad_id)", ["1|edge"]),
    "bayes_pooled_eff": (f"{FIXED} + (C(film, Sum) * C(group, Sum) | edge) + (1|dyad_id)",
                         ["1|edge", "C(film, Sum)|edge", "C(group, Sum)|edge", "C(film, Sum):C(group, Sum)|edge"]),
    "bayes_pooled_eff_fallback": (f"{FIXED} + (C(group, Sum) | edge) + (1|dyad_id)", ["1|edge", "C(group, Sum)|edge"]),
}
METHOD_LABELS = {"perm_film_unadj": "permutation, per film, uncorrected", "perm_film_maxT": "permutation, max-T within film",
                 "perm_24_maxT": "permutation, max-T over 24", "perm_mean_maxT": "permutation, max-T on film mean",
                 "bayes_edge": "Bayes per edge (M2)", "bayes_pooled_int": "Bayes (1|edge) (M3)",
                 "bayes_pooled_eff": "Bayes (film*group|edge) (M4)", "bayes_pooled_eff_fallback": "Bayes (group|edge) fallback"}
PLOT_METHODS = ["perm_film_maxT", "perm_24_maxT", "perm_mean_maxT", "bayes_edge", "bayes_pooled_int", "bayes_pooled_eff"]
FWER_BAND = (0.03, 0.07)              # scenario a, max-T within film [proposed]
POWER_TARGET = 0.80                   # scenario c, max-T within film [proposed]
LEAKAGE_MAX = 0.10                    # scenario c, model 4, any null edge [proposed]

warnings.filterwarnings("ignore")
for name in ("pymc", "pytensor", "bambi", "arviz"):
    logging.getLogger(name).setLevel(logging.ERROR)


def planted_effects(scenario):
    """Edges with a planted group effect, film contrast and interaction (from the gains of SCENARIOS)."""
    out = {"group": [], "film": [], "interaction": []}
    for edge, gains in SCENARIOS[scenario].items():
        g = np.asarray(gains)
        contrast = g[:, 0] - g[:, 1:].mean(axis=1)               # film contrast per group
        if not np.isclose(g[1].mean(), g[0].mean()):
            out["group"].append(edge)
        if not np.isclose(contrast.mean(), 0):
            out["film"].append(edge)
        if not np.isclose(contrast[1], contrast[0]):
            out["interaction"].append(edge)
    return out


def simulate_delta_table(rng, scenario, n_per_group, est):
    """Simulate one study (3 films, 2 groups) and return the tidy delta / z table against the within-group null."""
    n = 2 * n_per_group
    groups = np.r_[np.zeros(n_per_group, int), np.ones(n_per_group, int)]
    radii, f0s = sample_node_params(rng, n, POLE_RADIUS_RANGES), sample_node_params(rng, n, F0_RANGES)
    rows = []
    for f, film in enumerate(FILMS):
        edges_per_dyad = [[(*EDGES[edge], 1, gains[groups[d]][f]) for edge, gains in SCENARIOS[scenario].items()]
                          for d in range(n)]
        designs = zscore_designs(simulate_dyads_fixed_params(rng, radii, f0s, int(SEG_S * FS), FS, edges_per_dyad, N_LAGS))
        for g in (0, 1):
            members = np.flatnonzero(groups == g)
            real, roots, _ = real_matrices(designs[members], NODES, est)
            values, _, _ = surrogate_values(designs[members], NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng)
            stable = roots < STABILITY_MAX_ROOT
            delta, z = delta_and_z_all(real[stable], values[stable][:, stable], "pooled")   # within this group and film
            for i, dyad in enumerate(members[stable]):
                for edge, (s, t) in EDGES.items():
                    rows.append({"dyad_id": f"d{dyad:02d}", "dyad": int(dyad), "group": GROUPS[g], "film": film,
                                 "edge": edge, "delta": float(delta[i, t, s]), "z": float(z[i, t, s])})
    return pd.DataFrame(rows), groups


def permutation_records(df, groups, rng, dv):
    """Group-contrast permutation tests of one DV: per film (uncorrected, max-T over 8), max-T over 24, max-T on the film mean."""
    n = len(groups)
    wide = {film: np.full((n, len(EDGES)), np.nan) for film in FILMS}
    for (film, edge), sub in df.groupby(["film", "edge"]):
        wide[film][sub.dyad.to_numpy(), list(EDGES).index(edge)] = sub[dv].to_numpy()
    out = []

    def add(method, film, edge, p):
        out.append({"method": method, "dv": dv, "film": film, "edge": edge, "effect": "group", "p": float(p), "detected": bool(p < ALPHA)})

    for film in FILMS:
        res = group_permutation_maxT(wide[film], groups, rng, N_PERM)
        for e, edge in enumerate(EDGES):
            add("perm_film_unadj", film, edge, res["unadjusted"][e])
            add("perm_film_maxT", film, edge, res["maxT"][e])
    res = group_permutation_maxT(np.hstack([wide[film] for film in FILMS]), groups, rng, N_PERM)
    for f, film in enumerate(FILMS):
        for e, edge in enumerate(EDGES):
            add("perm_24_maxT", film, edge, res["maxT"][f * len(EDGES) + e])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        film_mean = np.nanmean(np.stack([wide[film] for film in FILMS]), axis=0)
    res = group_permutation_maxT(film_mean, groups, rng, N_PERM)
    for e, edge in enumerate(EDGES):
        add("perm_mean_maxT", "mean", edge, res["maxT"][e])
    return out


def bayes_fit(label, data, mcmc, seed):
    """Fit one model; return (model, idata, gate row, seconds)."""
    formula, required = MODELS[label]
    start = time.time()
    model, idata = fit_model_with_priors(formula, data, PRIORS, required, dict(mcmc, seed=seed))
    seconds = time.time() - start
    gate = convergence_row(label, idata, len(data), 0, GATE["rhat_max"], GATE["ess_min"], GATE["pareto_k_max"])
    return model, idata, gate, seconds


def effect_rows(method, edge, effects, gate):
    """Records of one edge's effects (summarize_draws dicts)."""
    return [{"method": method, "dv": "delta_std", "film": "all", "edge": edge, "effect": name, "estimate": s["estimate"],
             "hdi_low": s["hdi_low"], "hdi_high": s["hdi_high"], "detected": excludes_zero(s),
             "gate_pass": gate["pass_l9"]}
            for name, s in effects.items()]


def bayes_records(df, mcmc, seed):
    """Models 2-4 (and the model-4 fallback when model 4 fails the gate) on the within-edge standardized delta."""
    data, _ = standardize_within_edge(df, "delta")
    data = data.rename(columns={"delta_z": "dv"})
    data["film"] = pd.Categorical(data["film"], categories=FILMS)
    data["group"] = pd.Categorical(data["group"], categories=GROUPS)
    data["edge"] = pd.Categorical(data["edge"], categories=list(EDGES))
    records, gates, runtime = [], [], {}

    runtime["bayes_edge"] = 0.0
    for edge in EDGES:
        model, idata, gate, seconds = bayes_fit("bayes_edge", data[data.edge == edge].reset_index(drop=True), mcmc, seed)
        runtime["bayes_edge"] += seconds
        gates.append(dict(gate, edge=edge, seconds=seconds))
        records += effect_rows("bayes_edge", edge, population_effects(model, idata, FILMS, GROUPS, HDI_PROB), gate)

    model, idata, gate, seconds = bayes_fit("bayes_pooled_int", data, mcmc, seed)
    runtime["bayes_pooled_int"] = seconds
    gates.append(dict(gate, edge="all", seconds=seconds))
    common = population_effects(model, idata, FILMS, GROUPS, HDI_PROB)
    records += effect_rows("bayes_pooled_int", "population", common, gate)
    for edge in EDGES:                       # (1|edge) cancels in every contrast: each edge gets the common effects
        records += effect_rows("bayes_pooled_int", edge, common, gate)

    labels = ["bayes_pooled_eff"]
    while labels:
        label = labels.pop(0)
        model, idata, gate, seconds = bayes_fit(label, data, mcmc, seed)
        runtime[label] = seconds
        gates.append(dict(gate, edge="all", seconds=seconds))
        records += effect_rows(label, "population", population_effects(model, idata, FILMS, GROUPS, HDI_PROB), gate)
        for edge, effects in edge_effects(model, idata, list(EDGES), FILMS, GROUPS, data.dyad_id.iloc[0], HDI_PROB).items():
            records += effect_rows(label, edge, effects, gate)
        if label == "bayes_pooled_eff" and not gate["pass_l9"]:
            labels.append("bayes_pooled_eff_fallback")

    # unpooled sample estimate of the standardized group effect per edge (reference for bias / RMSE)
    for edge, sub in data.groupby("edge", observed=True):
        cell = sub.groupby(["group", "film"], observed=True).dv.mean()
        records.append({"method": "sample_mean_diff", "dv": "delta_std", "film": "all", "edge": edge, "effect": "group",
                        "estimate": float(cell["ASD"].mean() - cell["TD"].mean())})
    return records, gates, runtime


def run_task(task):
    """One replicated study of one scenario: simulate, permutation tests, Bayesian models."""
    rng = np.random.default_rng(task["seed"])
    start = time.time()
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    df, groups = simulate_delta_table(rng, task["scenario"], SETTINGS["n_per_group"], est)
    t_sim = time.time() - start
    perm_rng = np.random.default_rng([task["seed"], 1])
    start = time.time()
    records = permutation_records(df, groups, perm_rng, "delta") + permutation_records(df, groups, perm_rng, "z")
    runtime = {"simulation_and_delta": t_sim, "permutation": time.time() - start}
    bayes, gates, bayes_time = bayes_records(df, SETTINGS["mcmc"], int(task["seed"] % 2**31))
    runtime.update(bayes_time)
    tag = {"scenario": task["scenario"], "rep": task["rep"]}
    return ([dict(r, **tag) for r in records + bayes], [dict(g, **tag) for g in gates], [dict(runtime, **tag)],
            [dict(tag, n_rows=len(df), n_dropped_dyad_films=2 * SETTINGS["n_per_group"] * len(FILMS) - len(df) // len(EDGES))])


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def rate_row(flags):
    """Rate, Wilson interval and n of a boolean series."""
    k, n = int(np.sum(flags)), len(flags)
    low, high = wilson_interval(k, n)
    return {"rate": k / n, "ci_low": low, "ci_high": high, "n": n}


def edge_level(df):
    """One detection per (scenario, rep, method, dv, edge, effect): permutation per-film tests count as detected if any film rejects."""
    return df.groupby(["scenario", "rep", "method", "dv", "edge", "effect"], as_index=False).detected.any()


def detection_table(edge_df):
    """Detection rate per scenario, method, DV, edge and effect, with the planted flag."""
    rows = []
    for (scenario, method, dv, edge, effect), sub in edge_df.groupby(["scenario", "method", "dv", "edge", "effect"]):
        rows.append({"scenario": scenario, "method": method, "dv": dv, "edge": edge, "effect": effect,
                     "planted": edge in planted_effects(scenario)[effect], **rate_row(sub.detected)})
    return pd.DataFrame(rows)


def familywise_table(edge_df):
    """Share of studies with at least one detection on an edge without a planted effect (all null edges / without the reverse of planted edges)."""
    rows = []
    for (scenario, method, dv, effect), sub in edge_df.groupby(["scenario", "method", "dv", "effect"]):
        planted = planted_effects(scenario)[effect]
        reverse = {f"{b}->{a}" for a, b in (e.split("->") for e in planted)}
        for scope, excluded in (("all null edges", set(planted)), ("null edges except reverse of planted", set(planted) | reverse)):
            null = sub[~sub.edge.isin(excluded)]
            if null.empty:
                continue
            per_study = null.groupby("rep").detected.any()
            rows.append({"scenario": scenario, "method": method, "dv": dv, "effect": effect, "scope": scope,
                         "n_null_edges": null.edge.nunique(), **rate_row(per_study)})
    return pd.DataFrame(rows)


def shrinkage_table(df):
    """Mean estimate, bias and RMSE of the standardized group effect per edge against the Monte Carlo reference.

    The reference ("true" effect in standardized units) is the mean, over all replicates of the scenario, of the
    unpooled sample estimate (difference of the standardized group means, averaged over films), which is unbiased for
    the standardized estimand; on edges without any planted coupling difference it is ~0 except for measure leakage.
    """
    est = df[(df.effect == "group") & df.estimate.notna() & (df.edge != "population")]
    ref = est[est.method == "sample_mean_diff"].groupby(["scenario", "edge"]).estimate.mean().rename("reference")
    rows = []
    for (scenario, method, edge), sub in est.groupby(["scenario", "method", "edge"]):
        r = ref[(scenario, edge)]
        rows.append({"scenario": scenario, "method": method, "edge": edge, "planted": edge in planted_effects(scenario)["group"],
                     "reference": r, "mean_estimate": sub.estimate.mean(), "bias": sub.estimate.mean() - r,
                     "rmse": float(np.sqrt(np.mean((sub.estimate - r) ** 2))), "sd_estimate": sub.estimate.std(), "n": len(sub)})
    return pd.DataFrame(rows)


def plot_summary(detection, familywise, shrinkage, out_dir):
    """fig_group_film_bayes: A detection on planted edges, B family-wise rate on null edges, C estimated vs true group effect."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    width = 0.8 / len(PLOT_METHODS)
    det = detection[(detection.dv.isin(["delta", "delta_std"])) & detection.planted & (detection.effect == "group")]
    scen_a = sorted(det.scenario.unique())
    for i, method in enumerate(PLOT_METHODS):
        vals = [det[(det.scenario == s) & (det.method == method)].rate.mean() for s in scen_a]
        axes[0].bar(np.arange(len(scen_a)) + i * width, vals, width, label=METHOD_LABELS[method], color=f"C{i}")
    axes[0].set_xticks(np.arange(len(scen_a)) + 0.4 - width / 2); axes[0].set_xticklabels(scen_a)
    axes[0].axhline(POWER_TARGET, color="k", lw=0.6, ls=":"); axes[0].set_ylim(0, 1)
    axes[0].set_title("A  group effect detected on planted edges", fontsize=10); axes[0].set_xlabel("scenario"); axes[0].set_ylabel("detection rate")
    axes[0].legend(fontsize=7)

    fw = familywise[(familywise.dv.isin(["delta", "delta_std"])) & (familywise.effect == "group") & (familywise.scope == "all null edges")]
    scen_b = sorted(fw.scenario.unique())
    for i, method in enumerate(PLOT_METHODS):
        sub = fw[fw.method == method].set_index("scenario").reindex(scen_b)
        axes[1].bar(np.arange(len(scen_b)) + i * width, sub.rate, width, yerr=[sub.rate - sub.ci_low, sub.ci_high - sub.rate],
                    capsize=1.5, color=f"C{i}", error_kw={"lw": 0.6})
    axes[1].set_xticks(np.arange(len(scen_b)) + 0.4 - width / 2); axes[1].set_xticklabels(scen_b)
    axes[1].axhline(ALPHA, color="k", lw=0.6); axes[1].set_ylim(0, 1)
    axes[1].set_title("B  group effect on any null edge (family-wise)", fontsize=10); axes[1].set_xlabel("scenario"); axes[1].set_ylabel("rate")

    sh = shrinkage[shrinkage.planted]
    for i, method in enumerate(["sample_mean_diff", "bayes_edge", "bayes_pooled_int", "bayes_pooled_eff"]):
        sub = sh[sh.method == method]
        axes[2].errorbar(sub.reference + (i - 1.5) * 0.01, sub.mean_estimate, yerr=sub.sd_estimate, fmt="o", ms=4, capsize=2,
                         label=METHOD_LABELS.get(method, "unpooled sample difference"), color=f"C{PLOT_METHODS.index(method)}" if method in PLOT_METHODS else "grey")
    lim = [min(sh.reference.min(), sh.mean_estimate.min()) - 0.3, 0.1]
    axes[2].plot(lim, lim, "k", lw=0.6); axes[2].axhline(0, color="grey", lw=0.4); axes[2].axvline(0, color="grey", lw=0.4)
    axes[2].set_xlabel("true group effect (standardized, Monte Carlo reference)"); axes[2].set_ylabel("mean estimate (+/- SD over studies)")
    axes[2].set_title("C  shrinkage on planted edges (scenarios c, d, e)", fontsize=10); axes[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_group_film_bayes.{ext}", dpi=150)
    plt.close(fig)


def summarize(out_dir):
    """Build every summary table, the figure and summary.json from all raw_results_reps_*.csv in `out_dir`."""
    df = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("raw_results_reps_*.csv"))], ignore_index=True)
    gates = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("gates_reps_*.csv"))], ignore_index=True)
    runtime = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("runtime_reps_*.csv"))], ignore_index=True)
    df = df[df.scenario.isin(SETTINGS["scenarios"])]
    df["detected"] = df.detected.astype("boolean")
    edge_df = edge_level(df[df.detected.notna() & (df.edge != "population")])
    detection = detection_table(edge_df)
    detection.to_csv(out_dir / "summary_detection.csv", index=False)
    by_film = df[df.method.str.startswith("perm_") & (df.film != "mean")].groupby(["scenario", "method", "dv", "film", "edge"]).detected.mean().reset_index()
    by_film.to_csv(out_dir / "summary_detection_by_film.csv", index=False)
    familywise = familywise_table(edge_df)
    familywise.to_csv(out_dir / "summary_familywise_null.csv", index=False)
    shrinkage = shrinkage_table(df)
    shrinkage.to_csv(out_dir / "summary_shrinkage.csv", index=False)
    population = df[df.edge == "population"].groupby(["scenario", "method", "effect"]).agg(
        mean_estimate=("estimate", "mean"), detection_rate=("detected", "mean"), n=("rep", "size")).reset_index()
    population.to_csv(out_dir / "summary_population_effects.csv", index=False)
    gate_summary = gates.groupby(["scenario", "model"]).agg(n_fits=("pass_l9", "size"), fail_share=("pass_l9", lambda x: 1 - x.mean()),
                                                           divergent_share=("n_divergent", lambda x: (x > 0).mean()),
                                                           max_rhat=("max_rhat", "max"), min_ess=("min_bulk_ess", "min"),
                                                           mean_seconds=("seconds", "mean")).reset_index()
    gate_summary.to_csv(out_dir / "summary_gate.csv", index=False)
    runtime_summary = runtime.drop(columns=["rep"]).groupby("scenario").mean().reset_index()
    runtime_summary.to_csv(out_dir / "summary_runtime.csv", index=False)
    plot_summary(detection, familywise, shrinkage, out_dir)

    criteria, values = {}, {}
    if "a" in SETTINGS["scenarios"]:
        film_fw = df[(df.scenario == "a") & (df.method == "perm_film_maxT") & (df.dv == "delta")].groupby(["rep", "film"]).detected.any()
        values["a_maxT_within_film_fwer"] = rate_row(film_fw)
        criteria[f"[proposed] scenario a: max-T family-wise group-contrast rate (8 edges within film) within {FWER_BAND}"] = \
            bool(FWER_BAND[0] <= film_fw.mean() <= FWER_BAND[1])
    if "c" in SETTINGS["scenarios"]:
        c_film = df[(df.scenario == "c") & (df.method == "perm_film_maxT") & (df.dv == "delta") & (df.edge == "cg:ROI->child:ROI")]
        values["c_maxT_within_film_power"] = rate_row(c_film.detected)
        criteria[f"[proposed] scenario c: max-T (within film) detects the planted group effect in >= {POWER_TARGET:.0%} at N = 15 per group"] = \
            bool(c_film.detected.mean() >= POWER_TARGET)
        leak = familywise[(familywise.scenario == "c") & (familywise.method == "bayes_pooled_eff") & (familywise.effect == "group")]
        values["c_model4_leakage"] = leak.drop(columns=["scenario", "method", "dv", "effect"]).to_dict(orient="records")
        criteria[f"[proposed] scenario c: model-4 leakage into any null edge <= {LEAKAGE_MAX}"] = \
            bool(leak[leak.scope == "all null edges"].rate.iloc[0] <= LEAKAGE_MAX)
    settings = dict(SETTINGS, n_rep=int(df.groupby("scenario").rep.nunique().min()), mcmc={k: v for k, v in SETTINGS["mcmc"].items()})
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": settings, "criteria": criteria, "criterion_values": values,
                   "reps_per_scenario": df.groupby("scenario").rep.nunique().to_dict(),
                   "gate_fail_share": gate_summary.groupby("model").fail_share.mean().to_dict(),
                   "runtime_seconds_per_study": runtime.drop(columns=["rep", "scenario"]).mean().to_dict()}, f, indent=2, default=float)
    print(detection[detection.planted].round(3).to_string()); print(familywise.round(3).to_string()); print(criteria)


def main():
    """Run S13 (or only rebuild its summaries) and write to Results/<estimator>/S_13."""
    out_dir = results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    if not SUMMARIZE_ONLY:
        all_scenarios = list(SCENARIOS)
        seeds = spawn_seeds(BASE_SEED, len(all_scenarios) * N_REP_MAX)
        tasks = [{"scenario": s, "rep": r, "seed": seeds[all_scenarios.index(s) * N_REP_MAX + r]}
                 for r in REPS for s in SETTINGS["scenarios"]]
        start = time.time()
        results = run_parallel(run_task, tasks, N_JOBS)
        tag = f"{REPS.start}-{REPS.stop - 1}"
        for i, name in enumerate(["raw_results", "gates", "runtime", "rows"]):
            pd.DataFrame([r for res in results for r in res[i]]).to_csv(out_dir / f"{name}_reps_{tag}.csv", index=False)
        print(f"{len(tasks)} studies in {time.time() - start:.0f} s with {N_JOBS} processes")
    summarize(out_dir)


if __name__ == "__main__":
    main()
