"""S13 - Group and film effects across edges: permutation max-T vs Bayesian partial pooling (Table 2, S13).

Question: how do the per-edge permutation tests with max-T and Bayesian hierarchical models compare in estimating and
detecting group and film effects, and do partial-pooling models move an effect from one edge to the others?
Design: 2 groups x 15 dyads, 3 films x 60 s. Per-person node parameters are drawn once and kept across films; every
film has new innovations. 4-node Level-1 model; the family is the 8 cross-person edges. Scenarios (VAR gain, brain->brain
unless stated): a none; b cg brain->child brain 0.3 in all cells; c as b, ASD 0.1; d c + cg heart->child heart 0.3 TD /
0.1 ASD; e group difference (0.3 TD vs 0.1 ASD) only in film 1 (films 2, 3: 0.3 in both groups); f film effect without a
group effect (0.1 in film 1, 0.3 in films 2 and 3, both groups); g group x direction crossover (TD: cg->child 0.3,
child->cg 0.1 on brain->brain and heart->heart; ASD the reverse); h no coupling, common stimulus on all four nodes with an
ASD/TD gain ratio of 3 (S7 mechanism); i dyad covariate c ~ N(-0.5 TD / +0.5 ASD, 1), cg brain->child brain
max(0, 0.2 + 0.1 c) in both groups (no group effect given c).
Dependent variable: delta against the within-group null of each film (surrogates built from dyads of the same group and
film), standardized within edge for the Bayesian models (`src.group_model.standardize_within_edge`); the robust z
(delta / scaled MAD of the null) is a sensitivity DV for the permutation tests.
Group-label permutation tests of the delta DV (group contrasts and interaction) recompute the within-group null under
every permutation (`renull`, default since 2026-11-14), from surrogate tables over all 30 dyads of a film; the z DV is
tested on the fixed z.
Analyses per study:
  1. permutation: group contrast per edge and film, max-T over the 8 edges within each film; max-T over the 24
     edge x film tests; max-T over the 8 edges on the film-averaged delta; film contrast
     c = delta(F1) - mean(delta(F2), delta(F3)) per dyad: sign-flip test of the film effect (max-T over 8,
     `src.sim_inference.signflip_maxT`) and group-label permutation of the interaction (mean_ASD c - mean_TD c, max-T
     over 8, `group_permutation_maxT`);
  2. Bayesian per edge:            dv ~ film * group + (1|dyad)
  3. Bayesian, pooled intercepts:  dv ~ film * group + (1|edge) + (1|dyad)
  4. Bayesian, pooled effects:     dv ~ film * group + (film * group | edge) + (1|dyad); if a fit fails the convergence
     gate, the fallback (group | edge) model is fitted as well and reported separately (no silent refits).
  5. factorial permutation ANOVA (`src.perm_anova`, Freedman-Lane, whole-dyad rearrangements) on the raw delta
     cell vector (3 films x 8 edges): omnibus terms G, F, E, FxE, GxF, GxE, Gxdirection, Gxsource, Gxtarget, GxFxE,
     and per-edge G, F, GxF with max-F over the 8 edges. Group terms: `fixed` (delta at the observed within-group
     nulls, Freedman-Lane; with the covariate in Z in scenario i, and again without it: `_nocov`) and `renull`
     (raw-label permutation, delta recomputed against the within-group nulls of every relabelling; only without a
     covariate). Dyads missing a film are left out of this analysis (`n_dyads`). Rows go to fl_results_*.csv
     (methods fl_omnibus_*, fl_edge_*), with its own generator (seed, 2), after analyses 1-4; with `fl` it is the
     only analysis. Note: the PyMC fits are not bitwise reproducible between runs (the same seed gives one of a few
     outcomes, also with the code before this extension), the permutation results are.
Every run writes the per-dyad delta to S_13_delta_*.parquet and the surrogate tables to T_bc_<scenario>_*.npz
(float32, compressed).
Contrasts (95% HDI excludes 0 = detected): group effect (ASD - TD, mean over films), film contrast (film 1 - mean of
films 2 and 3, mean over groups), interaction (film contrast ASD - TD).
Convergence gate (`src.group_model.convergence_row`): R-hat < 1.01, bulk/tail ESS > 400, no divergences.

Usage: python S13_group_film_bayes.py [quick|full] [dDTF|GPDC] [summarize|fl]
  fl                  only simulate, save delta / T_bc and run the factorial permutation ANOVA (no max-T tests, no Bayes)
                      on SETTINGS['fl_scenarios'] with SETTINGS['fl_n_rep'] replicates per scenario
  SIM_NREP=<n>        replicates per scenario (default: SETTINGS)
  SIM_REPS=<a>-<b>    run only replicates a..b (inclusive) and write raw_results_<mode>_reps_<a>-<b>.csv; the summary is
                      built from every raw_results_<mode>_reps_*.csv in the folder ('summarize' rebuilds it without simulating)
  SIM_NJOBS=<n>       parallel processes (default 8)
  SIM_SCENARIOS=a,c   run only these scenarios (checks)
  SIM_OUT_DIR=<path>  write results here instead of Results/<estimator>/S_13 (checks, smoke tests)
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
from src.mvar_diag import ar_root_stability
from src.perm_anova import edge_factor_contrasts, f_statistic, freedman_lane, rearrangement_matrix, relabel_renull_test, rm_contrasts
from src.group_model import convergence_row, fit_model_with_priors, standardize_within_edge
from src.sim_generators import (build_coupling, make_stimulus, sample_node_params, sample_stimulus_gain,
                                simulate_dyads_fixed_params, zscore_designs)
from src.sim_group_bayes import edge_effects, excludes_zero, population_effects
from src import sim_progress
from src.sim_harness import results_dir, run_parallel, spawn_seeds
from src.sim_inference import (delta_and_z_all, family_pvalues, group_permutation_maxT, real_matrices, renull_delta,
                               signflip_maxT, surrogate_values, wilson_interval)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
ESTIMATOR = sys.argv[2] if len(sys.argv) > 2 else "dDTF"          # connectivity estimator: "dDTF" or "GPDC"
SUMMARIZE_ONLY = len(sys.argv) > 3 and sys.argv[3] == "summarize"
FL_ONLY = len(sys.argv) > 3 and sys.argv[3] == "fl"     # factorial permutation ANOVA only (no max-T tests, no Bayes)
N_JOBS = int(os.environ.get("SIM_NJOBS", 8))
SIM_ID = 13
BASE_SEED = 20260113
N_REP_MAX = 1000                      # seeds are indexed (scenario, replicate) up to this many replicates
MCMC_FULL = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.95, "cores": 1, "sd_prior_edge": "halfstudentt"}
SETTINGS = {
    "quick": {"n_rep": 1, "n_per_group": 6, "scenarios": ["a", "d", "f"],
              "mcmc": dict(MCMC_FULL, draws=150, tune=150, chains=2),
              "fl_scenarios": ["a", "g", "h", "i"], "fl_n_rep": {"a": 1, "g": 1, "h": 1, "i": 1}},
    "full": {"n_rep": 100, "n_per_group": 15, "scenarios": ["a", "b", "c", "d", "e", "f"], "mcmc": MCMC_FULL,
             "fl_scenarios": ["a", "c", "e", "f", "g", "h", "i"],                       # D3: 200 null, 100 power
             "fl_n_rep": {"a": 200, "h": 200, "i": 200, "c": 100, "e": 100, "f": 100, "g": 100}},
}[MODE]
if os.environ.get("SIM_NREP"):
    SETTINGS["n_rep"] = int(os.environ["SIM_NREP"])
    SETTINGS["fl_n_rep"] = {s: int(os.environ["SIM_NREP"]) for s in SETTINGS["fl_n_rep"]}
if os.environ.get("SIM_SCENARIOS"):
    keep = os.environ["SIM_SCENARIOS"].split(",")
    SETTINGS["scenarios"] = [s for s in SETTINGS["scenarios"] if s in keep]
    SETTINGS["fl_scenarios"] = [s for s in SETTINGS["fl_scenarios"] if s in keep]
REPS = range(max(SETTINGS["fl_n_rep"].values()) if FL_ONLY else SETTINGS["n_rep"])
if os.environ.get("SIM_REPS"):
    first, last = (int(x) for x in os.environ["SIM_REPS"].split("-"))
    REPS = range(first, last + 1)

FS = 2.5
SEG_S = 60
WIN_LEN_S, OVERLAP = 10.0, 0.5
MODEL_ORDER = 4
BAND_HZ = (0.15, 0.5)
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
    "f": {"cg:ROI->child:ROI": [[0.1, 0.3, 0.3], [0.1, 0.3, 0.3]]},
    "g": {"cg:ROI->child:ROI": [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]], "cg:HRV->child:HRV": [[0.3, 0.3, 0.3], [0.1, 0.1, 0.1]],
          "child:ROI->cg:ROI": [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]], "child:HRV->cg:HRV": [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]},
    "h": {},
    "i": {"cg:ROI->child:ROI": [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]},     # base gain; + slope * c (COVARIATE_SCENARIOS)
}
# common stimulus (S7 mechanism and STIM_RADIUS / STIM_F0 / sd_log of S7): a new stimulus time course per film, the same for
# all dyads of the film; person-specific gains (log-normal) drawn once per study, multiplied by `ratio` in ASD
STIM_RADIUS, STIM_F0 = 0.7, 0.3
STIMULUS_SCENARIOS = {"h": {"base_gain": 1.0, "sd_log": 0.5, "ratio": 3.0}}
# dyad-level covariate c ~ N(group_means[group], sd); gain of `edge` = max(0, base + slope * c) (D5)
COVARIATE_SCENARIOS = {"i": {"edge": "cg:ROI->child:ROI", "group_means": (-0.5, 0.5), "sd": 1.0, "slope": 0.1}}
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
                 "perm_contrast_unadj": "sign-flip, film contrast, uncorrected", "perm_contrast_maxT": "sign-flip, film contrast, max-T",
                 "perm_interaction_unadj": "permutation, interaction, uncorrected", "perm_interaction_maxT": "permutation, interaction, max-T",
                 "bayes_edge": "Bayes per edge (M2)", "bayes_pooled_int": "Bayes (1|edge) (M3)",
                 "bayes_pooled_eff": "Bayes (film*group|edge) (M4)", "bayes_pooled_eff_fallback": "Bayes (group|edge) fallback"}
PLOT_METHODS = ["perm_film_maxT", "perm_24_maxT", "perm_mean_maxT", "bayes_edge", "bayes_pooled_int", "bayes_pooled_eff"]
PLOT_METHODS_CONTRAST = {"film": ["perm_contrast_maxT", "bayes_edge", "bayes_pooled_int", "bayes_pooled_eff"],
                         "interaction": ["perm_interaction_maxT", "bayes_edge", "bayes_pooled_int", "bayes_pooled_eff"]}
FWER_BAND = (0.03, 0.07)              # scenario a, max-T within film [proposed]
POWER_TARGET = 0.80                   # scenario c, max-T within film [proposed]
LEAKAGE_MAX = 0.10                    # scenario c, model 4, any null edge [proposed]
# factorial permutation ANOVA (src.perm_anova): cell layout film-major (cell f * 8 + e), raw delta (D1), trace F (D2)
LEVELS = {"film": len(FILMS), "edge": len(EDGES)}
EDGE_CLASS = {c: edge_factor_contrasts(list(EDGES), c) for c in ("direction", "source", "target")}
FL_OMNIBUS = {   # term -> (contrast over the 24 cells, 'between' = group term (permutation) / 'within' (sign flip))
    "G": (rm_contrasts(LEVELS, ()), "between"),
    "F": (rm_contrasts(LEVELS, ("film",)), "within"),
    "E": (rm_contrasts(LEVELS, ("edge",)), "within"),
    "FxE": (rm_contrasts(LEVELS, ("film", "edge")), "within"),
    "GxF": (rm_contrasts(LEVELS, ("film",)), "between"),
    "GxE": (rm_contrasts(LEVELS, ("edge",)), "between"),
    "Gxdirection": (rm_contrasts(LEVELS, ("edge",), {"edge": EDGE_CLASS["direction"]}), "between"),
    "Gxsource": (rm_contrasts(LEVELS, ("edge",), {"edge": EDGE_CLASS["source"]}), "between"),
    "Gxtarget": (rm_contrasts(LEVELS, ("edge",), {"edge": EDGE_CLASS["target"]}), "between"),
    "GxFxE": (rm_contrasts(LEVELS, ("film", "edge")), "between"),
}
FL_EDGE = {      # per-edge terms on the 3-film vector of one edge, max-F over the 8 edges
    "G": (rm_contrasts({"film": len(FILMS)}, ()), "between"),
    "F": (rm_contrasts({"film": len(FILMS)}, ("film",)), "within"),
    "GxF": (rm_contrasts({"film": len(FILMS)}, ("film",)), "between"),
}
FL_MODES = {"between": "permute", "within": "signflip"}
FL_PLANTED_KEY = {"G": "group", "F": "film", "GxF": "interaction"}        # per-edge term -> key of `planted_effects`
CONSISTENCY_RTOL = 1e-9              # rounding only; p-values can differ by 1 / (1 + N_PERM) per exact tie with F_obs
DESCRIPTIVE_ONLY = {"dDTF": ["c leakage", "g"]}   # D6: reverse edges inherit the forward coupling (symmetric |kappa|)

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


def dyad_edges(scenario, groups, covariate, film_index):
    """Planted ``(source, target, lag, gain)`` list of every dyad in one film; covariate-dependent gain where configured."""
    cov = COVARIATE_SCENARIOS.get(scenario)
    out = []
    for d in range(len(groups)):
        edges = []
        for edge, gains in SCENARIOS[scenario].items():
            gain = gains[groups[d]][film_index]
            if cov is not None and edge == cov["edge"]:
                gain = max(0.0, gain + cov["slope"] * covariate[d])
            edges.append((*EDGES[edge], 1, gain))
        out.append(edges)
    return out


def check_planted_stability(radii, f0s, edges_per_dyad, scenario, film):
    """Raise if any dyad's planted VAR tensor has a companion eigenvalue on or outside the unit circle (no rescaling)."""
    for d, edges in enumerate(edges_per_dyad):
        _, max_root, stable = ar_root_stability(build_coupling(radii[d], f0s[d], FS, edges, N_LAGS))
        if not stable:
            raise ValueError(f"unstable planted VAR: scenario {scenario}, film {film}, dyad {d}, max root {max_root:.4f}, edges {edges}")


def simulate_delta_table(rng, scenario, n_per_group, est):
    """Simulate one study (3 films, 2 groups); tidy delta / z table against the within-group null, and the tables.

    The surrogate table of every film covers all dyads of both groups (the within-group blocks are the within-group
    null; the whole table is needed to re-null the group tests under permuted labels). Dyads with an unstable real fit
    in a film are left out of that film (NaN real values, rows and columns). Scenarios with a covariate draw it after
    the node parameters; scenarios with a stimulus draw the person gains once and a new stimulus per film (other
    scenarios draw exactly what they drew before these options existed). Every planted tensor is checked for stability.

    Returns
    -------
    df : pandas.DataFrame
    groups : np.ndarray, shape (n,)
    tables : dict of {film: (real, values)}
        real (n, n_edges) and values (n, n, n_edges) of the family edges, for `group_permutation_maxT(renull=True)`.
    covariate : np.ndarray, shape (n,), or None
    """
    n = 2 * n_per_group
    groups = np.r_[np.zeros(n_per_group, int), np.ones(n_per_group, int)]
    radii, f0s = sample_node_params(rng, n, POLE_RADIUS_RANGES), sample_node_params(rng, n, F0_RANGES)
    covariate, stim_gain = None, None
    if scenario in COVARIATE_SCENARIOS:
        cov = COVARIATE_SCENARIOS[scenario]
        covariate = rng.normal(np.asarray(cov["group_means"])[groups], cov["sd"])
    if scenario in STIMULUS_SCENARIOS:
        stim = STIMULUS_SCENARIOS[scenario]
        stim_gain = sample_stimulus_gain(rng, n, [stim["base_gain"]] * len(NODES), stim["sd_log"], np.where(groups == 1, stim["ratio"], 1.0))
    rows, tables = [], {}
    sources, targets = [s for s, _ in EDGES.values()], [t for _, t in EDGES.values()]
    for f, film in enumerate(FILMS):
        edges_per_dyad = dyad_edges(scenario, groups, covariate, f)
        check_planted_stability(radii, f0s, edges_per_dyad, scenario, film)
        stimulus = None if stim_gain is None else make_stimulus(rng, int(SEG_S * FS), FS, STIM_RADIUS, STIM_F0)
        designs = zscore_designs(simulate_dyads_fixed_params(rng, radii, f0s, int(SEG_S * FS), FS, edges_per_dyad, N_LAGS,
                                                             stimulus=stimulus, stimulus_gain=stim_gain))
        real, roots, _ = real_matrices(designs, NODES, est)
        values, _, _ = surrogate_values(designs, NODES, CHILD_NODES, est, STABILITY_MAX_ROOT, rng)
        unstable = roots >= STABILITY_MAX_ROOT
        real[unstable], values[unstable], values[:, unstable] = np.nan, np.nan, np.nan
        delta, z = delta_and_z_all(real, values, "within_group", groups)
        tables[film] = (real[:, targets, sources], values[:, :, targets, sources])
        for dyad in np.flatnonzero(~unstable):
            for edge, (s, t) in EDGES.items():
                rows.append({"dyad_id": f"d{dyad:02d}", "dyad": int(dyad), "group": GROUPS[groups[dyad]], "film": film,
                             "edge": edge, "delta": float(delta[dyad, t, s]), "z": float(z[dyad, t, s])})
    return pd.DataFrame(rows), groups, tables, covariate


def renull_args(tables, dv, films, combine=None):
    """Keyword arguments of `group_permutation_maxT`: re-nulled tables for the delta DV, the fixed-DV test for z.

    The robust z has no re-nulled version, so its group tests permute the labels on the fixed z (`renull=False`).
    `films` are stacked as columns (film x edge); `combine` maps them to the tested quantities.
    """
    if dv != "delta":
        return {"renull": False}
    return {"renull": True, "real": np.hstack([tables[f][0] for f in films]),
            "pair_values": np.concatenate([tables[f][1] for f in films], axis=2), "combine": combine}


def permutation_records(df, groups, rng, dv, tables):
    """Group-contrast permutation tests of one DV: per film (uncorrected, max-T over 8), max-T over 24, max-T on the film mean.

    For the delta DV the within-group null is recomputed under every permutation (`renull`, from `tables`); a dyad
    missing a film is then left out of the film-mean test (instead of averaging its other films).
    """
    n = len(groups)
    wide = {film: np.full((n, len(EDGES)), np.nan) for film in FILMS}
    for (film, edge), sub in df.groupby(["film", "edge"]):
        wide[film][sub.dyad.to_numpy(), list(EDGES).index(edge)] = sub[dv].to_numpy()
    out = []

    def add(method, film, edge, p):
        out.append({"method": method, "dv": dv, "film": film, "edge": edge, "effect": "group", "p": float(p), "detected": bool(p < ALPHA)})

    for film in FILMS:
        res = group_permutation_maxT(wide[film], groups, rng, N_PERM, **renull_args(tables, dv, [film]))
        for e, edge in enumerate(EDGES):
            add("perm_film_unadj", film, edge, res["unadjusted"][e])
            add("perm_film_maxT", film, edge, res["maxT"][e])
    res = group_permutation_maxT(np.hstack([wide[film] for film in FILMS]), groups, rng, N_PERM, **renull_args(tables, dv, FILMS))
    for f, film in enumerate(FILMS):
        for e, edge in enumerate(EDGES):
            add("perm_24_maxT", film, edge, res["maxT"][f * len(EDGES) + e])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        film_mean = np.nanmean(np.stack([wide[film] for film in FILMS]), axis=0)
    eye = np.eye(len(EDGES))
    res = group_permutation_maxT(film_mean, groups, rng, N_PERM, **renull_args(tables, dv, FILMS, np.vstack([eye / 3] * 3)))
    for e, edge in enumerate(EDGES):
        add("perm_mean_maxT", "mean", edge, res["maxT"][e])
    return out


def contrast_records(df, groups, rng, dv, tables):
    """Film effect (sign-flip, max-T over 8) and film x group interaction (label permutation, max-T over 8) of one DV.

    The contrast per dyad and edge is c = dv(F1) - mean(dv(F2), dv(F3)); a dyad missing a film is NaN for that edge.
    For the delta DV the interaction re-nulls the within-group null of every film under every permutation (`renull`).
    """
    n = len(groups)
    wide = {film: np.full((n, len(EDGES)), np.nan) for film in FILMS}
    for (film, edge), sub in df.groupby(["film", "edge"]):
        wide[film][sub.dyad.to_numpy(), list(EDGES).index(edge)] = sub[dv].to_numpy()
    contrast = wide["F1"] - (wide["F2"] + wide["F3"]) / 2
    eye = np.eye(len(EDGES))
    interaction = renull_args(tables, dv, FILMS, np.vstack([eye, -eye / 2, -eye / 2]))
    out = []
    for effect, label, res in (("film", "perm_contrast", signflip_maxT(contrast, rng, N_PERM, groups)),
                               ("interaction", "perm_interaction", group_permutation_maxT(contrast, groups, rng, N_PERM, **interaction))):
        for e, edge in enumerate(EDGES):
            for kind, key in (("unadj", "unadjusted"), ("maxT", "maxT")):
                p = float(res[key][e])
                out.append({"method": f"{label}_{kind}", "dv": dv, "film": "all", "edge": edge, "effect": effect,
                            "p": p, "detected": bool(p < ALPHA)})
    return out


def fl_design(groups, covariate):
    """Between-dyad columns of the factorial model: (X, Z) of group terms ('between') and of within-dyad terms ('within').

    Group coded -1 (TD) / +1 (ASD), sum-to-zero, so that the intercept of a within-dyad term is the unweighted mean of
    the group means. Group terms: X = group, Z = [1, covariate]; within-dyad terms: X = 1, Z = [group, covariate].
    """
    g = np.where(groups == 1, 1.0, -1.0)[:, None]
    ones = np.ones_like(g)
    extra = [] if covariate is None else [covariate[:, None]]
    return {"between": (g, np.hstack([ones] + extra)), "within": (ones, np.hstack([g] + extra))}


def fl_wide(df, n):
    """Raw delta as (n, film, edge); NaN where a dyad has no value in a film."""
    wide = np.full((n, len(FILMS), len(EDGES)), np.nan)
    for (film, edge), sub in df.groupby(["film", "edge"]):
        wide[sub.dyad.to_numpy(), FILMS.index(film), list(EDGES).index(edge)] = sub.delta.to_numpy()
    return wide


def fl_group_term_F(cells, labels):
    """F of every group term under one or more labellings: omnibus group terms, then per-edge G and GxF (8 edges each).

    Parameters
    ----------
    cells : np.ndarray, shape (n, 24) or (L, n, 24)
        Film-major delta cell vectors.
    labels : np.ndarray of bool, shape (L, n)

    Returns
    -------
    np.ndarray, shape (L, n_tests)
    """
    X = np.where(labels, 1.0, -1.0)[:, :, None]
    Z = np.ones((labels.shape[1], 1))
    cols = [f_statistic(cells @ C, X, Z) for C, kind in FL_OMNIBUS.values() if kind == "between"]
    per_edge = cells.reshape(cells.shape[:-1] + (len(FILMS), len(EDGES)))
    cols += [f_statistic(per_edge[..., e] @ C, X, Z) for C, kind in FL_EDGE.values() if kind == "between" for e in range(len(EDGES))]
    return np.stack([np.broadcast_to(c, (labels.shape[0],)) for c in cols], axis=1)


def fl_records(df, groups, covariate, tables, rng):
    """Factorial permutation ANOVA of one study (raw delta, 3 films x 8 edges): omnibus and per-edge (max-F) terms.

    Dyads missing a film are left out (complete cell vectors only). One permutation matrix and one sign-flip matrix
    are drawn per study and shared by all terms and edges. Variants: `fixed` (Freedman-Lane on delta at the observed
    within-group nulls; with the covariate in Z when there is one, then `fixed_nocov` without it) and, without a
    covariate, `renull` (`relabel_renull_test`, delta recomputed against the within-group nulls of every relabelling)
    and `labelperm` (per-edge G by plain relabelling of the fixed delta, for the consistency check).

    Returns
    -------
    list of dict
        Rows in the raw-results format (method, dv, film, edge, effect, p, detected) plus F and n_dyads.
    """
    n = len(groups)
    wide = fl_wide(df, n)
    used = np.isfinite(wide).all(axis=(1, 2))
    n_used = int(used.sum())
    Y = wide[used]
    cells = Y.reshape(n_used, -1)
    arrangements = {kind: rearrangement_matrix(n_used, N_PERM, mode, rng) for kind, mode in FL_MODES.items()}
    out = []

    def add(method, term, edge, p, F):
        out.append({"method": method, "dv": "delta", "film": "all", "edge": edge, "effect": term, "p": float(p),
                    "detected": bool(p < ALPHA), "F": float(F), "n_dyads": n_used})

    def add_edges(variant, term, F_obs, F_perm):
        res = family_pvalues(F_obs, F_perm)
        for e, edge in enumerate(EDGES):
            add(f"fl_edge_{variant}_unadj", term, edge, res["unadjusted"][e], F_obs[e])
            add(f"fl_edge_{variant}_maxF", term, edge, res["maxT"][e], F_obs[e])

    variants = [("fixed", covariate)] + ([] if covariate is None else [("fixed_nocov", None)])
    fixed_edge_null = {}
    for variant, cov in variants:
        design = fl_design(groups[used], None if cov is None else cov[used])
        for term, (C, kind) in FL_OMNIBUS.items():
            F_obs, F_perm = freedman_lane(cells @ C, *design[kind], arrangements[kind], FL_MODES[kind])
            add(f"fl_omnibus_{variant}", term, "all", family_pvalues(np.array([F_obs]), F_perm[:, None])["unadjusted"][0], F_obs)
        if variant != "fixed":
            continue
        for term, (C, kind) in FL_EDGE.items():
            res = [freedman_lane(Y[:, :, e] @ C, *design[kind], arrangements[kind], FL_MODES[kind]) for e in range(len(EDGES))]
            fixed_edge_null[term] = np.stack([r[1] for r in res], axis=1)
            add_edges("fixed", term, np.array([r[0] for r in res]), fixed_edge_null[term])
    if covariate is not None:
        return out

    observed = groups[used] == 1
    real = np.hstack([tables[film][0] for film in FILMS])
    pair_values = np.concatenate([tables[film][1] for film in FILMS], axis=2)

    def renulled(labels):
        full = np.tile(groups == 1, (len(labels), 1))
        full[:, used] = labels
        return renull_delta(real, pair_values, full)[:, used, :]

    assert np.allclose(renulled(observed[None])[0], cells), "re-nulled delta at the observed labels must equal delta"
    group_terms = [t for t, (_, kind) in FL_OMNIBUS.items() if kind == "between"]
    edge_terms = [t for t, (_, kind) in FL_EDGE.items() if kind == "between"]
    for variant, y_fn in (("renull", renulled), ("labelperm", lambda labels: np.broadcast_to(cells, (len(labels),) + cells.shape))):
        F_obs, F_perm = relabel_renull_test(y_fn, observed, arrangements["between"], fl_group_term_F)
        k = len(group_terms)
        if variant == "renull":
            for i, term in enumerate(group_terms):
                add("fl_omnibus_renull", term, "all", family_pvalues(F_obs[i:i + 1], F_perm[:, i:i + 1])["unadjusted"][0], F_obs[i])
        for j, term in enumerate(edge_terms):
            block = slice(k + j * len(EDGES), k + (j + 1) * len(EDGES))
            if variant == "renull":
                add_edges("renull", term, F_obs[block], F_perm[:, block])
            elif term == "G":
                res = family_pvalues(F_obs[block], F_perm[:, block])
                rel = np.abs(F_perm[:, block] - fixed_edge_null["G"]) / fixed_edge_null["G"].mean(axis=0)   # scale of the null
                for e, edge in enumerate(EDGES):
                    add("fl_edge_labelperm_unadj", term, edge, res["unadjusted"][e], F_obs[block][e])
                    out[-1]["null_max_rel_diff"] = float(rel[:, e].max())   # vs the Freedman-Lane null (same rearrangements)
    return out


def delta_rows(df, covariate, tag):
    """Long per-dyad table (dyad x film x edge) of delta, delta_std (`standardize_within_edge`), z and the covariate."""
    data, _ = standardize_within_edge(df, "delta")
    data = data.rename(columns={"delta_z": "delta_std"})
    data["covariate"] = np.nan if covariate is None else covariate[data.dyad.to_numpy()]
    data = data.assign(estimator=ESTIMATOR, **tag)
    return data[["estimator", "scenario", "rep", "dyad", "group", "film", "edge", "delta", "delta_std", "z", "covariate"]].to_dict("records")


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
    """One replicated study of one scenario: simulate, max-T tests and Bayes (unless FL_ONLY), factorial permutation ANOVA.

    Returns the tuple (raw results, gates, runtime, rows, factorial-ANOVA rows, factorial-ANOVA runtime, delta rows,
    tables): the first four are empty with FL_ONLY; the tables (float32) are those of `simulate_delta_table`.
    """
    rng = np.random.default_rng(task["seed"])
    start = time.time()
    win_len, step = window_geometry(WIN_LEN_S, OVERLAP, FS)
    est = dict(p=MODEL_ORDER, win_len=win_len, step=step, freqs=FREQS, fs=FS, estimator=ESTIMATOR, band_hz=BAND_HZ, box_cox_lambda=BOX_COX)
    df, groups, tables, covariate = simulate_delta_table(rng, task["scenario"], SETTINGS["n_per_group"], est)
    t_sim = time.time() - start
    tag = {"scenario": task["scenario"], "rep": task["rep"]}
    saved = (delta_rows(df, covariate, tag),
             {"groups": groups, "covariate": covariate,
              "tables": {film: (real.astype(np.float32), values.astype(np.float32)) for film, (real, values) in tables.items()}})

    def fl_part():
        start = time.time()
        fl = fl_records(df, groups, covariate, tables, np.random.default_rng([task["seed"], 2]))
        fl_runtime = {"simulation_and_delta": t_sim, "fl_anova": time.time() - start, "n_dyads": fl[0]["n_dyads"], **tag}
        return ([dict(r, **tag) for r in fl], [fl_runtime]) + saved

    if FL_ONLY:
        return ([], [], [], []) + fl_part()
    perm_rng = np.random.default_rng([task["seed"], 1])
    start = time.time()
    records = permutation_records(df, groups, perm_rng, "delta", tables) + permutation_records(df, groups, perm_rng, "z", tables)
    records += contrast_records(df, groups, perm_rng, "delta", tables) + contrast_records(df, groups, perm_rng, "z", tables)   # drawn after the group tests
    runtime = {"simulation_and_delta": t_sim, "permutation": time.time() - start}
    bayes, gates, bayes_time = bayes_records(df, SETTINGS["mcmc"], int(task["seed"] % 2**31))
    runtime.update(bayes_time)
    return ([dict(r, **tag) for r in records + bayes], [dict(g, **tag) for g in gates], [dict(runtime, **tag)],
            [dict(tag, n_rows=len(df), n_dropped_dyad_films=2 * SETTINGS["n_per_group"] * len(FILMS) - len(df) // len(EDGES))]) + fl_part()


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
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
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

    cells = [(effect, scenario) for effect in PLOT_METHODS_CONTRAST
             for scenario in sorted(detection[detection.planted & (detection.effect == effect)].scenario.unique())]
    for i in range(len(PLOT_METHODS_CONTRAST["film"])):
        vals = [detection[(detection.scenario == sc) & (detection.effect == eff) & detection.planted & detection.dv.isin(["delta", "delta_std"])
                          & (detection.method == PLOT_METHODS_CONTRAST[eff][i])].rate.mean() for eff, sc in cells]
        axes[3].bar(np.arange(len(cells)) + i * width * 1.5, vals, width * 1.5, color=f"C{i + 6}" if i == 0 else f"C{i + 2}",
                    label="permutation, max-T (sign-flip / label)" if i == 0 else METHOD_LABELS[PLOT_METHODS_CONTRAST["film"][i]])
    axes[3].set_xticks(np.arange(len(cells)) + 0.45); axes[3].set_xticklabels([f"{eff}\n{sc}" for eff, sc in cells], fontsize=8)
    axes[3].axhline(POWER_TARGET, color="k", lw=0.6, ls=":"); axes[3].set_ylim(0, 1)
    axes[3].set_title("D  film effect and interaction detected on planted edges", fontsize=10); axes[3].set_ylabel("detection rate")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"fig_group_film_bayes.{ext}", dpi=150)
    plt.close(fig)


def summarize(out_dir):
    """Build every summary table, the figure and summary.json from all raw_results_<MODE>_reps_*.csv in `out_dir`."""
    df = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"raw_results_{MODE}_reps_*.csv"))], ignore_index=True)
    gates = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"gates_{MODE}_reps_*.csv"))], ignore_index=True)
    runtime = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"runtime_{MODE}_reps_*.csv"))], ignore_index=True)
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
    for effect, method in (("film", "perm_contrast_maxT"), ("interaction", "perm_interaction_maxT")):
        if "a" in SETTINGS["scenarios"]:
            row = familywise[(familywise.scenario == "a") & (familywise.method == method) & (familywise.dv == "delta")
                             & (familywise.scope == "all null edges")].iloc[0]
            values[f"a_{method}_fwer"] = rate_row(edge_df[(edge_df.scenario == "a") & (edge_df.method == method) & (edge_df.dv == "delta")].groupby("rep").detected.any())
            criteria[f"[proposed] scenario a: max-T family-wise {effect} rate (8 edges) within {FWER_BAND}"] = bool(FWER_BAND[0] <= row.rate <= FWER_BAND[1])
    if "f" in SETTINGS["scenarios"]:
        f_det = edge_df[(edge_df.scenario == "f") & (edge_df.method == "perm_contrast_maxT") & (edge_df.dv == "delta") & (edge_df.edge == "cg:ROI->child:ROI")]
        values["f_signflip_maxT_power"] = rate_row(f_det.detected)
        criteria[f"[proposed] scenario f: sign-flip max-T detects the planted film effect in >= {POWER_TARGET:.0%}"] = bool(f_det.detected.mean() >= POWER_TARGET)
        f_int = edge_df[(edge_df.scenario == "f") & (edge_df.method == "perm_interaction_maxT") & (edge_df.dv == "delta")].groupby("rep").detected.any()
        values["f_interaction_maxT_fwer"] = rate_row(f_int)
        criteria[f"[proposed] scenario f (no interaction): max-T family-wise interaction rate <= {FWER_BAND[1]}"] = bool(f_int.mean() <= FWER_BAND[1])
    settings = dict(SETTINGS, n_rep=int(df.groupby("scenario").rep.nunique().min()), mcmc={k: v for k, v in SETTINGS["mcmc"].items()})
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"mode": MODE, "settings": settings, "criteria": criteria, "criterion_values": values,
                   "reps_per_scenario": df.groupby("scenario").rep.nunique().to_dict(),
                   "gate_fail_share": gate_summary.groupby("model").fail_share.mean().to_dict(),
                   "runtime_seconds_per_study": runtime.drop(columns=["rep", "scenario"]).mean().to_dict()}, f, indent=2, default=float)
    print(detection[detection.planted].round(3).to_string()); print(familywise.round(3).to_string()); print(criteria)
    planted = detection[detection.planted & detection.dv.isin(["delta", "delta_std"])]
    readouts = {"family-wise rate, scenario a (all methods)": sim_progress.range_text(familywise[(familywise.scenario == "a") & (familywise.scope == "all null edges")].rate.tolist()),
                "detection on planted edges": sim_progress.range_text(planted.rate.tolist()),
                "gate failures": sim_progress.range_text(gate_summary.fail_share.tolist())}
    return readouts, criteria


def fl_variant(method):
    """'renull' or 'fixed' from a factorial-ANOVA method name."""
    return "renull" if "renull" in method else "fixed"


def fl_omnibus_table(fl):
    """Omnibus rejection rate (Wilson CI) per scenario, variant, covariate in Z and term."""
    rows = []
    for (scenario, method, term), sub in fl[fl.method.str.startswith("fl_omnibus")].groupby(["scenario", "method", "effect"]):
        rows.append({"estimator": ESTIMATOR, "scenario": scenario, "variant": fl_variant(method),
                     "covariate_in_Z": method == "fl_omnibus_fixed" and scenario in COVARIATE_SCENARIOS, "term": term,
                     **rate_row(sub.detected)})
    return pd.DataFrame(rows)


def fl_edge_table(fl):
    """Per-edge rejection rate, unadjusted and max-F adjusted, per scenario, variant, term and edge (with the planted flag)."""
    edge = fl[fl.method.str.match(r"fl_edge_(fixed|renull)_")].copy()
    edge["variant"] = edge.method.str.split("_").str[2]
    edge["adjustment"] = edge.method.str.split("_").str[3]
    rates = edge.groupby(["scenario", "variant", "effect", "edge", "adjustment"]).detected.mean().unstack("adjustment")
    out = rates.rename(columns={"unadj": "rate_unadj", "maxF": "rate_maxF"}).reset_index().rename(columns={"effect": "term"})
    out["n"] = edge.groupby(["scenario", "variant", "effect", "edge"]).rep.nunique().to_numpy()
    out["planted"] = [e in planted_effects(s)[FL_PLANTED_KEY[t]] for s, t, e in zip(out.scenario, out.term, out.edge)]
    out.insert(0, "estimator", ESTIMATOR)
    return out[["estimator", "scenario", "variant", "term", "edge", "planted", "rate_unadj", "rate_maxF", "n"]]


def fl_select(fl, scenario, method, term, edge=None):
    """Rows of one scenario, method and term (and edge)."""
    sub = fl[(fl.scenario == scenario) & (fl.method == method) & (fl.effect == term)]
    return sub if edge is None else sub[sub.edge == edge]


def fl_familywise(fl, scenario, method, term, excluded=()):
    """Share of studies with at least one rejection among the edges not in `excluded` (Wilson CI)."""
    sub = fl_select(fl, scenario, method, term)
    return rate_row(sub[~sub.edge.isin(excluded)].groupby("rep").detected.any())


def in_band(rate):
    """The S13 'within band' rule: point estimate inside FWER_BAND."""
    return bool(FWER_BAND[0] <= rate <= FWER_BAND[1])


def fl_criteria(fl, omnibus):
    """[proposed] acceptance criteria of the factorial permutation ANOVA, and the values behind them.

    The renull variant is the primary test of group terms without a covariate; scenario i uses the fixed variant
    with the covariate in Z. Criteria listed in DESCRIPTIVE_ONLY for this estimator are recorded as 'descriptive'.
    """
    criteria, values = {}, {}
    present = set(fl.scenario)
    descriptive = DESCRIPTIVE_ONLY.get(ESTIMATOR, [])

    def omni(scenario, method, term):
        return rate_row(fl_select(fl, scenario, method, term, "all").detected)

    def put(key, text, met, value, descriptive_key=None):
        values[key] = value
        criteria[f"[proposed] {text}"] = "descriptive" if descriptive_key in descriptive else bool(met)

    if "a" in present:
        for method in ("fl_omnibus_fixed", "fl_omnibus_renull"):
            rates = omnibus[(omnibus.scenario == "a") & (omnibus.variant == fl_variant(method))].set_index("term").rate
            put(f"a_{method}", f"scenario a: factorial ANOVA omnibus rate of every term within {FWER_BAND} ({fl_variant(method)})",
                all(in_band(r) for r in rates), rates.to_dict())
        for variant, terms in (("fixed", ["G", "F", "GxF"]), ("renull", ["G", "GxF"])):
            fw = {t: fl_familywise(fl, "a", f"fl_edge_{variant}_maxF", t) for t in terms}
            put(f"a_edge_{variant}_maxF_fwer", f"scenario a: per-edge max-F family-wise rate of {', '.join(terms)} within {FWER_BAND} ({variant})",
                all(in_band(v["rate"]) for v in fw.values()), fw)
    if "c" in present:
        planted = planted_effects("c")["group"]
        reverse = [f"{b}->{a}" for a, b in (e.split("->") for e in planted)]
        v = omni("c", "fl_omnibus_renull", "GxE")
        put("c_GxE_renull", f"scenario c: omnibus GxE detected >= {POWER_TARGET:.0%} (renull)", v["rate"] >= POWER_TARGET, v)
        v = rate_row(fl_select(fl, "c", "fl_edge_renull_maxF", "G", planted[0]).detected)
        put("c_edge_G_renull_maxF_power", f"scenario c: per-edge max-F G detects the planted edge >= {POWER_TARGET:.0%} (renull)",
            v["rate"] >= POWER_TARGET, v)
        leak = {"all null edges": fl_familywise(fl, "c", "fl_edge_renull_maxF", "G", planted),
                "null edges except reverse of planted": fl_familywise(fl, "c", "fl_edge_renull_maxF", "G", planted + reverse)}
        put("c_edge_G_renull_leakage", f"scenario c: per-edge max-F G leakage into any null edge <= {LEAKAGE_MAX} (renull; descriptive only "
            "for the measure whose reverse edges inherit the forward coupling, D6)", leak["all null edges"]["rate"] <= LEAKAGE_MAX, leak, "c leakage")
    if "e" in present:
        v = omni("e", "fl_omnibus_renull", "GxF")
        put("e_GxF_renull", f"scenario e: omnibus GxF detected >= {POWER_TARGET:.0%} (renull)", v["rate"] >= POWER_TARGET, v)
        v = rate_row(fl_select(fl, "e", "fl_edge_renull_maxF", "GxF", planted_effects("e")["interaction"][0]).detected)
        put("e_edge_GxF_renull_maxF_power", f"scenario e: per-edge max-F GxF detects the planted edge >= {POWER_TARGET:.0%} (renull)",
            v["rate"] >= POWER_TARGET, v)
    if "f" in present:
        v = omni("f", "fl_omnibus_fixed", "F")
        put("f_F", f"scenario f: omnibus F detected >= {POWER_TARGET:.0%}", v["rate"] >= POWER_TARGET, v)
        v = omni("f", "fl_omnibus_renull", "GxF")
        put("f_GxF_renull", f"scenario f: omnibus GxF rate within {FWER_BAND} (renull)", in_band(v["rate"]), v)
    if "g" in present:
        v = omni("g", "fl_omnibus_renull", "Gxdirection")
        put("g_Gxdirection_renull", f"scenario g: omnibus G x direction detected >= {POWER_TARGET:.0%} (renull; descriptive only for the "
            "measure whose reverse edges inherit the forward coupling, D6)", v["rate"] >= POWER_TARGET, v, "g")
        v = omni("g", "fl_omnibus_renull", "G")
        put("g_G_renull", f"scenario g: omnibus G main effect within {FWER_BAND} (renull; descriptive only for the measure whose "
            "reverse edges inherit the forward coupling, D6)", in_band(v["rate"]), v, "g")
    if "h" in present:
        renull = omnibus[(omnibus.scenario == "h") & (omnibus.variant == "renull")].set_index("term").rate
        fixed = omnibus[(omnibus.scenario == "h") & (omnibus.variant == "fixed") & omnibus.term.str.startswith("G")].set_index("term").rate
        put("h_G_terms", f"scenario h: every group term within {FWER_BAND} (renull; fixed reported)", all(in_band(r) for r in renull),
            {"renull": renull.to_dict(), "fixed": fixed.to_dict()})
    if "i" in present:
        v = omni("i", "fl_omnibus_fixed", "G")
        put("i_G_with_covariate", f"scenario i: omnibus G within {FWER_BAND} with the covariate in Z (rate without it reported)",
            in_band(v["rate"]), {"covariate_in_Z": v, "covariate_omitted": omni("i", "fl_omnibus_fixed_nocov", "G")})
    label = fl[fl.method == "fl_edge_labelperm_unadj"].set_index(["scenario", "rep", "edge"])
    fixed = fl[(fl.method == "fl_edge_fixed_unadj") & (fl.effect == "G")].set_index(["scenario", "rep", "edge"]).p.reindex(label.index)
    diff = np.abs(fixed - label.p)
    put("consistency_labelperm", f"consistency: per-edge G (fixed, Z = intercept) permutation distribution equals plain label "
        f"permutation of F on the same rearrangements (relative difference <= {CONSISTENCY_RTOL})",
        bool(label.null_max_rel_diff.max() <= CONSISTENCY_RTOL),
        {"null_max_rel_diff": float(label.null_max_rel_diff.max()), "p_max_abs_difference": float(diff.max()),
         "share_identical_p": float((diff == 0).mean()), "n_tests": int(len(label))})
    return criteria, values


def summarize_fl(out_dir):
    """Factorial-ANOVA summaries (summary_fl_omnibus.csv, summary_fl_edge.csv) and criteria, merged into summary.json.

    Reads every fl_results_<MODE>_*.csv (runs with and without `fl`; a study present in several files is counted once:
    the permutation results of a study are the same in both kinds of run, same seeds).
    """
    fl = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"fl_results_{MODE}_*.csv"))], ignore_index=True)
    fl = fl.drop_duplicates(["scenario", "rep", "method", "effect", "edge"])
    runtime = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"fl_runtime_{MODE}_*.csv"))], ignore_index=True)
    omnibus, edge = fl_omnibus_table(fl), fl_edge_table(fl)
    omnibus.to_csv(out_dir / "summary_fl_omnibus.csv", index=False)
    edge.to_csv(out_dir / "summary_fl_edge.csv", index=False)
    criteria, values = fl_criteria(fl, omnibus)
    path = out_dir / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {"mode": MODE, "settings": {"n_rep": None}, "criteria": {}, "criterion_values": {}}
    summary["criteria"].update(criteria)
    summary["criterion_values"].update(values)
    summary["fl_reps_per_scenario"] = fl.groupby("scenario").rep.nunique().to_dict()
    summary["fl_runtime_seconds_per_study"] = runtime.drop(columns=["rep", "scenario"]).mean().to_dict()
    summary["fl_min_dyads_used"] = int(runtime.n_dyads.min())
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(omnibus.round(3).to_string()); print(criteria)
    null = omnibus[omnibus.scenario.isin(["a", "h"]) | ((omnibus.scenario == "i") & omnibus.covariate_in_Z)]
    return {"factorial ANOVA omnibus rate under the null (a, h, i with c)": sim_progress.range_text(null.rate.tolist())}, criteria


def study_lines(task, result):
    """Progress lines for one finished study: runtime split, and a warning per Bayesian fit that fails the gate."""
    fl_runtime = result[5][0]
    if FL_ONLY:
        return [f"study scenario {task['scenario']} rep {task['rep']}: simulation {fl_runtime['simulation_and_delta']:.0f} s, "
                f"factorial ANOVA {fl_runtime['fl_anova']:.1f} s"]
    runtime = result[2][0]
    bayes = sum(v for k, v in runtime.items() if k.startswith("bayes"))
    lines = [f"study scenario {task['scenario']} rep {task['rep']}: simulation {runtime['simulation_and_delta']:.0f} s, "
             f"permutation {runtime['permutation']:.1f} s, factorial ANOVA {fl_runtime['fl_anova']:.1f} s, Bayes {bayes:.0f} s"]
    for gate in result[1]:
        if not gate["pass_l9"]:
            lines.append(f"WARNING gate failed: scenario {task['scenario']} rep {task['rep']}, {gate['model']} (edge {gate['edge']}): "
                         f"R-hat {gate['max_rhat']:.4f}, bulk ESS {gate['min_bulk_ess']:.0f}, tail ESS {gate['min_tail_ess']:.0f}, "
                         f"divergences {gate['n_divergent']}")
    return lines


def write_fl_outputs(out_dir, tasks, results, tag):
    """Factorial-ANOVA rows and runtime (CSV), per-dyad delta (parquet) and the surrogate tables per scenario (npz)."""
    run = f"{MODE}_flrun" if FL_ONLY else MODE
    pd.DataFrame([r for res in results for r in res[4]]).to_csv(out_dir / f"fl_results_{run}_reps_{tag}.csv", index=False)
    pd.DataFrame([r for res in results for r in res[5]]).to_csv(out_dir / f"fl_runtime_{run}_reps_{tag}.csv", index=False)
    pd.DataFrame([r for res in results for r in res[6]]).to_parquet(out_dir / f"S_13_delta_{run}_reps_{tag}.parquet", index=False)
    for scenario in sorted({t["scenario"] for t in tasks}):
        arrays = {}
        for task, res in zip(tasks, results):
            if task["scenario"] != scenario:
                continue
            key = f"rep{task['rep']:04d}"
            arrays[f"{key}_groups"] = res[7]["groups"]
            if res[7]["covariate"] is not None:
                arrays[f"{key}_covariate"] = res[7]["covariate"].astype(np.float32)
            for film, (real, values) in res[7]["tables"].items():
                arrays[f"{key}_{film}_real"], arrays[f"{key}_{film}_values"] = real, values
        np.savez_compressed(out_dir / f"T_bc_{scenario}_{run}_reps_{tag}.npz", edges=np.array(list(EDGES)), **arrays)


def main():
    """Run S13 (or only rebuild its summaries) and write to Results/<estimator>/S_13 (or SIM_OUT_DIR)."""
    out_dir = Path(os.environ["SIM_OUT_DIR"]) if os.environ.get("SIM_OUT_DIR") else results_dir(Path(__file__).resolve().parent, SIM_ID, ESTIMATOR)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_progress.begin(f"S13 {ESTIMATOR}{' fl' if FL_ONLY else ''}", Path(__file__).resolve().parent / "Results", mode=MODE,
                       replicates=f"{REPS.start}-{REPS.stop - 1}", n_per_group=SETTINGS["n_per_group"])
    if not SUMMARIZE_ONLY:
        all_scenarios = list(SCENARIOS)
        seeds = spawn_seeds(BASE_SEED, len(all_scenarios) * N_REP_MAX)
        if FL_ONLY:
            tasks = [{"scenario": s, "rep": r, "seed": seeds[all_scenarios.index(s) * N_REP_MAX + r]}
                     for r in REPS for s in SETTINGS["fl_scenarios"] if r < SETTINGS["fl_n_rep"][s]]
        else:
            tasks = [{"scenario": s, "rep": r, "seed": seeds[all_scenarios.index(s) * N_REP_MAX + r]}
                     for r in REPS for s in SETTINGS["scenarios"]]
        start = time.time()
        results = run_parallel(run_task, tasks, N_JOBS, result_lines=study_lines)
        tag = f"{REPS.start}-{REPS.stop - 1}"
        if not FL_ONLY:
            for i, name in enumerate(["raw_results", "gates", "runtime", "rows"]):
                pd.DataFrame([r for res in results for r in res[i]]).to_csv(out_dir / f"{name}_{MODE}_reps_{tag}.csv", index=False)
        write_fl_outputs(out_dir, tasks, results, tag)
        print(f"{len(tasks)} studies in {time.time() - start:.0f} s with {N_JOBS} processes")
    readouts, criteria = {}, {}
    if any(out_dir.glob(f"raw_results_{MODE}_reps_*.csv")):
        readouts, criteria = summarize(out_dir)
    if any(out_dir.glob(f"fl_results_{MODE}_*.csv")):
        fl_readouts, fl_criteria = summarize_fl(out_dir)
        readouts.update(fl_readouts)
        criteria.update(fl_criteria)
    sim_progress.finish(out_dir, readouts, {k: v for k, v in criteria.items() if isinstance(v, bool)})


if __name__ == "__main__":
    main()
