"""Stage 6 - group model: Delta ffDTF vs surrogate (H2/H4).

Stage 5 is done and untouched: for every real dyad x film x edge it wrote a
signed `delta_dtf` and `z_vs_surrogate` against a film-matched surrogate null
(`Interbrain_ffDTF_analysis/05_surrogate/stage05_delta_table.csv`, 1512 rows =
42 dyads x 3 films x 12 edges). Stage 6 fits the group-level Bayesian model(s)
on that tidy table and answers H2 (caregiver->child TPJ coupling, reduced in
ASD?) and H4 (caregiver->child HRV co-regulation, altered in ASD -- sign not
assumed). `delta_dtf`/`z_vs_surrogate` stay SIGNED everywhere in every output
of this stage too -- never `abs()` or clipped.

Locked decisions (L1-L9) -- implemented as stated, not silently resolved
differently:

- L1: DV = `delta_dtf` (standardized per edge, D3), Student-t family for
  robustness to outliers. `z_vs_surrogate` is modelled too, but ONLY for the
  asymmetry track (L6) -- never for the plain per-edge models, and never
  `abs()`-ed either DV.
- L2: fixed effects `film * group`, sum-to-zero contrasts on both (`C(.,
  Sum)`) so `group` reads as a main effect and the interaction stays clean.
  `film` is a 3-level valence factor (Peppa/Incredibles/Brave), NOT an
  ordered scale -- never averaged across films as if it were one.
- L3: planned contrast `Incredibles vs (Peppa+Brave)/2`, reported two-sided
  (posterior mean, 95% HDI, `P(>0)` and `P(<0)`), overall and within each
  group. Bambi has no `brms::hypothesis()`/`emmeans` equivalent, so this
  script computes it the estimated-marginal-means way: predict the six
  film x group cell means at the population level
  (`include_group_specific=False`) and take linear combinations of those
  posterior draws -- algebraically identical to a contrast on the fixed
  effects, and it generalises cleanly to the interaction test.
- L4: inference = posterior mean, 95% HDI, and directional posterior
  probability `P(effect>0)`/`P(effect<0)` for every reported contrast. No
  p-values as the primary object (see L7 for the one supplementary
  exception).
- L5: confirmatory edge set = the 6 Stage-5 emphasis edges (H2_primary,
  H2_reverse, H4_primary, H4_reverse, and the 2 exploratory cross
  brain-heart edges). The 6 "other" edges are not modelled by default
  (`FIT_OTHER_EDGES` toggle below).
- L6: asymmetry track. For H2 and H4, `asym = value(primary) -
  value(reverse)` per dyad x film, same `film*group + (1|dyad_id)`
  structure. Primary DV = `z_vs_surrogate` (scale-free, robust to the
  child/adult maturation artefact); `delta_dtf` (standardized) is a
  sensitivity model. Intercept (== grand mean over the 6 film x group cells,
  read off the same posterior-prediction machinery as L3) `> 0` means
  caregiver-leading.
- L7: mild BH-FDR across the PRIMARY family only = {H2_primary group effect,
  H4_primary group effect} -- kept deliberately small and named
  (`PRIMARY_FAMILY` below). Feeds `2*min(P(>0),P(<0))`, an analog p-value
  documented as such, into `src.group_model.bh_fdr`. Reported as a
  supplementary column beside the Bayesian summary, never as the primary
  inference. Exploratory edges get no correction.
- L8: drop `real_stable == False` rows before fitting; report the drop count
  per model even when it is zero (currently: zero for every edge -- a
  visible no-op, not a silent one).
- L9: 4 chains, `DRAWS` post-warmup draws/chain (>= ~2000), `target_accept`
  starting at 0.95, fixed seed. Pass criteria: max Rhat < 1.01, min bulk/tail
  ESS > 400, zero divergences (else the count is reported, not hidden), a
  sane `pp_check`, LOO Pareto-k mostly < 0.7. No silent retry ladder: a
  model that fails L9 is reported as failing in the diagnostics table and the
  gate, which is the trigger to consider the 2-variable fallback -- not
  something this script patches on its own.

Open decisions (D0-D5) -- default implemented, alternative documented:

- D0: engine. The plan's literal contract names `stage06_group_model.R` +
  brms `hypothesis()`. Checked before writing any code: `brms`/`rstan`/
  `cmdstanr` are NOT installed in this environment's R, while `bambi` (0.17.2)
  and `arviz` (0.23.4) already are in `.venv`. Per this prompt's own
  instruction ("if the chosen engine is not installed, stop and say so"),
  this was surfaced to the project owner, who chose **Bambi + ArviZ**. This
  single Python script therefore does both the fit and the gate (keeping one
  toolchain, per the D0 Python-engine branch), reusing `src/group_model.py`
  for the model-fitting/contrast/diagnostic-plot functions (every MCMC
  setting is passed in explicitly, bundled into `MCMC_CONFIG` below) and
  `src/reporting.py` for the HTML-fragment renderers -- this script itself is
  config + orchestration. The output contract (tidy CSVs + PNGs + gate) is
  identical to what the brms branch would have produced.
- D1 (CORRECTNESS FIX): the plan/note formula is `(1|dyad_id) + (1|child_id)
  + (1|caregiver_id)`, but in this design child and caregiver are 1:1 with
  dyad (no member appears in more than one dyad) -- the three grouping
  factors are the same partition of the rows, so the three variance
  components are not jointly identifiable. Collapsed to `(1|dyad_id)` only
  (42 groups x 3 films per edge, a clean repeated-measures term). Surfaced
  in the run summary and the gate header for Jarek to ratify or override.
- D2: per-edge models (default, confirmatory) vs one pooled cross-edge model
  with `+ (1|edge)` partial pooling across the 6 emphasis edges
  (`FIT_POOLED_MODEL`, default True) as a shrinkage companion/cross-check --
  it does not replace the per-edge results.
- D3: `delta_dtf` is tiny (~1e-4 to 1e-3); standardized per edge (z-score
  across that edge's kept rows) before fitting, with explicit
  weakly-informative priors on the standardized scale (Intercept, fixed
  effects ~ Normal(0,1); group/edge SD, sigma ~ half-Student-t(3,0,1)).
  Contrasts are reported in standardized units and back-transformed to raw
  Delta-units (`unit` column). `z_vs_surrogate` (already ~unit scale) is
  modelled on its native scale with the same prior set, `unit="native"`.
- D4: baseline model (no covariate) is primary. `ADD_AGE_COVARIATE` /
  `ADD_IAF_COVARIATE` toggle sensitivity models adding mean-centred
  `age_months` and/or the dyadic `iaf_distance` column of
  `Exploratory_spectral_analysis/04_band_assignment/iaf_metrics.csv` (path
  and column name confirmed against the actual file before writing this
  script -- it does NOT live under `Interbrain_ffDTF_analysis/`). A missing
  file/column when a toggle is on errors loudly; no silent skip.
- D5: no expected sign is encoded for the H4 group effect --
  `P(ASD>TD)`/`P(ASD<TD)` are always reported symmetrically.

Hand-off note (not a Stage 6 decision, just a flag): `pipeline_plan.md` Stage
6 and `notatka_projekt_DTF_HRV_H2_H4.md` still describe the group model with
the stale three-term RE formula this script corrects via D1 -- worth fixing
in those docs, not done here.

Writes `Interbrain_ffDTF_analysis/06_group/` per the module-level OUTPUT_DIR:
fitted `idata` per model (`models/*.nc`), `stage06_contrasts.csv`,
`stage06_diagnostics.csv`, `stage06_primary_summary.csv`, QC figures, and the
interactive `group_model_gate.html`.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.group_model import (
    asymmetry_dv, asymmetry_specs_from_topology, bh_fdr, edge_subset, load_delta_table, standardize_within_edge,
    add_covariates, build_formula, build_priors, fit_model, fit_model_with_priors, common_terms_of,
    reference_grid, compute_contrasts, back_transform,
    convergence_row, plot_forest, plot_ppc_figure, plot_pareto_k_figure, plot_edge_funnel,
    compute_localization_rows,
)
from src.design import assert_edges_known, node_names
from src.io_utils import ensure_dir, safe_label
from src.pipeline_config import load_stage_config
from src.reporting import (
    render_diagnostics_table, render_contrast_table, render_primary_table, render_edge_panel,
    render_localization_table,
)

# ---------------------------------------------------------------------------
# Configuration -- settings live in pipeline_config.json (shared + this
# stage's own section); only paths/values computed from PROJECT_ROOT or
# other config values stay here. See src.pipeline_config.load_stage_config.
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")
CFG = load_stage_config(CONFIG_PATH, "stage06_group_model")

ANALYSIS_ROOT = PROJECT_ROOT / CFG["ANALYSIS_ROOT_NAME"]
INPUT_CSV = ANALYSIS_ROOT / CFG["INPUT_SUBDIR"] / CFG["INPUT_FILENAME"]
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / CFG["OUTPUT_SUBDIR"])
MODELS_DIR = ensure_dir(OUTPUT_DIR / "models")
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

ENGINE = CFG["ENGINE"]  # D0, ratified by the project owner (brms/cmdstanr not installed; see module docstring)

# Node topology (L6): validate edge_topology/PRIMARY_FAMILY against the
# actual node names before anything below relies on them.
NODE_NAMES = node_names(CFG["nodes"])
assert_edges_known(
    [(edge["source"], edge["target"]) for edge in CFG["edge_topology"]], NODE_NAMES,
    context="stage06 pipeline_config.json's edge_topology",
)
assert_edges_known(
    [tuple(row[0].split("->")) for row in CFG["PRIMARY_FAMILY"]], NODE_NAMES,
    context="stage06 pipeline_config.json's PRIMARY_FAMILY",
)

EMPHASIS_EDGES = [
    (f"{edge['source']}->{edge['target']}", edge["class"]) for edge in CFG["edge_topology"]
]  # L5
FIT_OTHER_EDGES = CFG["FIT_OTHER_EDGES"]  # L5: the 6 non-emphasis edges are not modelled by default
FIT_POOLED_MODEL = CFG["FIT_POOLED_MODEL"]  # D2: shrinkage cross-check over the 6 emphasis edges, companion not replacement

ADD_AGE_COVARIATE = CFG["ADD_AGE_COVARIATE"]  # D4
ADD_IAF_COVARIATE = CFG["ADD_IAF_COVARIATE"]  # D4
IAF_METRICS_CSV = PROJECT_ROOT / "Exploratory_spectral_analysis" / "04_band_assignment" / "iaf_metrics.csv"
IAF_DISTANCE_COLUMN = CFG["IAF_DISTANCE_COLUMN"]  # confirmed against the actual file's header before writing this script

# L9 MCMC config
CHAINS = CFG["CHAINS"]
DRAWS = CFG["DRAWS"]
TUNE = CFG["TUNE"]            # L-conv-1: was 1000 -- longer warmup for mass-matrix adaptation at the (1|edge) funnel neck
TARGET_ACCEPT = CFG["TARGET_ACCEPT"]   # L-conv-1: was 0.95 -- smaller step size to clear the funnel at 6 edge-groups
SEED = CFG["SEED"]
HDI_PROB = CFG["HDI_PROB"]
CORES = CFG["CORES"]  # macOS fix: PyMC's default "fork" worker start crashes (SIGSEGV, child dies pre-exec) once
           # Accelerate/vecLib BLAS threads are live in the parent. "spawn"/"forkserver" avoid that
           # crash but both re-import this module as __main__ during worker bootstrap, and this
           # top-level pipeline script has no `if __name__ == "__main__":` guard -- that re-import
           # re-runs the whole pipeline recursively and fails the same way every time. cores=1 sidesteps
           # all three failure modes by never creating a worker subprocess: the CHAINS chains are still
           # drawn (for rhat/ESS across chains), just sequentially in this one process. Slower, not
           # parallel, but the only option that doesn't require restructuring the whole script.

# Convergence-fix toggles (Stage 6 addendum). L-conv-2: SD_PRIOR_EDGE scopes
# ONLY the (1|edge) intercept SD hyperprior -- 1|dyad_id and every M1/M2
# varying-slope SD keep their existing HalfStudentT(3,1) untouched.
SD_PRIOR_EDGE = CFG["SD_PRIOR_EDGE"]  # D-conv-B: "halfstudentt" (nu=3,sigma=1, current) | "halfnormal" (sigma=1, tighter)
FIT_EDGE_AS_FIXED = CFG["FIT_EDGE_AS_FIXED"]  # D-conv-A: M3 = film*group*edge fixed (sum contrasts), no (1|edge); side-by-side with M0/M1/M2, cannot funnel by construction

# Bundled for src.group_model's fit_model/fit_model_with_priors/build_priors,
# which take every MCMC setting as an explicit argument rather than reading a
# module-level constant.
MCMC_CONFIG = {
    "draws": DRAWS, "tune": TUNE, "chains": CHAINS, "target_accept": TARGET_ACCEPT,
    "seed": SEED, "cores": CORES, "sd_prior_edge": SD_PRIOR_EDGE,
}

# L9 pass criteria
RHAT_MAX = CFG["RHAT_MAX"]
ESS_MIN = CFG["ESS_MIN"]
PARETO_K_MAX = CFG["PARETO_K_MAX"]

# L7: the one named primary family for BH-FDR (kept deliberately small)
PRIMARY_FAMILY = [tuple(row) for row in CFG["PRIMARY_FAMILY"]]

DV_MAIN = CFG["DV_MAIN"]  # "delta_dtf"
FILMS = CFG["FILMS"]
GROUPS = CFG["GROUPS"]

# ROI info (display-only, for the gate header) -- must match Stage 2's
# ROI_LABEL/ROI_CHANNELS (scripts/stage02_envelopes.py), since the
# `child:ROI`/`cg:ROI` edges in INPUT_CSV are envelopes computed over this
# electrode set.
ROI_LABEL = CFG["ROI_LABEL"]
ROI_CHANNELS = CFG["ROI_CHANNELS"]

# Estimator info (display-only, for the gate header) -- must match Stage 5's
# ESTIMATOR (scripts/stage05_surrogate.py), since INPUT_CSV's `real_ffdtf`/
# `delta_dtf`/`z_vs_surrogate` columns are computed from that estimator's
# cube (Stage 5's CSV carries no ESTIMATOR column of its own; this is a
# manually-synced label, not read from the data).
ESTIMATOR = CFG["ESTIMATOR"]  # or "ffDTF"/"GPDC" -- must match Stage 5's ESTIMATOR

# Box-Cox info (display-only, for the gate header) -- must match Stage 5's
# BOX_COX_LAMBDA (scripts/stage05_surrogate.py); same manually-synced,
# not-read-from-data caveat as ESTIMATOR above.
BOX_COX_LAMBDA = CFG["BOX_COX_LAMBDA"]  # (x**lambda - 1) / lambda; -1 = no transform -- must match Stage 5's BOX_COX_LAMBDA




EXTRA_TERMS = (["age_months_c"] if ADD_AGE_COVARIATE else []) + (["iaf_distance_c"] if ADD_IAF_COVARIATE else [])




# ---------------------------------------------------------------------------
# 1. Load and confirm schema (per this prompt's own instruction: confirm the
#    real file before trusting the spec's quoted schema)
# ---------------------------------------------------------------------------
delta_table = load_delta_table(INPUT_CSV)
required_columns = {
    "dyad_id", "film", "source", "target", "edge", "edge_class", "group", "age_months",
    "real_ffdtf", "null_median", "null_std", "n_null", "delta_dtf", "z_vs_surrogate", "real_stable",
}
assert required_columns.issubset(delta_table.columns), f"missing columns: {required_columns - set(delta_table.columns)}"
print(f"Loaded {INPUT_CSV} : {delta_table.shape}, engine={ENGINE} (D0)")

other_edges = sorted(set(delta_table["edge"].unique()) - {edge for edge, _ in EMPHASIS_EDGES})
edges_to_fit = list(EMPHASIS_EDGES) + ([(edge, "other") for edge in other_edges] if FIT_OTHER_EDGES else [])

contrast_rows = []
diagnostics_rows = []
forest_rows_by_edge = {}
edge_sd_lookup = {}

# ---------------------------------------------------------------------------
# 2. Per-edge models (confirmatory, D2 default)
# ---------------------------------------------------------------------------
for edge, edge_class in edges_to_fit:
    subset = edge_subset(delta_table, [edge])
    n_before = len(subset)
    subset = subset[subset["real_stable"]]
    n_dropped = n_before - len(subset)  # L8, reported even when zero
    subset = add_covariates(subset, ADD_AGE_COVARIATE, ADD_IAF_COVARIATE, IAF_METRICS_CSV, IAF_DISTANCE_COLUMN)
    subset, edge_stats = standardize_within_edge(subset, DV_MAIN)
    sd = float(edge_stats.loc[edge_stats["edge"] == edge, "sd"].iloc[0])
    edge_sd_lookup[edge] = sd

    formula = build_formula(f"{DV_MAIN}_z", EXTRA_TERMS)
    model, idata = fit_model(formula, subset, MCMC_CONFIG)
    az.to_netcdf(idata, MODELS_DIR / f"{safe_label(edge)}__baseline.nc")

    grid = reference_grid(FILMS, GROUPS, EXTRA_TERMS)
    contrasts = compute_contrasts(model, idata, grid, HDI_PROB)
    diagnostics_rows.append(convergence_row(f"{edge} ({edge_class})", idata, len(subset), n_dropped, RHAT_MAX, ESS_MIN, PARETO_K_MAX))

    forest_rows = []
    for contrast_name in ("group_effect", "film_contrast_overall", "film_contrast_TD", "film_contrast_ASD", "interaction_film_group"):
        summary_std = contrasts[contrast_name]
        summary_raw = back_transform(summary_std, sd)
        for unit, summary in (("std", summary_std), ("raw", summary_raw)):
            contrast_rows.append({
                "model": edge, "edge": edge, "edge_class": edge_class, "dv": DV_MAIN,
                "contrast": contrast_name, "unit": unit, **summary,
            })
        forest_rows.append({"label": f"{contrast_name} (raw)", **summary_raw})
    forest_rows_by_edge[edge] = forest_rows

    forest_fig = plot_forest(forest_rows, f"{edge} ({edge_class}) -- contrasts, raw Delta-units")
    forest_fig.savefig(QC_DIR / f"{safe_label(edge)}_forest.png")
    plt.close(forest_fig)

    ppc_fig = plot_ppc_figure(model, idata, f"{edge} pp_check")
    ppc_fig.savefig(QC_DIR / f"{safe_label(edge)}_ppcheck.png")
    plt.close(ppc_fig)

    loo_result = az.loo(idata, pointwise=True)
    loo_fig = plot_pareto_k_figure(loo_result, f"{edge} Pareto-k", PARETO_K_MAX)
    loo_fig.savefig(QC_DIR / f"{safe_label(edge)}_loo.png")
    plt.close(loo_fig)

    print(f"Fit {edge} ({edge_class}): n={len(subset)} dropped_unstable={n_dropped} "
          f"max_rhat={diagnostics_rows[-1]['max_rhat']:.3f} n_divergent={diagnostics_rows[-1]['n_divergent']} "
          f"max_pareto_k={diagnostics_rows[-1]['max_pareto_k']:.2f} pass_l9={diagnostics_rows[-1]['pass_l9']}")

# ---------------------------------------------------------------------------
# 3. Asymmetry track (L6): H2 and H4, primary DV z_vs_surrogate + sensitivity delta_dtf
# ---------------------------------------------------------------------------
ASYMMETRY_SPECS = asymmetry_specs_from_topology(CFG["edge_topology"])  # D3: derived, e.g. H2/H4 primary/reverse pairs
ASYMMETRY_DV_TRACKS = [("z_vs_surrogate", "native", "z"), (DV_MAIN, "std", "delta")]  # primary, sensitivity (L6)
asymmetry_summary_rows = []  # for run summary / gate primary section

for hypothesis, forward_edge, reverse_edge in ASYMMETRY_SPECS:
    for dv_col, unit_label, tag in ASYMMETRY_DV_TRACKS:
        asym = asymmetry_dv(delta_table, forward_edge, reverse_edge, dv_col)
        n_before = len(asym)
        asym = asym[asym["real_stable"]]
        n_dropped = n_before - len(asym)
        asym = add_covariates(asym, ADD_AGE_COVARIATE, ADD_IAF_COVARIATE, IAF_METRICS_CSV, IAF_DISTANCE_COLUMN)

        sd = None
        if unit_label == "std":
            asym = asym.assign(edge=f"{hypothesis}_asym_{tag}")
            asym, asym_stats = standardize_within_edge(asym, "asym")
            sd = float(asym_stats["sd"].iloc[0])
            fit_col = "asym_z"
        else:
            fit_col = "asym"

        formula = build_formula(fit_col, EXTRA_TERMS)
        model, idata = fit_model(formula, asym, MCMC_CONFIG)
        model_label = f"asym_{hypothesis}__{tag}"
        az.to_netcdf(idata, MODELS_DIR / f"{model_label}.nc")

        grid = reference_grid(FILMS, GROUPS, EXTRA_TERMS)
        contrasts = compute_contrasts(model, idata, grid, HDI_PROB)
        diagnostics_rows.append(convergence_row(f"{model_label} ({hypothesis} caregiver-leading, {dv_col})", idata, len(asym), n_dropped, RHAT_MAX, ESS_MIN, PARETO_K_MAX))

        forest_rows = []
        for contrast_name, summary_std in list(contrasts.items()):
            summary_report = back_transform(summary_std, sd) if sd is not None else summary_std
            contrast_rows.append({
                "model": model_label, "edge": f"{forward_edge} vs {reverse_edge}", "edge_class": f"{hypothesis}_asymmetry",
                "dv": dv_col, "contrast": contrast_name, "unit": ("raw" if sd is not None else unit_label), **summary_report,
            })
            label = "intercept_caregiver_leading" if contrast_name == "grand_mean" else contrast_name
            forest_rows.append({"label": f"{label} ({dv_col})", **summary_report})
            if contrast_name == "grand_mean":
                asymmetry_summary_rows.append({"hypothesis": hypothesis, "dv": dv_col, "tag": tag, **summary_report})

        forest_fig = plot_forest(forest_rows, f"{hypothesis} asymmetry ({dv_col}) -- caregiver-leading if intercept > 0")
        forest_fig.savefig(QC_DIR / f"{model_label}_forest.png")
        plt.close(forest_fig)

        ppc_fig = plot_ppc_figure(model, idata, f"{model_label} pp_check")
        ppc_fig.savefig(QC_DIR / f"{model_label}_ppcheck.png")
        plt.close(ppc_fig)

        print(f"Fit {model_label}: n={len(asym)} dropped_unstable={n_dropped} "
              f"max_rhat={diagnostics_rows[-1]['max_rhat']:.3f} pass_l9={diagnostics_rows[-1]['pass_l9']}")

# ---------------------------------------------------------------------------
# 4. Pooled cross-edge model (D2 companion, shrinkage cross-check)
# ---------------------------------------------------------------------------
pooled_diagnostics = None
pooled_contrasts = None
if FIT_POOLED_MODEL:
    pooled_frames = []
    for edge, edge_class in EMPHASIS_EDGES:
        subset = edge_subset(delta_table, [edge])
        subset = subset[subset["real_stable"]]
        subset = add_covariates(subset, ADD_AGE_COVARIATE, ADD_IAF_COVARIATE, IAF_METRICS_CSV, IAF_DISTANCE_COLUMN)
        subset, _ = standardize_within_edge(subset, DV_MAIN)
        pooled_frames.append(subset)
    pooled_data = pd.concat(pooled_frames, ignore_index=True)
    pooled_data["edge"] = pd.Categorical(pooled_data["edge"])

    pooled_formula = build_formula(f"{DV_MAIN}_z", EXTRA_TERMS, extra_grouping="edge")
    pooled_model, pooled_idata = fit_model(pooled_formula, pooled_data, MCMC_CONFIG)
    az.to_netcdf(pooled_idata, MODELS_DIR / "pooled_emphasis.nc")

    pooled_grid = reference_grid(FILMS, GROUPS, EXTRA_TERMS)
    pooled_contrasts = compute_contrasts(pooled_model, pooled_idata, pooled_grid, HDI_PROB)
    pooled_diagnostics = convergence_row(f"pooled_emphasis ({len(EMPHASIS_EDGES)} edges, (1|edge))", pooled_idata, len(pooled_data), 0, RHAT_MAX, ESS_MIN, PARETO_K_MAX)
    diagnostics_rows.append(pooled_diagnostics)
    pooled_funnel_paths = plot_edge_funnel(pooled_idata, "pooled_D2", QC_DIR)

    pooled_forest_rows = [
        {"label": name, **pooled_contrasts[name]}
        for name in ("group_effect", "film_contrast_overall", "film_contrast_TD", "film_contrast_ASD", "interaction_film_group")
    ]
    for row in pooled_forest_rows:
        contrast_rows.append({
            "model": "pooled_emphasis", "edge": "pooled(6 emphasis edges)", "edge_class": "pooled",
            "dv": DV_MAIN, "contrast": row["label"], "unit": "std",
            "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"],
            "p_gt0": row["p_gt0"], "p_lt0": row["p_lt0"],
        })
    pooled_forest_fig = plot_forest(pooled_forest_rows, "Pooled emphasis-edge model (standardized, shrinkage cross-check)")
    pooled_forest_fig.savefig(QC_DIR / "pooled_forest.png")
    plt.close(pooled_forest_fig)

    pooled_ppc_fig = plot_ppc_figure(pooled_model, pooled_idata, "pooled_emphasis pp_check")
    pooled_ppc_fig.savefig(QC_DIR / "pooled_ppcheck.png")
    plt.close(pooled_ppc_fig)

    print(f"Fit pooled_emphasis: n={len(pooled_data)} max_rhat={pooled_diagnostics['max_rhat']:.3f} "
          f"pass_l9={pooled_diagnostics['pass_l9']}")

# ---------------------------------------------------------------------------
# 5. Tidy CSV outputs
# ---------------------------------------------------------------------------
contrasts_df = pd.DataFrame(contrast_rows)
contrasts_df.to_csv(OUTPUT_DIR / "stage06_contrasts.csv", index=False)

diagnostics_df = pd.DataFrame(diagnostics_rows)
diagnostics_df.to_csv(OUTPUT_DIR / "stage06_diagnostics.csv", index=False)

primary_rows = []
p_like_values = []
for edge, contrast_name, tag in PRIMARY_FAMILY:
    row = contrasts_df[(contrasts_df["edge"] == edge) & (contrasts_df["contrast"] == contrast_name) & (contrasts_df["unit"] == "raw")].iloc[0]
    p_like = 2 * min(row["p_gt0"], row["p_lt0"])
    p_like_values.append(p_like)
    primary_rows.append({"family": "primary", "label": f"{tag} group_effect", **row.to_dict(), "p_like_2sided": p_like})
primary_summary_df = pd.DataFrame(primary_rows)
primary_summary_df["bh_fdr"] = bh_fdr(p_like_values)  # L7: BH applied within this named 2-member family only

for entry in asymmetry_summary_rows:
    label = f"{entry['hypothesis']}_asym_intercept_{entry['tag']}"
    primary_summary_df = pd.concat([primary_summary_df, pd.DataFrame([{
        "family": "asymmetry", "label": label, "model": f"asym_{entry['hypothesis']}__{entry['tag']}",
        "edge": None, "edge_class": f"{entry['hypothesis']}_asymmetry", "dv": entry["dv"], "contrast": "grand_mean",
        "unit": "raw" if entry["tag"] == "delta" else "native", "estimate": entry["estimate"],
        "hdi_low": entry["hdi_low"], "hdi_high": entry["hdi_high"], "p_gt0": entry["p_gt0"], "p_lt0": entry["p_lt0"],
        "p_like_2sided": np.nan, "bh_fdr": np.nan,
    }])], ignore_index=True)
primary_summary_df.to_csv(OUTPUT_DIR / "stage06_primary_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Primary-contrasts and asymmetry summary figures
# ---------------------------------------------------------------------------
primary_forest_rows = [
    {"label": row["label"], "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
    for row in primary_rows
]
primary_fig = plot_forest(primary_forest_rows, "H2/H4 primary group effects (raw Delta-units)")
primary_fig.savefig(QC_DIR / "primary_contrasts.png")
plt.close(primary_fig)

asymmetry_forest_rows = [
    {"label": f"{entry['hypothesis']} caregiver-leading ({entry['dv']})", "estimate": entry["estimate"],
     "hdi_low": entry["hdi_low"], "hdi_high": entry["hdi_high"]}
    for entry in asymmetry_summary_rows
]
asymmetry_fig = plot_forest(asymmetry_forest_rows, "H2/H4 asymmetry: intercept > 0 = caregiver-leading")
asymmetry_fig.savefig(QC_DIR / "asymmetry.png")
plt.close(asymmetry_fig)

# ---------------------------------------------------------------------------
# 7. Run summary
# ---------------------------------------------------------------------------
summary_lines = [
    f"engine (D0) = {ENGINE}  |  RE structure (D1) = (1|dyad_id) only -- (1|child)+(1|caregiver) dropped, not identifiable (members 1:1 with dyad)",
    f"H4 group-effect sign is reported two-sided; TD>ASD is NOT assumed (D5).",
    f"MCMC config (Stage 6 addendum, L-conv-1): TARGET_ACCEPT={TARGET_ACCEPT} (was 0.95), TUNE={TUNE} (was 1000) "
    f"-- applied to every model in this run, per-edge through pooled/M1/M2/M3.",
    f"Sampler cores: cores={CORES} (sequential, no worker subprocesses) -- macOS fix: fork() of a process "
    f"with live Accelerate/vecLib BLAS threads segfaults the child pre-exec, and spawn/forkserver both "
    f"re-import this unguarded top-level script as __main__ during worker bootstrap, recursing into the "
    f"whole pipeline; applied to every model.fit() call in this run.",
    "",
    "Per-edge convergence (L9: rhat<{:.2f}, ess>{}, 0 divergences, pareto_k<{:.1f}):".format(RHAT_MAX, ESS_MIN, PARETO_K_MAX),
]
for row in diagnostics_rows:
    status = "PASS" if row["pass_l9"] else "FAIL"
    summary_lines.append(
        f"  [{status}] {row['model']}: n={row['n_rows']} dropped_unstable={row['n_dropped_unstable']} "
        f"max_rhat={row['max_rhat']:.3f} min_ess_bulk={row['min_bulk_ess']:.0f} min_ess_tail={row['min_tail_ess']:.0f} "
        f"n_divergent={row['n_divergent']} max_pareto_k={row['max_pareto_k']:.2f} loo_elpd={row['loo_elpd']:.1f}"
    )

summary_lines.append("")
summary_lines.append("H2/H4 primary group effects (raw Delta-units), BH-FDR over this 2-member family only:")
for _, row in primary_summary_df[primary_summary_df["family"] == "primary"].iterrows():
    summary_lines.append(
        f"  {row['label']}: estimate={row['estimate']:+.5f} HDI95=[{row['hdi_low']:+.5f}, {row['hdi_high']:+.5f}] "
        f"P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f} bh_fdr={row['bh_fdr']:.3f}"
    )

summary_lines.append("")
summary_lines.append("H2/H4 asymmetry (caregiver-leading if intercept > 0):")
for _, row in primary_summary_df[primary_summary_df["family"] == "asymmetry"].iterrows():
    summary_lines.append(
        f"  {row['label']}: estimate={row['estimate']:+.5f} HDI95=[{row['hdi_low']:+.5f}, {row['hdi_high']:+.5f}] "
        f"P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
    )

failed_models = [row["model"] for row in diagnostics_rows if not row["pass_l9"]]
summary_lines.append("")
if failed_models:
    summary_lines.append(f"L9 FAILURES ({len(failed_models)}): {', '.join(failed_models)} -- consider the 2-variable fallback (project note S11).")
else:
    summary_lines.append("L9: all models pass convergence/PPC/LOO criteria.")

print("\n" + "\n".join(summary_lines))
(OUTPUT_DIR / "stage06_run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# 8. Interactive HTML gate (mirrors Stage 5's surrogate_gate.html style)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 6 group model gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .edge-panel { display: none; }
  .edge-panel.active { display: block; }
  .header-line { font-family: monospace; margin-bottom: 0.5em; }
  .badge { padding: 0.1em 0.5em; border-radius: 3px; color: white; font-size: 0.85em; }
  .badge-ok { background: #2e8b2e; }
  .badge-bad { background: #b03030; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 900px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
  table.contrasts, table.diag { border-collapse: collapse; font-size: 0.85em; margin-bottom: 1em; }
  table.contrasts th, table.contrasts td, table.diag th, table.diag td { border: 1px solid #ccc; padding: 0.2em 0.5em; text-align: right; }
  table.contrasts th:first-child, table.contrasts td:first-child,
  table.diag th:first-child, table.diag td:first-child { text-align: left; }
  .callout { background: #fff6e0; border: 1px solid #e0c060; padding: 0.6em 1em; margin-bottom: 1em; }
</style>
</head>
<body>
<h1>Stage 6 group model gate</h1>
<p class="header-line">
   engine (D0) = <b>__ENGINE__</b> (brms/cmdstanr not installed; bambi+arviz already were -- ratified by project owner).
   DV = standardized <b>delta_dtf</b> (per-edge z-score, D3), family=<b>Student-t</b>,
   formula=<b>film * group (sum contrasts) + (1|dyad_id)</b> (D1), planned film contrast =
   <b>Incredibles vs (Peppa+Brave)/2</b> (two-sided), FDR family (L7) = {H2_primary, H4_primary} group effects only.<br>
   input = <b>__INPUT_CSV__</b>. ROI (edges named <code>*:ROI</code>) = <b>__ROI_LABEL__</b>,
   electrodes <b>__ROI_CHANNELS__</b> (must match Stage 2's ROI_LABEL/ROI_CHANNELS).
   estimator = <b>__ESTIMATOR__</b> (must match Stage 5's ESTIMATOR -- not read from INPUT_CSV, which carries no
   ESTIMATOR column). box_cox_lambda = <b>__BOX_COX_LAMBDA__</b> (-1 = no transform; must match Stage 5's
   BOX_COX_LAMBDA, same not-read-from-data caveat).
</p>
<div class="callout">
  <b>D1 (correctness fix):</b> the plan's <code>(1|dyad_id) + (1|child_id) + (1|caregiver_id)</code> was dropped to
  <code>(1|dyad_id)</code> only -- in this design child and caregiver are 1:1 with dyad, so the three grouping factors
  are the same partition of the rows and are not jointly identifiable. Flagging for Jarek to ratify or override.<br>
  <b>H4 sign:</b> reported two-sided everywhere -- <code>P(ASD&gt;TD)</code> and <code>P(ASD&lt;TD)</code> both shown.
  TD &gt; ASD is NOT assumed (D5); negative interpersonal HRV synchrony can be adaptive.
</div>
<h2>Convergence (L9 gate)</h2>
<div id="diagnostics">__DIAGNOSTICS_TABLE__</div>
<h2>Primary contrasts</h2>
<div class="row">
  <img src="qc/primary_contrasts.png" alt="H2/H4 primary group effects">
  <img src="qc/asymmetry.png" alt="H2/H4 asymmetry">
</div>
<div id="primary">__PRIMARY_TABLE__</div>
<h2>Run summary</h2>
<div id="summary">__SUMMARY__</div>
<h2>Per-edge diagnostics</h2>
<label for="edge-select">Edge: </label>
<select id="edge-select"></select>
<div id="panels">__PANELS__</div>
__POOLED_SECTION__
<script>
const edgeIds = __EDGE_IDS_JSON__;
const select = document.getElementById('edge-select');
for (const id of edgeIds) {
  const opt = document.createElement('option');
  opt.value = id; opt.textContent = id;
  select.appendChild(opt);
}
function showEdge(id) {
  document.querySelectorAll('.edge-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  if (panel) panel.classList.add('active');
}
select.onchange = () => showEdge(select.value);
if (edgeIds.length) showEdge(edgeIds[0]);
</script>
</body>
</html>
"""


panels_html = "\n".join(render_edge_panel(edge, edge_class, contrasts_df, safe_label) for edge, edge_class in edges_to_fit)
edge_ids = [edge for edge, _ in edges_to_fit]

pooled_section = ""
if FIT_POOLED_MODEL:
    pooled_rows_df = contrasts_df[contrasts_df["model"] == "pooled_emphasis"]
    pooled_section = (
        '<h2>Pooled cross-edge model (D2 shrinkage cross-check)</h2>'
        '<p>Standardized units only (edges have different raw scales); a companion to the per-edge results above, not a replacement.</p>'
        '<div class="row"><img src="qc/pooled_forest.png" alt="pooled forest">'
        '<img src="qc/pooled_ppcheck.png" alt="pooled pp_check"></div>'
        + render_contrast_table(pooled_rows_df)
    )

html = HTML_TEMPLATE.replace("__DIAGNOSTICS_TABLE__", render_diagnostics_table(diagnostics_rows))
html = html.replace("__PRIMARY_TABLE__", render_primary_table(primary_summary_df))
html = html.replace("__SUMMARY__", "\n".join(summary_lines))
html = html.replace("__PANELS__", panels_html)
html = html.replace("__POOLED_SECTION__", pooled_section)
html = html.replace("__EDGE_IDS_JSON__", json.dumps(edge_ids))
html = html.replace("__ENGINE__", ENGINE)
html = html.replace("__INPUT_CSV__", str(INPUT_CSV.relative_to(PROJECT_ROOT)))
html = html.replace("__ROI_LABEL__", ROI_LABEL)
html = html.replace("__ROI_CHANNELS__", "/".join(ROI_CHANNELS))
html = html.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
(OUTPUT_DIR / "group_model_gate.html").write_text(html, encoding="utf-8")
print(f"\nWrote gate to {OUTPUT_DIR / 'group_model_gate.html'}")

# ---------------------------------------------------------------------------
# D6 (open, exploratory) -- cross-edge localization of the film x group
# interaction. Appended after the Stage 6 gate above; nothing above this line
# is touched.
#
# The pooled model (Section 4) found a credible film x group interaction in
# standardized units, while the per-edge H2_primary model shows none -- so
# the pooled interaction is a cross-edge signal that does not live on the
# primary H2 edge. This is NOT a two-stage omnibus-then-post-hoc test: the
# pooled fixed effect and any per-edge re-test would be separate estimators
# with no shared multiplicity protection (the classic forking-paths failure
# mode). Instead this fits one richer hierarchy that lets the film x group
# interaction vary by edge (M1, primary) and, as a heavier sensitivity check,
# lets the whole film*group surface vary by edge (M2, gated by
# `FIT_M2_FULL_VARYING`). Both are compared to the interaction-fixed pooled
# model (M0 = `pooled_model`/`pooled_idata` from Section 4, reused unrefit)
# via `az.compare` (LOO); per-edge localization is read off M1's shrunk
# group-specific deviations, warranted only if letting the interaction vary
# actually improves predictive fit.
#
# Caveats (surfaced again in the gate section below):
#  - Exploratory, uncorrected. The interaction was never in the L7 FDR family
#    (PRIMARY_FAMILY = the two H2/H4 GROUP effects only). Every quantity here
#    is hypothesis-generating, not confirmatory.
#  - Six edges is very little for a variance component -- the edge-level SD
#    is estimated from 6 groups, so per-edge deviations are heavily shrunk
#    and the `az.compare` ranking is low-powered. A "win" for the varying
#    model is weak evidence.
#  - `(...|edge)` treats the 6 emphasis edges as exchangeable, which is
#    scientifically questionable (interbrain-EEG, interbrain-HRV, and cross
#    brain-heart edges are qualitatively different objects). A typed
#    grouping could be more defensible -- flagged as an open modelling
#    choice, not resolved here.
#  - Standardized units only (`unit="std"`), same as the pooled model -- no
#    raw back-transform across edges with different raw scales.
#  - Signed everywhere. Never abs() or clip any interaction/contrast.
# ---------------------------------------------------------------------------
if not FIT_POOLED_MODEL:
    raise RuntimeError("D6 requires the D2 pooled model -- set FIT_POOLED_MODEL=True.")

FIT_M2_FULL_VARYING = True  # D6 sensitivity: full film*group | edge (heavy, 6-edge-limited)

edge_class_lookup = dict(EMPHASIS_EDGES)






edge_categories = list(pooled_data["edge"].cat.categories)
localization_diagnostics_rows = []
group_sd_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))

# --- M1: interaction varies by edge (primary D6 model) ---------------------
m1_formula = build_formula(f"{DV_MAIN}_z", EXTRA_TERMS, extra_grouping="edge") + " + (0 + C(film, Sum):C(group, Sum) | edge)"
m1_priors = {**build_priors(SD_PRIOR_EDGE), "C(film, Sum):C(group, Sum)|edge": group_sd_prior}
m1_model, m1_idata = fit_model_with_priors(
    m1_formula, pooled_data, m1_priors, required_group_terms=["C(film, Sum):C(group, Sum)|edge"], mcmc_config=MCMC_CONFIG,
)
az.to_netcdf(m1_idata, MODELS_DIR / "pooled_emphasis_varying_interaction.nc")
localization_diagnostics_rows.append(convergence_row("pooled_varying_interaction (D6, M1)", m1_idata, len(pooled_data), 0, RHAT_MAX, ESS_MIN, PARETO_K_MAX))
m1_funnel_paths = plot_edge_funnel(m1_idata, "M1", QC_DIR)
print(f"Fit D6 M1 (interaction varies by edge): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
      f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- M2: full film*group varies by edge (sensitivity, gated) ---------------
m2_model, m2_idata = None, None
if FIT_M2_FULL_VARYING:
    m2_formula = (
        build_formula(f"{DV_MAIN}_z", EXTRA_TERMS, extra_grouping="edge")
        + " + (0 + C(film, Sum) + C(group, Sum) + C(film, Sum):C(group, Sum) | edge)"
    )
    m2_priors = {
        **build_priors(SD_PRIOR_EDGE),
        "C(film, Sum)|edge": group_sd_prior,
        "C(group, Sum)|edge": group_sd_prior,
        "C(film, Sum):C(group, Sum)|edge": group_sd_prior,
    }
    m2_model, m2_idata = fit_model_with_priors(
        m2_formula, pooled_data, m2_priors,
        required_group_terms=["C(film, Sum)|edge", "C(group, Sum)|edge", "C(film, Sum):C(group, Sum)|edge"], mcmc_config=MCMC_CONFIG,
    )
    az.to_netcdf(m2_idata, MODELS_DIR / "pooled_emphasis_varying_full.nc")
    localization_diagnostics_rows.append(convergence_row("pooled_varying_full (D6, M2)", m2_idata, len(pooled_data), 0, RHAT_MAX, ESS_MIN, PARETO_K_MAX))
    print(f"Fit D6 M2 (full film*group varies by edge): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
          f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- M3: edge as a FIXED factor (no pooling) -- D-conv-A --------------------
# Un-shrunk counterpart to M0's full pooling (M1 reads between the two). By
# construction there is no (1|edge)/(...|edge) hyperprior, so M3 cannot
# funnel; plot_edge_funnel is deliberately NOT called on it (its assert would
# correctly fire on a model with no edge SD term).
m3_model, m3_idata = None, None
if FIT_EDGE_AS_FIXED:
    extra = "".join(f" + {t}" for t in EXTRA_TERMS)
    m3_formula = f"{DV_MAIN}_z ~ C(film, Sum) * C(group, Sum) * C(edge, Sum){extra} + (1|dyad_id)"
    # Fixed three-way crossing -> enumerate the common terms bambi actually
    # builds and set each to Normal(0,1); catch any silently-defaulted term
    # loudly rather than fitting with an unreviewed default prior.
    m3_probe = bmb.Model(m3_formula, pooled_data, family="t")
    common_terms = common_terms_of(m3_probe)
    m3_priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "1|dyad_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1)),
        "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1),
        **{name: bmb.Prior("Normal", mu=0, sigma=1) for name in common_terms},
    }
    m3_model, m3_idata = fit_model_with_priors(
        m3_formula, pooled_data, m3_priors, required_group_terms=[], mcmc_config=MCMC_CONFIG,
    )
    az.to_netcdf(m3_idata, MODELS_DIR / "pooled_emphasis_edge_fixed.nc")
    localization_diagnostics_rows.append(
        convergence_row("edge_fixed (D6, M3, no pooling)", m3_idata, len(pooled_data), 0, RHAT_MAX, ESS_MIN, PARETO_K_MAX))
    print(f"Fit D6 M3 (edge as fixed factor): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
          f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- loo_compare: the localization warrant ----------------------------------
models_for_loo = {"interaction_fixed": pooled_idata, "interaction_varying": m1_idata}
if FIT_M2_FULL_VARYING:
    models_for_loo["full_varying"] = m2_idata
if FIT_EDGE_AS_FIXED:
    models_for_loo["edge_fixed"] = m3_idata  # D-conv-C: reference point only, never drives the heterogeneity verdict
loo_compare = az.compare(models_for_loo, ic="loo")
compare_reset = loo_compare.reset_index().rename(columns={"index": "model"})
compare_reset.to_csv(OUTPUT_DIR / "stage06_localization_compare.csv", index=False)

compare_fig, compare_ax = plt.subplots(figsize=(6, 2.5))
az.plot_compare(loo_compare, ax=compare_ax)
compare_fig.tight_layout()
compare_fig.savefig(QC_DIR / "localization_loo_compare.png")
plt.close(compare_fig)

best_model = loo_compare.index[0]
second_row = loo_compare.iloc[1]
elpd_diff = float(second_row["elpd_diff"])
dse = float(second_row["dse"])

if best_model == "interaction_varying" and elpd_diff > 2 * dse:
    verdict = f"cross-edge heterogeneity supported; per-edge localization is interpretable (weakly, n={len(EMPHASIS_EDGES)} edges)."
elif best_model == "interaction_fixed":
    verdict = "no support for heterogeneity; the pooled interaction is best read as a single shared effect, not localized."
elif best_model == "full_varying":
    m2_pass = localization_diagnostics_rows[-1]["pass_l9"]
    if m2_pass:
        verdict = (f"the richer surface (M2) fits best, but at {len(EMPHASIS_EDGES)} edges M2 is over-parameterized; treat M1's "
                   "per-edge interaction as the estimate and M2 as agreement/robustness only, not the reference.")
    else:
        verdict = ("M2 ranks best by ELPD but FAILED the L9 convergence gate -- this ranking is not trustworthy; "
                   "fall back to M1 as the localization model of record.")
elif best_model == "edge_fixed":
    verdict = ("the no-pooling M3 (edge fixed) ranks first, but it is a different model class "
               "(un-shrunk edge estimates) and is NOT evidence of shrinkage-model heterogeneity; "
               "read the M0/M1/M2 comparison for the heterogeneity verdict, with M3 as an un-shrunk reference only.")
else:
    verdict = "inconclusive; models indistinguishable in predictive fit."

if bool(loo_compare["warning"].any()):
    verdict += " NOTE: az.compare raised a Pareto-k warning for at least one model -- treat this comparison itself as unreliable."

print(f"D6 loo_compare verdict: {verdict}")



# --- per-edge localization table (from one posterior each, M1 and M2) ------
localization_rows = []




localization_rows.extend(compute_localization_rows(
    "M1_interaction_varying", m1_model, m1_idata, [e for e, _ in EMPHASIS_EDGES], edge_categories,
    edge_class_lookup, pooled_data, FILMS, GROUPS, EXTRA_TERMS, HDI_PROB,
))
if FIT_M2_FULL_VARYING:
    localization_rows.extend(compute_localization_rows(
        "M2_full_varying", m2_model, m2_idata, [e for e, _ in EMPHASIS_EDGES], edge_categories,
        edge_class_lookup, pooled_data, FILMS, GROUPS, EXTRA_TERMS, HDI_PROB,
    ))
if FIT_EDGE_AS_FIXED:
    localization_rows.extend(compute_localization_rows(
        "M3_edge_fixed", m3_model, m3_idata, [e for e, _ in EMPHASIS_EDGES], edge_categories,
        edge_class_lookup, pooled_data, FILMS, GROUPS, EXTRA_TERMS, HDI_PROB,
    ))

localization_df = pd.DataFrame(localization_rows)
localization_df.to_csv(OUTPUT_DIR / "stage06_localization.csv", index=False)

interaction_rows_m1 = localization_df[
    (localization_df["model"] == "M1_interaction_varying") & (localization_df["contrast"] == "interaction_film_group")
]
localized_edges = [
    row["edge"] for _, row in interaction_rows_m1.iterrows()
    if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
]

localization_forest_rows = [
    {"label": f"{row['edge']} ({row['edge_class']})" if row["edge"] != "population" else "population (M1)",
     "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
    for _, row in interaction_rows_m1.iterrows()
]

interaction_rows_m2 = None
m2_localized_edges = []
if FIT_M2_FULL_VARYING:
    interaction_rows_m2 = localization_df[
        (localization_df["model"] == "M2_full_varying") & (localization_df["contrast"] == "interaction_film_group")
    ]
    m2_localized_edges = [
        row["edge"] for _, row in interaction_rows_m2.iterrows()
        if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
    ]
    localization_forest_rows += [
        {"label": f"{row['edge']} -- M2" if row["edge"] != "population" else "population (M2)",
         "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
        for _, row in interaction_rows_m2.iterrows()
    ]

interaction_rows_m3 = None
m3_localized_edges = []
if FIT_EDGE_AS_FIXED:
    interaction_rows_m3 = localization_df[
        (localization_df["model"] == "M3_edge_fixed") & (localization_df["contrast"] == "interaction_film_group")
    ]
    m3_localized_edges = [
        row["edge"] for _, row in interaction_rows_m3.iterrows()
        if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
    ]
    localization_forest_rows += [
        {"label": f"{row['edge']} -- M3" if row["edge"] != "population" else "population (M3)",
         "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
        for _, row in interaction_rows_m3.iterrows()
    ]

localization_forest_title = "D6: per-edge film x group interaction (standardized) -- M1 primary" + (
    ", M2 sensitivity below" if FIT_M2_FULL_VARYING else ""
) + (", M3 (un-shrunk, no pooling) below" if FIT_EDGE_AS_FIXED else "")
localization_forest_fig = plot_forest(localization_forest_rows, localization_forest_title)
localization_forest_fig.savefig(QC_DIR / "localization_interaction_forest.png")
plt.close(localization_forest_fig)

# --- D6 summary text ---------------------------------------------------------
localization_summary_lines = [
    "D6 (open, exploratory) -- cross-edge localization of the film x group interaction.",
    "Not in the L7 FDR family; hypothesis-generating only. Standardized units (unit=std) throughout, signed.",
    f"Sampler/prior settings (Stage 6 addendum): TARGET_ACCEPT={TARGET_ACCEPT}, TUNE={TUNE}, "
    f"SD_PRIOR_EDGE={SD_PRIOR_EDGE!r} (scopes 1|edge only), FIT_EDGE_AS_FIXED={FIT_EDGE_AS_FIXED}.",
    "",
    "loo_compare:",
]
for _, row in compare_reset.iterrows():
    localization_summary_lines.append(
        f"  {row['model']}: rank={int(row['rank'])} elpd_loo={row['elpd_loo']:.2f} p_loo={row['p_loo']:.2f} "
        f"elpd_diff={row['elpd_diff']:.2f} dse={row['dse']:.2f} weight={row['weight']:.3f} warning={row['warning']}"
    )
localization_summary_lines.append(f"  verdict: {verdict}")
localization_summary_lines.append("")
localization_summary_lines.append("M1/M2 convergence:")
for row in localization_diagnostics_rows:
    status = "PASS" if row["pass_l9"] else "FAIL"
    localization_summary_lines.append(
        f"  [{status}] {row['model']}: max_rhat={row['max_rhat']:.3f} min_ess_bulk={row['min_bulk_ess']:.0f} "
        f"n_divergent={row['n_divergent']} max_pareto_k={row['max_pareto_k']:.2f}"
    )
localization_summary_lines.append("")
localization_summary_lines.append("M1 per-edge film x group interaction (standardized):")
for _, row in interaction_rows_m1.iterrows():
    localization_summary_lines.append(
        f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
        f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
    )
if FIT_M2_FULL_VARYING:
    localization_summary_lines.append("")
    localization_summary_lines.append("M2 per-edge film x group interaction (standardized, sensitivity):")
    for _, row in interaction_rows_m2.iterrows():
        localization_summary_lines.append(
            f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
            f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
        )
if FIT_EDGE_AS_FIXED:
    localization_summary_lines.append("")
    localization_summary_lines.append("M3 per-edge film x group interaction (standardized, un-shrunk, no pooling):")
    for _, row in interaction_rows_m3.iterrows():
        localization_summary_lines.append(
            f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
            f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
        )
localization_summary_lines.append("")
localization_summary_lines.append(
    f"M1 edges with interaction HDI excluding zero (which edges carry the effect): "
    f"{', '.join(localized_edges) if localized_edges else 'none'}"
)
if FIT_M2_FULL_VARYING:
    localization_summary_lines.append(
        f"M2 agreement on that set: {', '.join(m2_localized_edges) if m2_localized_edges else 'none'}"
    )
if FIT_EDGE_AS_FIXED:
    localization_summary_lines.append(
        f"M3 (un-shrunk) agreement on that set: {', '.join(m3_localized_edges) if m3_localized_edges else 'none'}"
    )

print("\n" + "\n".join(localization_summary_lines))
(OUTPUT_DIR / "stage06_localization_summary.txt").write_text("\n".join(localization_summary_lines), encoding="utf-8")

# --- D6 gate section (appended into the already-written gate HTML) ---------



m1_rows_gate = localization_df[localization_df["model"] == "M1_interaction_varying"]
if FIT_M2_FULL_VARYING:
    m2_rows_gate = localization_df[localization_df["model"] == "M2_full_varying"]
    m2_gate_section = (
        '<h4>M2 sensitivity (full film*group varies by edge -- heavier, agreement check only)</h4>'
        + render_localization_table(m2_rows_gate)
    )
else:
    m2_gate_section = "<h4>M2 sensitivity</h4><p>not fit (FIT_M2_FULL_VARYING=False)</p>"

if FIT_EDGE_AS_FIXED:
    m3_rows_gate = localization_df[localization_df["model"] == "M3_edge_fixed"]
    m3_gate_section = (
        '<h4>M3 (edge as a fixed factor -- un-shrunk, no pooling; D-conv-A)</h4>'
        '<p>The un-shrunk counterpart to M0\'s full pooling, with M1 read between them. '
        'By construction M3 has no <code>(...|edge)</code> hyperprior and cannot funnel; it is shown as an ELPD '
        'reference point only (D-conv-C) and never drives the heterogeneity verdict above.</p>'
        + render_localization_table(m3_rows_gate)
    )
else:
    m3_gate_section = "<h4>M3 (edge fixed)</h4><p>not fit (FIT_EDGE_AS_FIXED=False)</p>"

funnel_row_items = "".join(
    f'<img src="qc/{p.name}" alt="{p.stem}">' for p in (pooled_funnel_paths + m1_funnel_paths)
)

d6_section = f"""
<h2>Cross-edge localization of the film&times;group interaction (D6 &mdash; exploratory, not confirmatory)</h2>
<div class="callout">
  <b>Exploratory, uncorrected:</b> the film&times;group interaction was never in the L7 FDR family (only the two
  H2/H4 group effects are). Every quantity in this section is hypothesis-generating.<br>
  <b>Low power:</b> the edge-level SD is estimated from only 6 edges; per-edge deviations are heavily shrunk and
  the loo_compare ranking below is low-powered.<br>
  <b>Exchangeability:</b> <code>(...|edge)</code> treats the 6 emphasis edges as exchangeable, which is
  scientifically questionable (interbrain-EEG, interbrain-HRV, and cross brain-heart edges are qualitatively
  different) -- a typed grouping could be more defensible; not resolved here.<br>
  <b>Units:</b> standardized only (<code>unit=std</code>); no raw back-transform across edges.<br>
  <b>Model status:</b> <b>M1</b> (interaction varies by edge) is primary. <b>M2</b> (full film*group varies by
  edge) is a heavier, 6-edge-limited sensitivity check shown for agreement only. <b>M3</b> (edge as a fixed
  factor, un-shrunk) sits alongside M0/M1/M2 as the no-pooling end of the shrinkage spectrum
  (M0 fully pooled &rarr; M1 partially pooled &rarr; M3 un-shrunk) -- never promoted to the reference answer on
  an ELPD win alone.<br>
  <b>Convergence-fix settings (Stage 6 addendum):</b> TARGET_ACCEPT=<code>{TARGET_ACCEPT}</code>,
  TUNE=<code>{TUNE}</code>, SD_PRIOR_EDGE=<code>{SD_PRIOR_EDGE}</code> (scopes the <code>1|edge</code> SD only).
</div>
<p><b>loo_compare verdict:</b> {verdict}</p>
<div class="row"><img src="qc/localization_loo_compare.png" alt="D6 loo_compare"></div>
<h3>M1/M2/M3 convergence</h3>
{render_diagnostics_table(localization_diagnostics_rows)}
<h3>(1|edge) funnel diagnostic</h3>
<p>Divergences clustered at small SD = confirmed funnel, treatable by the SD_PRIOR_EDGE / target_accept levers;
divergences elsewhere = look beyond the edge geometry. M3 has no edge SD term and is intentionally not plotted.</p>
<div class="row">{funnel_row_items}</div>
<h3>Per-edge interaction (standardized)</h3>
<div class="row"><img src="qc/localization_interaction_forest.png" alt="D6 per-edge interaction forest"></div>
<h4>M1 (primary)</h4>
{render_localization_table(m1_rows_gate)}
{m2_gate_section}
{m3_gate_section}
"""

gate_path = OUTPUT_DIR / "group_model_gate.html"
gate_html = gate_path.read_text(encoding="utf-8")
gate_html = gate_html.replace("</body>", d6_section + "\n</body>")
gate_path.write_text(gate_html, encoding="utf-8")
print(f"\nAppended D6 localization section to {gate_path}")
