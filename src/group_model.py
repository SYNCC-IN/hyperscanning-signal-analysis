"""Reusable helpers for the Stage 6 group model (Delta ffDTF vs surrogate).

Small, literal-free pieces shared between the model-fitting script
(`scripts/stage06_group_model.py`) and anything downstream that needs to read
the same tidy table the same way: category ordering, edge selection,
within-edge standardization (so standardized contrasts can be back-transformed
to raw Delta-units), the forward-minus-reverse asymmetry DV, a small
Benjamini-Hochberg helper for the one named primary family, and the bambi
model-fitting/contrast/diagnostic-plot functions themselves. Every MCMC
setting (draws/tune/chains/target_accept/seed/cores), pass-criterion, and
covariate toggle is passed in explicitly (usually bundled into a small dict,
e.g. `mcmc_config`) rather than read from a module-level constant -- the
script owns those values, this module only consumes them.
"""

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bambi.terms.group_specific import GroupSpecificTerm


def load_delta_table(csv_path):
    """Load Stage 5's tidy delta table with `film`/`group` as ordered categoricals.

    Parameters
    ----------
    csv_path : str or Path
        Path to `stage05_delta_table.csv`.

    Returns
    -------
    pd.DataFrame
        Same columns as the CSV. `film` is a categorical ordered
        `[Peppa, Incredibles, Brave]` (a fixed factor, not a scale -- ordering
        only fixes label order in tables/plots). `group` is categorical
        `[TD, ASD]`. `dyad_id` and `edge_class` are plain categoricals.
    """
    df = pd.read_csv(csv_path)
    df["film"] = pd.Categorical(df["film"], categories=["Peppa", "Incredibles", "Brave"])
    df["group"] = pd.Categorical(df["group"], categories=["TD", "ASD"])
    df["dyad_id"] = pd.Categorical(df["dyad_id"])
    df["edge_class"] = pd.Categorical(df["edge_class"])
    return df


def edge_subset(df, edges):
    """Rows of `df` whose `edge` matches any entry in `edges`.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (or a subset of it), with an `edge` column
        formatted as `"{source}->{target}"`.
    edges : list of str or list of tuple
        Each item is either an `"source->target"` string or a
        `(source, target)` tuple.

    Returns
    -------
    pd.DataFrame
        Matching rows (copy).
    """
    edge_strings = [e if isinstance(e, str) else f"{e[0]}->{e[1]}" for e in edges]
    return df[df["edge"].isin(edge_strings)].copy()


def standardize_within_edge(df, value_col):
    """Z-score `value_col` within each `edge` group.

    Standardizing per edge (rather than globally) is what makes the model's
    weakly-informative priors, set on a standardized scale, actually
    weakly-informative for every edge regardless of that edge's raw magnitude
    (see Stage 6 decision D3). The returned per-edge mean/sd let a caller
    back-transform standardized contrasts to raw Delta-units.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an `edge` column and `value_col`.
    value_col : str
        Column to standardize.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with an added `{value_col}_z` column.
    pd.DataFrame
        Per-edge `mean`/`sd` used, columns `edge`, `mean`, `sd`.
    """
    stats = df.groupby("edge", observed=True)[value_col].agg(["mean", "std"]).rename(columns={"std": "sd"})
    stats_reset = stats.reset_index()
    out = df.merge(stats, on="edge", how="left")
    out[f"{value_col}_z"] = (out[value_col] - out["mean"]) / out["sd"]
    out = out.drop(columns=["mean", "sd"])
    return out, stats_reset


def asymmetry_dv(df, forward_edge, reverse_edge, value_col):
    """Per dyad x film forward-minus-reverse asymmetry of `value_col`.

    `asym = value(forward_edge) - value(reverse_edge)`, signed: a positive
    value means the forward direction (caregiver->child for the H2/H4 primary
    edges) dominates -- "caregiver-leading" (Stage 6 L6).

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (all edges).
    forward_edge, reverse_edge : str or tuple
        Single-edge identifiers, as accepted by `edge_subset`.
    value_col : str
        Column to difference (e.g. `z_vs_surrogate` or `delta_dtf`).

    Returns
    -------
    pd.DataFrame
        One row per `dyad_id` x `film`: `dyad_id`, `film`, `group`,
        `age_months`, `asym`, `real_stable` (True only if both directions are
        stable for that dyad x film).
    """
    forward = edge_subset(df, [forward_edge]).set_index(["dyad_id", "film"])
    reverse = edge_subset(df, [reverse_edge]).set_index(["dyad_id", "film"])
    merged = forward[[value_col, "group", "age_months", "real_stable"]].join(
        reverse[[value_col, "real_stable"]], lsuffix="_fwd", rsuffix="_rev"
    )
    merged["asym"] = merged[f"{value_col}_fwd"] - merged[f"{value_col}_rev"]
    merged["real_stable"] = merged["real_stable_fwd"] & merged["real_stable_rev"]
    return merged.reset_index()[["dyad_id", "film", "group", "age_months", "asym", "real_stable"]]


def bh_fdr(pvalue_like):
    """Benjamini-Hochberg adjusted values for a small named family.

    Intended for a directional-probability-derived score such as
    `2 * min(P(effect > 0), P(effect < 0))` -- an analog p-value, not a
    frequentist one. The caller is responsible for documenting that; this
    function applies the standard BH step-up procedure exactly as it would to
    real p-values.

    Parameters
    ----------
    pvalue_like : array-like
        p-value-like scores, one per hypothesis in the family.

    Returns
    -------
    np.ndarray
        BH-adjusted values, same order as the input.
    """
    values = np.asarray(pvalue_like, dtype=float)
    n = values.size
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def add_covariates(df, add_age_covariate, add_iaf_covariate, iaf_metrics_csv=None, iaf_distance_column=None):
    """Add mean-centred covariate columns to `df` per the D4 toggles.

    A no-op copy when both toggles are off (the default/primary path). When a
    toggle is on and its source file/column is missing, `pd.read_csv`/column
    lookup raises naturally -- no silent skip.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (or a subset).
    add_age_covariate : bool
    add_iaf_covariate : bool
    iaf_metrics_csv : str or Path, optional
        Required if `add_iaf_covariate` is True.
    iaf_distance_column : str, optional
        Required if `add_iaf_covariate` is True.

    Returns
    -------
    pd.DataFrame
        Copy of `df`, with `age_months_c` and/or `iaf_distance_c` added.
    """
    out = df.copy()
    if add_age_covariate:
        out["age_months_c"] = out["age_months"] - out["age_months"].mean()
    if add_iaf_covariate:
        iaf = pd.read_csv(iaf_metrics_csv)
        dyad_iaf = iaf.drop_duplicates("dyad_id").set_index("dyad_id")[iaf_distance_column]
        out["iaf_distance_c"] = out["dyad_id"].map(dyad_iaf) - dyad_iaf.mean()
    return out


def build_formula(dv_col, extra_terms, extra_grouping=None):
    """Stage 6 model formula: `film * group` (sum contrasts) + dyad random intercept (D1).

    Parameters
    ----------
    dv_col : str
        Dependent variable column name.
    extra_terms : list of str
        D4 covariate column names (e.g. `["age_months_c"]`) to add as fixed effects.
    extra_grouping : str or None, optional
        Extra `(1|term)` grouping factor to append (used for the D2 pooled
        model's `(1|edge)`).

    Returns
    -------
    str
        A formula string bambi understands.
    """
    extra = "".join(f" + {t}" for t in extra_terms)
    grouping = " + (1|dyad_id)" + (f" + (1|{extra_grouping})" if extra_grouping else "")
    return f"{dv_col} ~ C(film, Sum) * C(group, Sum){extra}{grouping}"


def edge_sd_hyperprior(sd_prior_edge):
    """The (1|edge) SD hyperprior, selected by `sd_prior_edge` (D-conv-B).

    Scopes ONLY the `1|edge` intercept SD (L-conv-2) -- `1|dyad_id` and every
    M1/M2 varying-slope SD keep their own `HalfStudentT(3,1)` regardless of
    this toggle. Loud on an unrecognized value rather than a silent default.

    Parameters
    ----------
    sd_prior_edge : {"halfstudentt", "halfnormal"}

    Returns
    -------
    bambi.priors.Prior
    """
    if sd_prior_edge == "halfstudentt":
        return bmb.Prior("HalfStudentT", nu=3, sigma=1)
    if sd_prior_edge == "halfnormal":
        return bmb.Prior("HalfNormal", sigma=1)
    raise ValueError(f"unknown sd_prior_edge={sd_prior_edge!r} (expected 'halfstudentt' or 'halfnormal')")


def build_priors(sd_prior_edge):
    """D3 weakly-informative priors on the standardized (or ~unit, for `z_vs_surrogate`) scale.

    Unused keys (e.g. `1|edge` when the formula has no such term) are
    silently ignored by bambi -- not a masked error, just how bambi resolves
    a prior dict against a formula.

    Parameters
    ----------
    sd_prior_edge : {"halfstudentt", "halfnormal"}
        Passed to `edge_sd_hyperprior`.

    Returns
    -------
    dict
        Bambi `Prior` objects keyed by term name.
    """
    group_sd_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))
    return {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "C(film, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "C(group, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "C(film, Sum):C(group, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "1|dyad_id": group_sd_prior,
        "1|edge": bmb.Prior("Normal", mu=0, sigma=edge_sd_hyperprior(sd_prior_edge)),
        "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1),
    }


def fit_model(formula, data, mcmc_config, family="t"):
    """Fit one bambi model at the Stage 6 L9 MCMC configuration.

    Parameters
    ----------
    formula : str
        Model formula (see `build_formula`).
    data : pd.DataFrame
        Rows to fit on.
    mcmc_config : dict
        `{"draws", "tune", "chains", "target_accept", "seed", "cores"}`.
    family : str, optional
        Bambi family name (default `"t"`, L1).

    Returns
    -------
    bambi.Model, arviz.InferenceData
        The fitted model and its posterior (with `log_likelihood` for LOO).
    """
    model = bmb.Model(formula, data, family=family, priors=build_priors(mcmc_config["sd_prior_edge"]))
    idata = model.fit(
        draws=mcmc_config["draws"], tune=mcmc_config["tune"], chains=mcmc_config["chains"],
        target_accept=mcmc_config["target_accept"], random_seed=mcmc_config["seed"],
        progressbar=False, idata_kwargs={"log_likelihood": True}, cores=mcmc_config["cores"],
    )
    return model, idata


def fit_model_with_priors(formula, data, priors, required_group_terms, mcmc_config, family="t"):
    """Fit one bambi model at the Stage 6 L9 MCMC config with an explicit priors dict.

    Mirrors `fit_model`, but takes `priors` directly instead of
    `build_priors()` -- the D6 varying-slope models need extra
    group-specific prior keys `build_priors()` does not define. Deliberate
    small duplication of `fit_model`'s body (explicit over silently patching
    a shared function). Asserts that `required_group_terms` are exactly
    among the group-specific term names bambi assigned: a mismatch would
    mean a prior silently fell back to bambi's default, which must be caught
    loudly rather than fit anyway.

    Parameters
    ----------
    formula : str
    data : pd.DataFrame
    priors : dict
        Bambi `Prior` objects keyed by term name.
    required_group_terms : iterable of str
        Group-specific term names that must appear in the built model (e.g.
        `["C(film, Sum):C(group, Sum)|edge"]`).
    mcmc_config : dict
        `{"draws", "tune", "chains", "target_accept", "seed", "cores"}`.
    family : str, optional
        Bambi family name (default `"t"`, L1).

    Returns
    -------
    bambi.Model, arviz.InferenceData
    """
    model = bmb.Model(formula, data, family=family, priors=priors)
    term_names = set(model.distributional_components["mu"].terms.keys())
    missing = set(required_group_terms) - term_names
    assert not missing, f"prior keys not found among model terms {sorted(term_names)}: {missing}"
    idata = model.fit(
        draws=mcmc_config["draws"], tune=mcmc_config["tune"], chains=mcmc_config["chains"],
        target_accept=mcmc_config["target_accept"], random_seed=mcmc_config["seed"],
        progressbar=False, idata_kwargs={"log_likelihood": True}, cores=mcmc_config["cores"],
    )
    return model, idata


def common_terms_of(model):
    """Names of `model`'s fixed (common, non-group-specific) terms, excluding `Intercept`.

    Verified against bambi 0.17.2: common-vs-group-specific is distinguished
    by class identity (`bambi.terms.common.CommonTerm` vs
    `bambi.terms.group_specific.GroupSpecificTerm`), not a boolean attribute.

    Parameters
    ----------
    model : bambi.Model
        A built (not necessarily fit) model, e.g. from a probe fit.

    Returns
    -------
    list of str
    """
    return [
        name for name, term in model.distributional_components["mu"].terms.items()
        if not isinstance(term, GroupSpecificTerm) and name != "Intercept"
    ]


def reference_grid(films, groups, extra_terms):
    """The 6 film x group cells (group rows in `groups` order) used for every contrast.

    Any covariate in `extra_terms` is held at its centred reference (0, since
    covariates are mean-centred) so contrasts read at that reference value.

    Parameters
    ----------
    films : list of str
    groups : list of str
    extra_terms : list of str
        D4 covariate column names to add at 0.

    Returns
    -------
    pd.DataFrame
        6 rows: `film` (categorical, ordered as `films`), `group`
        (categorical, ordered as `groups`), plus any covariate columns at 0.
    """
    rows = [{"film": f, "group": g} for g in groups for f in films]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=films)
    grid["group"] = pd.Categorical(grid["group"], categories=groups)
    for term in extra_terms:
        grid[term] = 0.0
    return grid


def predict_cell_means(model, idata, grid):
    """Posterior draws of the population-level mean at each `reference_grid` row.

    Uses `include_group_specific=False` so the six cells are marginal
    (population-level) means, not tied to any specific dyad -- this is what
    makes the film/group contrasts below well-defined regardless of the sum
    contrast coding used to fit the model.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    grid : pd.DataFrame
        From `reference_grid()`.

    Returns
    -------
    xarray.DataArray
        Dims `(chain, draw, __obs__)`, `__obs__` indexing `grid`'s rows.
    """
    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=False)
    return preds.posterior["mu"]


def cell_indices(grid, film=None, group=None):
    """Row positions of `grid` matching the given `film`/`group` (either may be None = any).

    Parameters
    ----------
    grid : pd.DataFrame
        From `reference_grid()`.
    film, group : str or None
        Value to match, or None to match all.

    Returns
    -------
    list of int
        Positional indices into `grid`.
    """
    mask = pd.Series(True, index=grid.index)
    if film is not None:
        mask &= grid["film"] == film
    if group is not None:
        mask &= grid["group"] == group
    return list(grid.index[mask])


def summarize_draws(draws, hdi_prob):
    """Posterior mean, HDI, and directional probabilities for a draws array (L4).

    Parameters
    ----------
    draws : xarray.DataArray or np.ndarray
        Posterior draws of one scalar contrast.
    hdi_prob : float
        HDI mass (e.g. 0.95).

    Returns
    -------
    dict
        `estimate`, `hdi_low`, `hdi_high`, `p_gt0`, `p_lt0`.
    """
    flat = np.asarray(draws).flatten()
    hdi = az.hdi(flat, hdi_prob=hdi_prob)
    return {
        "estimate": float(flat.mean()),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
        "p_gt0": float((flat > 0).mean()),
        "p_lt0": float((flat < 0).mean()),
    }


def compute_contrasts(model, idata, grid, hdi_prob):
    """The standard L3 contrast set read off one model's posterior (estimated marginal means).

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    grid : pd.DataFrame
        From `reference_grid()`.
    hdi_prob : float
        Passed to `summarize_draws`.

    Returns
    -------
    dict of dict
        Keys `group_effect`, `film_contrast_overall`, `film_contrast_TD`,
        `film_contrast_ASD`, `interaction_film_group`, `grand_mean` -- each a
        `summarize_draws` dict of signed posterior draws.
    """
    mu = predict_cell_means(model, idata, grid)

    def cell(film=None, group=None):
        return mu.isel(__obs__=cell_indices(grid, film, group)).mean("__obs__")

    def film_contrast(group=None):
        return cell("Incredibles", group) - (cell("Peppa", group) + cell("Brave", group)) / 2

    group_effect = (cell(group="ASD") - cell(group="TD")).values.flatten()
    film_overall = film_contrast(group=None).values.flatten()
    film_td = film_contrast(group="TD").values.flatten()
    film_asd = film_contrast(group="ASD").values.flatten()
    interaction = film_asd - film_td
    grand_mean = mu.isel(__obs__=list(grid.index)).mean("__obs__").values.flatten()

    return {
        "group_effect": summarize_draws(group_effect, hdi_prob),
        "film_contrast_overall": summarize_draws(film_overall, hdi_prob),
        "film_contrast_TD": summarize_draws(film_td, hdi_prob),
        "film_contrast_ASD": summarize_draws(film_asd, hdi_prob),
        "interaction_film_group": summarize_draws(interaction, hdi_prob),
        "grand_mean": summarize_draws(grand_mean, hdi_prob),
    }


def back_transform(summary, sd):
    """Rescale a standardized contrast summary to raw Delta-units (D3).

    Valid for pure differences (all contrasts in `compute_contrasts` except
    `grand_mean`, which is an absolute level and needs `+ mean` too --
    handled by the caller, not here).

    Parameters
    ----------
    summary : dict
        A `summarize_draws` output.
    sd : float
        The edge's raw-scale standard deviation used to standardize.

    Returns
    -------
    dict
        Same keys, `estimate`/`hdi_low`/`hdi_high` scaled by `sd`;
        `p_gt0`/`p_lt0` unchanged (scaling by a positive number preserves sign).
    """
    return {
        "estimate": summary["estimate"] * sd,
        "hdi_low": summary["hdi_low"] * sd,
        "hdi_high": summary["hdi_high"] * sd,
        "p_gt0": summary["p_gt0"],
        "p_lt0": summary["p_lt0"],
    }


def convergence_row(model_label, idata, n_rows, n_dropped_unstable, rhat_max, ess_min, pareto_k_max):
    """One diagnostics-table row: Rhat/ESS/divergences/LOO for one fitted model (L9).

    Parameters
    ----------
    model_label : str
    idata : arviz.InferenceData
    n_rows : int
        Rows the model was fit on (after the L8 `real_stable` filter).
    n_dropped_unstable : int
        Rows dropped by the L8 filter.
    rhat_max, ess_min, pareto_k_max : float
        L9 pass-criteria thresholds.

    Returns
    -------
    dict
        Row for `stage06_diagnostics.csv`, including a `pass_l9` boolean.
    """
    summary = az.summary(idata)
    n_divergent = int(idata.sample_stats["diverging"].values.sum())
    loo = az.loo(idata, pointwise=True)
    max_rhat = float(summary["r_hat"].max())
    min_bulk_ess = float(summary["ess_bulk"].min())
    min_tail_ess = float(summary["ess_tail"].min())
    max_pareto_k = float(np.max(loo.pareto_k.values))
    pass_l9 = (max_rhat < rhat_max) and (min_bulk_ess > ess_min) and (min_tail_ess > ess_min) \
        and (n_divergent == 0) and (max_pareto_k < pareto_k_max)
    return {
        "model": model_label,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "n_divergent": n_divergent,
        "loo_elpd": float(loo.elpd_loo),
        "loo_se": float(loo.se),
        "max_pareto_k": max_pareto_k,
        "n_rows": n_rows,
        "n_dropped_unstable": n_dropped_unstable,
        "pass_l9": bool(pass_l9),
    }


def plot_forest(rows, title):
    """Horizontal forest plot (posterior mean + HDI) for a list of named contrasts.

    Parameters
    ----------
    rows : list of dict
        Each with `label`, `estimate`, `hdi_low`, `hdi_high`.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 1.2))
    ys = np.arange(len(rows))
    estimates = [r["estimate"] for r in rows]
    lo_err = [r["estimate"] - r["hdi_low"] for r in rows]
    hi_err = [r["hdi_high"] - r["estimate"] for r in rows]
    axis.errorbar(estimates, ys, xerr=[lo_err, hi_err], fmt="o", color="black", capsize=3)
    axis.axvline(0, color="red", linestyle="--", linewidth=1)
    axis.set_yticks(ys)
    axis.set_yticklabels([r["label"] for r in rows])
    axis.invert_yaxis()
    axis.set_xlabel("estimate (HDI)")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_ppc_figure(model, idata, title):
    """Posterior-predictive density overlay (L9 `pp_check`).

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    pps = model.predict(idata, kind="response", inplace=False)
    az.plot_ppc(pps, num_pp_samples=100)
    figure = plt.gcf()
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_pareto_k_figure(loo_result, title, pareto_k_max):
    """Pareto-k diagnostic scatter with the warning line (L9).

    Parameters
    ----------
    loo_result : arviz.stats.ELPDData
        From `az.loo(idata, pointwise=True)`.
    title : str
    pareto_k_max : float
        Warning-line threshold (e.g. 0.7).

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=(5, 3.5))
    pareto_k = loo_result.pareto_k.values
    axis.scatter(np.arange(len(pareto_k)), pareto_k, s=12)
    axis.axhline(pareto_k_max, color="red", linestyle="--")
    axis.set_xlabel("observation index")
    axis.set_ylabel("pareto k")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_edge_funnel(idata, tag, qc_dir):
    """Save a plot_pair of each (edge offset, edge SD) pair with divergences flagged.

    Diagnoses the (1|edge)/(...|edge) funnel: divergences bunched at small SD
    are the classic neck, confirming the geometry is the cause. Loud if the
    model carries no edge SD variable (a wiring error by the caller).

    Parameters
    ----------
    idata : arviz.InferenceData
        A fitted D6 model's posterior (must include `sample_stats.diverging`).
    tag : str
        Slug for the output filename (e.g. "pooled_D2", "M1").
    qc_dir : pathlib.Path
        Directory the figures are saved into.

    Returns
    -------
    list of Path
        Saved figure paths, one per edge `*_sigma` variable found.
    """
    sigma_vars = [v for v in idata.posterior.data_vars if v.endswith("_sigma") and "edge" in v]
    assert sigma_vars, f"[{tag}] no edge '*_sigma' variable in posterior -- model has no (...|edge) term?"
    saved_paths = []
    for sigma_var in sigma_vars:
        offset_var = sigma_var[: -len("_sigma")]
        if offset_var not in idata.posterior.data_vars:
            continue  # a sigma with no matching offset vector (rare); skip this one, keep the others
        ax = az.plot_pair(
            idata, var_names=[sigma_var, offset_var], divergences=True,
            marginals=False, kind="scatter", scatter_kwargs={"alpha": 0.15},
        )
        fig = ax.ravel()[0].figure if hasattr(ax, "ravel") else ax.figure
        fig.suptitle(f"{tag}: {sigma_var} funnel (divergences in orange)", fontsize=9)
        fig.tight_layout()
        safe = sigma_var.replace("|", "_").replace(":", "-").replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "")
        out_path = qc_dir / f"funnel_{tag}_{safe}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


def per_edge_contrasts(model, idata, edges, grid_categories, pooled_data, films, groups, extra_terms, hdi_prob):
    """Per-edge film/interaction contrasts from an edge-varying model (standardized units, signed).

    Predicts the 6 film x group cell means for every edge at one shared,
    in-sample `dyad_id` (so the dyad random intercept is identical within
    each edge and cancels exactly in every within-edge contrast) with
    `include_group_specific=True` (so the per-edge group-specific deviations
    enter), then reads `Incredibles - (Peppa+Brave)/2` within each group and
    their difference, per edge -- the same linear combinations as
    `compute_contrasts`, edge-scoped. Model-agnostic: reused unchanged for
    both M1 and M2.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    edges : list of str
        Edge strings to score (the 6 emphasis edges).
    grid_categories : list of str
        `edge` categories exactly as used at fit time (category order
        matters for bambi's internal indexing).
    pooled_data : pd.DataFrame
        The data the model was fit on (its first `dyad_id` is used as the
        shared reference dyad).
    films, groups : list of str
    extra_terms : list of str
        D4 covariate column names to hold at 0.
    hdi_prob : float
        Passed to `summarize_draws`.

    Returns
    -------
    dict of dict
        `edge -> {"interaction_film_group", "film_contrast_TD",
        "film_contrast_ASD"} -> summarize_draws() dict`.
    """
    ref_dyad = pooled_data["dyad_id"].iloc[0]
    rows = [{"film": f, "group": g, "edge": e, "dyad_id": ref_dyad}
            for e in edges for g in groups for f in films]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=films)
    grid["group"] = pd.Categorical(grid["group"], categories=groups)
    grid["edge"] = pd.Categorical(grid["edge"], categories=grid_categories)
    grid["dyad_id"] = grid["dyad_id"].astype(str)
    for term in extra_terms:  # D4 covariates held at centred reference
        grid[term] = 0.0

    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=True)
    mu = preds.posterior["mu"]  # dims (chain, draw, __obs__)

    def cell(edge, film=None, group=None):
        mask = grid["edge"] == edge
        if film is not None:
            mask &= grid["film"] == film
        if group is not None:
            mask &= grid["group"] == group
        return mu.isel(__obs__=list(grid.index[mask])).mean("__obs__")

    def film_contrast(edge, group):
        return cell(edge, "Incredibles", group) - (cell(edge, "Peppa", group) + cell(edge, "Brave", group)) / 2

    out = {}
    for e in edges:
        td = film_contrast(e, "TD").values.flatten()
        asd = film_contrast(e, "ASD").values.flatten()
        out[e] = {
            "film_contrast_TD": summarize_draws(td, hdi_prob),
            "film_contrast_ASD": summarize_draws(asd, hdi_prob),
            "interaction_film_group": summarize_draws(asd - td, hdi_prob),  # signed, no abs()
        }
    return out


def population_interaction_marginal_over_edge(model, idata, edges, grid_categories, pooled_data,
                                               films, groups, extra_terms, hdi_prob):
    """Film x group interaction marginalized (equal-weight average) over a FIXED edge factor.

    Needed for an edge-as-fixed-factor model only: `edge` there is a common
    (fixed), Sum-coded term, not a `(...|edge)` group-specific one, so
    `reference_grid()` (no `edge` column at all) cannot be predicted from
    that model's formula and `include_group_specific=False` has nothing to
    switch off. Equal-weight averaging over Sum-coded levels exactly cancels
    the edge main effect and every edge interaction term (that is what Sum
    contrasts are for), so this reproduces the same population quantity
    `compute_contrasts` reads off directly for edge-varying models -- just
    via explicit marginalization instead of dropping a group-specific term.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    edges : list of str
        The edge categories to average over (the 6 emphasis edges).
    grid_categories : list of str
        `edge` categories exactly as used at fit time.
    pooled_data : pd.DataFrame
        The data the model was fit on (its first `dyad_id` is used as the
        shared reference dyad).
    films, groups : list of str
    extra_terms : list of str
        D4 covariate column names to hold at 0.
    hdi_prob : float
        Passed to `summarize_draws`.

    Returns
    -------
    dict
        `summarize_draws()` output for the marginal `interaction_film_group`.
    """
    ref_dyad = pooled_data["dyad_id"].iloc[0]
    rows = [{"film": f, "group": g, "edge": e, "dyad_id": ref_dyad}
            for g in groups for f in films for e in edges]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=films)
    grid["group"] = pd.Categorical(grid["group"], categories=groups)
    grid["edge"] = pd.Categorical(grid["edge"], categories=grid_categories)
    grid["dyad_id"] = grid["dyad_id"].astype(str)
    for term in extra_terms:
        grid[term] = 0.0

    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=True)
    mu = preds.posterior["mu"]

    def cell(film=None, group=None):
        mask = pd.Series(True, index=grid.index)
        if film is not None:
            mask &= grid["film"] == film
        if group is not None:
            mask &= grid["group"] == group
        return mu.isel(__obs__=list(grid.index[mask])).mean("__obs__")  # equal-weight average, incl. over edges

    def film_contrast(group):
        return cell(film="Incredibles", group=group) - (cell(film="Peppa", group=group) + cell(film="Brave", group=group)) / 2

    interaction = (film_contrast("ASD") - film_contrast("TD")).values.flatten()
    return summarize_draws(interaction, hdi_prob)


def compute_localization_rows(model_tag, model, idata, emphasis_edges, edge_categories, edge_class_lookup,
                               pooled_data, films, groups, extra_terms, hdi_prob):
    """Per-edge + population interaction/film rows for one D6 model (returned, not appended).

    Parameters
    ----------
    model_tag : str
    model : bambi.Model
    idata : arviz.InferenceData
    emphasis_edges : list of str
        The 6 emphasis edge strings.
    edge_categories : list of str
        `edge` categories exactly as used at fit time.
    edge_class_lookup : dict
        `{edge: edge_class}` for the emphasis edges.
    pooled_data : pd.DataFrame
    films, groups : list of str
    extra_terms : list of str
    hdi_prob : float

    Returns
    -------
    list of dict
        Rows for `stage06_localization.csv`: one per (edge, contrast) plus a
        final "population" row.
    """
    rows = []
    per_edge = per_edge_contrasts(model, idata, emphasis_edges, edge_categories, pooled_data, films, groups, extra_terms, hdi_prob)
    for edge, contrasts in per_edge.items():
        for contrast_name, summary in contrasts.items():
            rows.append({
                "model": model_tag, "edge": edge, "edge_class": edge_class_lookup[edge],
                "contrast": contrast_name, "unit": "std", **summary,
            })
    edge_is_fixed_term = "C(edge, Sum)" in model.distributional_components["mu"].terms
    if edge_is_fixed_term:
        population = population_interaction_marginal_over_edge(
            model, idata, emphasis_edges, edge_categories, pooled_data, films, groups, extra_terms, hdi_prob)
    else:
        population = compute_contrasts(model, idata, reference_grid(films, groups, extra_terms), hdi_prob)["interaction_film_group"]
    rows.append({
        "model": model_tag, "edge": "population", "edge_class": "population",
        "contrast": "interaction_film_group", "unit": "std", **population,
    })
    return rows
