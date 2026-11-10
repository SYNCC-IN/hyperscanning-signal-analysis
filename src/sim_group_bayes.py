"""Edge-wise group, film and interaction effects from Bambi models, for the simulation studies (S13).

Thin layer over `src.group_model` (model fitting, priors, convergence gate, draw summaries): the
pipeline's `compute_contrasts`/`per_edge_contrasts` hard-code the film and group names of the real
study, so the cell-mean contrasts are written here for any film/group labels. Every setting (priors,
MCMC configuration, HDI mass) is passed in by the calling script.

Effects (estimated-marginal-means style, from the posterior of the film x group cell means):
- group effect: mean over films of (group[1] - group[0]);
- film contrast: films[0] - mean(films[1:]), averaged over groups;
- interaction: film contrast in group[1] minus film contrast in group[0].
"""

import numpy as np
import pandas as pd

try:
    from .group_model import summarize_draws
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.group_model import summarize_draws


def cell_grid(films, groups, edges=None, dyad_id=None):
    """Prediction grid of every film x group cell (optionally x edge, at one reference dyad).

    Parameters
    ----------
    films, groups : list of str
        Category orders as used at fit time.
    edges : list of str, optional
        Edge categories as used at fit time; adds an `edge` column (one block of cells per edge).
    dyad_id : str, optional
        In-sample dyad to predict at (its random intercept is the same in every cell and cancels in
        every contrast); required by models with ``(...|edge)`` terms predicted with group-specific effects.

    Returns
    -------
    pandas.DataFrame
    """
    rows = [{"film": f, "group": g, **({"edge": e} if edges is not None else {})}
            for e in (edges if edges is not None else [None]) for g in groups for f in films]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=films)
    grid["group"] = pd.Categorical(grid["group"], categories=groups)
    if edges is not None:
        grid["edge"] = pd.Categorical(grid["edge"], categories=edges)
    if dyad_id is not None:
        grid["dyad_id"] = dyad_id
    return grid


def effect_draws(mu, grid, films, groups):
    """Posterior draws of the group effect, film contrast and interaction from cell-mean draws.

    Parameters
    ----------
    mu : np.ndarray, shape (n_draws, n_cells)
        Cell-mean draws, columns in the order of `grid`.
    grid : pandas.DataFrame
        Rows = the film x group cells of one edge (or of the population).
    films, groups : list of str
        ``films[0]`` is the contrasted film; ``groups[1] - groups[0]`` is the group effect.

    Returns
    -------
    dict of {str: np.ndarray}
        'group', 'film', 'interaction' draws.
    """
    def cell(film=None, group=None):
        mask = np.ones(len(grid), bool)
        if film is not None:
            mask &= (grid["film"] == film).to_numpy()
        if group is not None:
            mask &= (grid["group"] == group).to_numpy()
        return mu[:, mask].mean(axis=1)

    def film_contrast(group=None):
        return cell(films[0], group) - np.mean([cell(f, group) for f in films[1:]], axis=0)

    return {"group": cell(group=groups[1]) - cell(group=groups[0]),
            "film": film_contrast(),
            "interaction": film_contrast(groups[1]) - film_contrast(groups[0])}


def predict_mu(model, idata, grid, include_group_specific):
    """Posterior draws of the mean at every grid row, flattened over chains.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    grid : pandas.DataFrame
    include_group_specific : bool

    Returns
    -------
    np.ndarray, shape (n_draws, len(grid))
    """
    preds = model.predict(idata, data=grid, kind="response_params", inplace=False,
                          include_group_specific=include_group_specific)
    mu = preds.posterior["mu"].values
    return mu.reshape(-1, mu.shape[-1])


def population_effects(model, idata, films, groups, hdi_prob):
    """Population-level (common) group effect, film contrast and interaction of a fitted model.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    films, groups : list of str
    hdi_prob : float

    Returns
    -------
    dict of {str: dict}
        Effect name -> `summarize_draws` output.
    """
    grid = cell_grid(films, groups)
    draws = effect_draws(predict_mu(model, idata, grid, False), grid, films, groups)
    return {name: summarize_draws(d, hdi_prob) for name, d in draws.items()}


def edge_effects(model, idata, edges, films, groups, dyad_id, hdi_prob):
    """Edge-specific group effect, film contrast and interaction of a model with an edge grouping factor.

    Cells are predicted with the group-specific terms at one in-sample reference dyad, so the
    edge-specific deviations enter and the dyad intercept cancels in every contrast. With only
    ``(1|edge)`` the edge intercept also cancels, so every edge gets the common effects.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    edges : list of str
        Edge categories as used at fit time.
    films, groups : list of str
    dyad_id : str
        Reference dyad (in the fitted data).
    hdi_prob : float

    Returns
    -------
    dict of {str: dict of {str: dict}}
        edge -> effect name -> `summarize_draws` output.
    """
    grid = cell_grid(films, groups, edges, dyad_id)
    mu = predict_mu(model, idata, grid, True)
    out = {}
    for edge in edges:
        rows = (grid["edge"] == edge).to_numpy()
        draws = effect_draws(mu[:, rows], grid[rows].reset_index(drop=True), films, groups)
        out[edge] = {name: summarize_draws(d, hdi_prob) for name, d in draws.items()}
    return out


def excludes_zero(summary):
    """True if the HDI of a `summarize_draws` output excludes 0."""
    return bool(summary["hdi_low"] > 0 or summary["hdi_high"] < 0)
