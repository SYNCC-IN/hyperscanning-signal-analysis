"""Pairing tables T of the real-data pipeline: loading and the array layouts the permutation tests expect.

Stage 5 of `scripts/pipeline_exploratory_mlutimodal_dDTF/` writes one NetCDF per film x estimator with
``T[edge, child_dyad, cg_dyad]`` (diagonal = real dyads, NaN = unstable fit, per-dyad `group`, `age_months`,
`real_stable`). The tests of `src.sim_inference` / `src.perm_anova` take ``real (n, columns)`` and
``pair_values (n, n, columns)`` (NaN diagonal); with several films the columns are film-major (column
``f * n_edges + e``), the layout of `src.perm_anova.rm_contrasts`.
"""

import numpy as np
import xarray as xr


def load_film_table(nc_path, edges):
    """Stable real dyads of one film and their pairing table, restricted to `edges`.

    A real dyad whose own fit is unstable is dropped with its row and column (`src.sim_inference.run_study`).

    Parameters
    ----------
    nc_path : pathlib.Path or str
        Stage-5 pairing table of one film and estimator.
    edges : list of str
        Edge names (``"source->target"``), in the column order wanted.

    Returns
    -------
    dict
        ``ids`` (list of str), ``group`` (np.ndarray of str), ``age`` (np.ndarray), ``real`` (n, n_edges),
        ``pair_values`` (n, n, n_edges; NaN diagonal), ``n_dyads`` (before dropping), ``unstable_real`` (list of
        dropped ids), ``n_pairs_unstable`` (unstable pseudo-dyads among the kept dyads), ``attrs``.
    """
    ds = xr.load_dataset(nc_path)
    table = ds["T"].sel(edge=edges).transpose("child_dyad", "cg_dyad", "edge").values
    stable = ds["real_stable"].values.astype(bool)
    keep = np.flatnonzero(stable)
    pair_values = table[np.ix_(keep, keep)].copy()
    real = pair_values[np.arange(len(keep)), np.arange(len(keep))].copy()
    pair_values[np.arange(len(keep)), np.arange(len(keep))] = np.nan
    off = ~np.eye(len(keep), dtype=bool)
    ids = [str(d) for d in ds["dyad"].values]
    return {"ids": [ids[i] for i in keep], "group": ds["group"].values[keep].astype(str), "age": ds["age_months"].values[keep],
            "real": real, "pair_values": pair_values, "n_dyads": len(ids),
            "unstable_real": [ids[i] for i in np.flatnonzero(~stable)],
            "n_pairs_unstable": int((~np.isfinite(pair_values[:, :, 0]) & off).sum()), "attrs": dict(ds.attrs)}


def stack_films(film_tables, films):
    """Film tables on the union of their dyads, films stacked as film-major columns.

    A dyad absent from a film (missing, or dropped as unstable) has NaN real values in that film's columns and NaN
    rows and columns in its pair values, so it drops out of every statistic of that film (NaN-tolerant means and
    medians of `src.sim_inference`).

    Parameters
    ----------
    film_tables : dict of {str: dict}
        `load_film_table` output per film.
    films : list of str
        Film order of the columns.

    Returns
    -------
    dict
        ``ids`` (sorted union), ``group`` (np.ndarray of str), ``age``, ``real`` (N, F * E),
        ``pair_values`` (N, N, F * E), ``present`` (N, F) bool.
    """
    ids = sorted(set().union(*(film_tables[f]["ids"] for f in films)))
    n, n_edges = len(ids), film_tables[films[0]]["real"].shape[1]
    real = np.full((n, len(films) * n_edges), np.nan)
    pair_values = np.full((n, n, len(films) * n_edges), np.nan)
    present = np.zeros((n, len(films)), dtype=bool)
    group, age = np.empty(n, dtype=object), np.full(n, np.nan)
    for f, film in enumerate(films):
        table = film_tables[film]
        rows = np.array([ids.index(d) for d in table["ids"]])
        cols = slice(f * n_edges, (f + 1) * n_edges)
        real[rows, cols] = table["real"]
        pair_values[np.ix_(rows, rows, np.arange(f * n_edges, (f + 1) * n_edges))] = table["pair_values"]
        present[rows, f] = True
        for r, g, a in zip(rows, table["group"], table["age"]):
            assert group[r] is None or group[r] == g, f"group of {ids[r]} differs between films"
            group[r], age[r] = g, a
    return {"ids": ids, "group": group.astype(str), "age": age, "real": real, "pair_values": pair_values,
            "present": present}


def apply_film_contrasts(Y, weights, n_edges):
    """Film contrasts of film-major columns: ``sum_f w_f * Y[..., film f, edge e]`` for every contrast and edge.

    Films with weight 0 are left out of the sum, so a NaN there does not spread into the contrast. With no zero
    weight this equals ``Y @ contrast_matrix(weights, n_edges)``.

    Parameters
    ----------
    Y : np.ndarray, shape (..., n_films * n_edges)
    weights : np.ndarray, shape (n_contrasts, n_films)
    n_edges : int

    Returns
    -------
    np.ndarray, shape (..., n_contrasts * n_edges)
        Contrast-major columns (column ``c * n_edges + e``).
    """
    wide = Y.reshape(Y.shape[:-1] + (weights.shape[1], n_edges))
    out = [sum(w * wide[..., f, :] for f, w in enumerate(row) if w != 0) for row in weights]
    return np.concatenate(out, axis=-1)


def contrast_matrix(weights, n_edges):
    """Linear map from film-major columns to contrast-major columns: ``kron(weights.T, I_edges)``.

    Parameters
    ----------
    weights : np.ndarray, shape (n_contrasts, n_films)
    n_edges : int

    Returns
    -------
    np.ndarray, shape (n_films * n_edges, n_contrasts * n_edges)
        The `combine` argument of `src.sim_inference.group_permutation_maxT`.
    """
    return np.kron(np.asarray(weights, dtype=float).T, np.eye(n_edges))
