"""Surrogate-null inference and study-level readouts for the simulation studies.

One *study* is one simulated set of dyads that saw the same stimulus. For a study we
(i) estimate the real connectivity of every dyad, (ii) estimate it for every foreign
(child from dyad b, caregiver from dyad c) surrogate pair, (iii) turn the real values
into signed delta and robust z against the surrogate null, and (iv) read out dyad-level
and study-level decisions. All functions take their settings as arguments.

Tests (none assumes a symmetric or normal null):
- dyad level: empirical one-sided p-value of the real value among its surrogate values;
- study level: pairing permutation test. Under H0 (the two persons of a dyad are independent
  given the film) the real pairing of children and caregivers is exchangeable with any
  re-pairing, so the mean real value is compared with the means of randomly re-paired studies;
- group contrast: permutation of the group labels, difference of group means of delta.
"""

import numpy as np
from scipy.stats import median_abs_deviation

try:
    from .sim_estimation import edge_matrix
    from .surrogate import surrogate_pairs
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.sim_estimation import edge_matrix
    from src.surrogate import surrogate_pairs


def surrogate_design(designs, child_dyad, cg_dyad, nodes, child_nodes):
    """Design of a surrogate dyad: child-role nodes from one dyad, caregiver-role nodes from another.

    Parameters
    ----------
    designs : np.ndarray, shape (n_dyads, k_total, n_samples)
        z-scored designs of all dyads of one stimulus.
    child_dyad, cg_dyad : int
        Dyads supplying the child-role and the caregiver-role rows.
    nodes : sequence of int
        Rows analysed (in this order), e.g. all measured nodes.
    child_nodes : sequence of int
        Row indices that belong to the child role.

    Returns
    -------
    np.ndarray, shape (len(nodes), n_samples)
    """
    return np.stack([designs[child_dyad][n] if n in child_nodes else designs[cg_dyad][n] for n in nodes])


def real_matrices(designs, nodes, est):
    """Band-averaged connectivity matrix of every real dyad.

    Parameters
    ----------
    designs : np.ndarray, shape (n_dyads, k_total, n_samples)
    nodes : sequence of int
        Rows analysed.
    est : dict
        Keyword arguments of `src.sim_estimation.edge_matrix` (except `design`).

    Returns
    -------
    matrices : np.ndarray, shape (n_dyads, k, k)
    max_roots : np.ndarray, shape (n_dyads,)
    p_used : np.ndarray, shape (n_dyads,)
    """
    results = [edge_matrix(d[list(nodes)], **est) for d in designs]
    return (np.stack([r[0] for r in results]), np.array([r[1] for r in results]),
            np.array([r[2] for r in results]))


def surrogate_values(designs, nodes, child_nodes, est, stability_max_root, rng, max_pairs=None):
    """Connectivity of every foreign pairing, arranged as a (child dyad, caregiver dyad) table.

    Unstable fits (max companion root >= `stability_max_root`) are left as NaN, as in the
    pipeline's null construction.

    Parameters
    ----------
    designs : np.ndarray, shape (n_dyads, k_total, n_samples)
    nodes, child_nodes : sequence of int
    est : dict
        Keyword arguments of `edge_matrix` (except `design`).
    stability_max_root : float
    rng : numpy.random.Generator
        Used only when `max_pairs` subsamples the pairs.
    max_pairs : int, optional
        Cap on the number of surrogate pairs (random subsample); None uses all N(N-1).

    Returns
    -------
    values : np.ndarray, shape (n_dyads, n_dyads, k, k)
        ``values[b, c]`` = matrix of the surrogate with child rows from dyad b, caregiver rows from c.
    n_attempted, n_unstable : int
    """
    n_dyads = designs.shape[0]
    pairs = surrogate_pairs(list(range(n_dyads)))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    values = np.full((n_dyads, n_dyads, len(nodes), len(nodes)), np.nan)
    n_unstable = 0
    for child_dyad, cg_dyad in pairs:
        matrix, root, _ = edge_matrix(surrogate_design(designs, child_dyad, cg_dyad, nodes, child_nodes), **est)
        if root < stability_max_root:
            values[child_dyad, cg_dyad] = matrix
        else:
            n_unstable += 1
    return values, len(pairs), n_unstable


def null_mask(pair_values, dyad, mode, groups=None):
    """Which surrogate pairs form the null of one real dyad.

    Parameters
    ----------
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k)
        From `surrogate_values`.
    dyad : int
        The real dyad being scored.
    mode : {'pooled', 'exclude', 'within_group'}
        'pooled': every valid foreign pair (the pipeline's film-level null).
        'exclude': pairs that involve neither member of `dyad` (leave-one-dyad-out null).
        'within_group': pairs formed only from dyads of the same group as `dyad`.
    groups : np.ndarray, shape (n_dyads,), optional
        Group label per dyad (needed for 'within_group').

    Returns
    -------
    np.ndarray of bool, shape (n_dyads, n_dyads)
    """
    n_dyads = pair_values.shape[0]
    mask = np.isfinite(pair_values[:, :, 0, 0])
    index = np.arange(n_dyads)
    if mode == "exclude":
        mask = mask & (index[:, None] != dyad) & (index[None, :] != dyad)
    elif mode == "within_group":
        mask = mask & (groups[:, None] == groups[dyad]) & (groups[None, :] == groups[dyad])
    return mask


def null_stats(pair_values, dyad, mode, groups=None):
    """Median and scaled MAD of the surrogate null that applies to one real dyad.

    Parameters
    ----------
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k)
        From `surrogate_values`.
    dyad : int
    mode : {'pooled', 'exclude', 'within_group'}
        See `null_mask`.
    groups : np.ndarray, shape (n_dyads,), optional

    Returns
    -------
    median, mad : np.ndarray, shape (k, k)
        Median and median absolute deviation scaled by 1.4826 (`scale='normal'`).
    n_pairs : int
        Number of surrogate pairs used.
    """
    mask = null_mask(pair_values, dyad, mode, groups)
    values = pair_values[mask]
    return np.median(values, axis=0), median_abs_deviation(values, axis=0, scale="normal"), int(mask.sum())


def delta_and_z_all(real_matrices_, pair_values, mode, groups=None):
    """Signed delta and robust z of every dyad's every edge against its surrogate null.

    ``delta = real - median(null)``, ``z = delta / (1.4826 MAD(null))`` (same maths as
    `src.surrogate.delta_and_z`, vectorized over edges).

    Parameters
    ----------
    real_matrices_ : np.ndarray, shape (n_dyads, k, k)
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k)
    mode : {'pooled', 'exclude', 'within_group'}
    groups : np.ndarray, optional

    Returns
    -------
    delta, z : np.ndarray, shape (n_dyads, k, k)
    """
    delta = np.empty_like(real_matrices_)
    z = np.empty_like(real_matrices_)
    for dyad in range(real_matrices_.shape[0]):
        median, mad, _ = null_stats(pair_values, dyad, mode, groups)
        delta[dyad] = real_matrices_[dyad] - median
        with np.errstate(invalid="ignore", divide="ignore"):      # 0 / 0 on the diagonal
            z[dyad] = delta[dyad] / mad
    return delta, z


def dyad_pvalues(real_matrices_, pair_values, mode, groups=None):
    """Empirical one-sided p-value of every dyad's every edge against its surrogate null.

    ``p = (1 + #{null >= real}) / (1 + n_null)``, with the null of `null_mask`.

    Parameters
    ----------
    real_matrices_ : np.ndarray, shape (n_dyads, k, k)
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k)
    mode : {'pooled', 'exclude', 'within_group'}
    groups : np.ndarray, optional

    Returns
    -------
    np.ndarray, shape (n_dyads, k, k)
    """
    p = np.empty_like(real_matrices_)
    for dyad in range(real_matrices_.shape[0]):
        values = pair_values[null_mask(pair_values, dyad, mode, groups)]
        p[dyad] = (1 + np.sum(values >= real_matrices_[dyad], axis=0)) / (1 + values.shape[0])
    return p


def pairing_permutation_pvalue(table, rng, n_perm, groups=None):
    """One-sided pairing permutation p-value for 'the real pairing has a larger mean than re-pairings'.

    Parameters
    ----------
    table : np.ndarray, shape (n_dyads, n_dyads)
        Value of one edge for child of dyad b paired with caregiver of dyad c; the diagonal holds
        the real dyads. NaN entries (unstable surrogate fits) are left out of the means.
    rng : numpy.random.Generator
    n_perm : int
        Number of random re-pairings (caregivers permuted across children).
    groups : np.ndarray, shape (n_dyads,), optional
        If given, caregivers are permuted only within groups (the within-group null).

    Returns
    -------
    float
    """
    n_dyads = table.shape[0]
    observed = np.mean(np.diag(table))
    perms = np.tile(np.arange(n_dyads), (n_perm, 1))
    for label in (np.unique(groups) if groups is not None else [None]):
        members = np.arange(n_dyads) if label is None else np.flatnonzero(groups == label)
        perms[:, members] = rng.permuted(perms[:, members], axis=1)
    permuted = np.nanmean(table[np.arange(n_dyads), perms], axis=1)
    return float((1 + np.sum(permuted >= observed)) / (1 + n_perm))


def permutation_pvalue(values_a, values_b, rng, n_perm):
    """Two-sided permutation p-value for a difference in means between two groups.

    Parameters
    ----------
    values_a, values_b : np.ndarray
    rng : numpy.random.Generator
    n_perm : int

    Returns
    -------
    float
    """
    observed = abs(np.mean(values_a) - np.mean(values_b))
    pooled = np.concatenate([values_a, values_b])
    in_a = rng.permuted(np.tile(np.arange(pooled.size) < values_a.size, (n_perm, 1)), axis=1)
    diffs = np.abs((in_a * pooled).sum(1) / values_a.size - (~in_a * pooled).sum(1) / values_b.size)
    return float((1 + np.sum(diffs >= observed)) / (1 + n_perm))


def edge_summary(table, delta, z, p_dyad, edge, groups, within_group, dyad_alpha, alpha, rng, n_perm):
    """Dyad-level, study-level and group-level readouts for one directed edge.

    Parameters
    ----------
    table : np.ndarray, shape (n_dyads, n_dyads, k, k)
        Pairing table of `score_study` (diagonal = real dyads).
    delta, z, p_dyad : np.ndarray, shape (n_dyads, k, k)
        From `delta_and_z_all` and `dyad_pvalues` (``[dyad, target, source]``).
    edge : tuple of int
        ``(source, target)`` positions in the analysed node order.
    groups : np.ndarray or None
        Group label (0/1) per dyad; None skips the group contrast.
    within_group : bool
        Re-pair only within groups in the study-level test.
    dyad_alpha : float
        Dyad-level significance level (one-sided).
    alpha : float
        Study-level and group-level significance level.
    rng : numpy.random.Generator
    n_perm : int
        Number of permutations of the study-level and group-level tests.

    Returns
    -------
    dict
        dyad_rate (share of dyads with p < dyad_alpha), study_p / study_reject (pairing permutation
        test of a larger mean in real dyads), group_p / group_reject (permutation test of a group
        difference in mean delta), dir_dyad (share of dyads with delta_forward > delta_reverse),
        dir_study (1 if the median forward-minus-reverse delta is positive), median_z, mean_z, sd_z,
        median_delta.
    """
    source, target = edge
    d_fwd, d_rev = delta[:, target, source], delta[:, source, target]
    z_edge = z[:, target, source]
    ok = np.isfinite(z_edge)
    study_p = pairing_permutation_pvalue(table[:, :, target, source], rng, n_perm, groups if within_group else None)
    group_p = np.nan
    if groups is not None:
        group_p = permutation_pvalue(d_fwd[ok & (groups == 0)], d_fwd[ok & (groups == 1)], rng, n_perm)
    return {
        "dyad_rate": float(np.mean(p_dyad[ok, target, source] < dyad_alpha)),
        "study_p": study_p, "study_reject": bool(study_p < alpha),
        "group_p": group_p, "group_reject": bool(group_p < alpha) if groups is not None else np.nan,
        "dir_dyad": float(np.mean(d_fwd[ok] > d_rev[ok])),
        "dir_study": float(np.median(d_fwd[ok] - d_rev[ok]) > 0),
        "median_z": float(np.median(z_edge[ok])), "mean_z": float(np.mean(z_edge[ok])),
        "sd_z": float(np.std(z_edge[ok])), "median_delta": float(np.median(d_fwd[ok])),
    }


def score_study(real, pair_values, diagonal, mode, groups, edges, dyad_alpha, alpha, rng, n_perm):
    """Readouts of every edge of one study against one surrogate null.

    Parameters
    ----------
    real : np.ndarray, shape (n_dyads, k, k)
        Real matrices of the (stable) dyads.
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k)
        Surrogate values of the same dyads (from `surrogate_values`).
    diagonal : np.ndarray, shape (n_dyads, k, k)
        Values placed on the diagonal of the pairing table: `real` for a same-film null, or the real
        dyads of the film the surrogates come from (other-film null).
    mode : {'pooled', 'exclude', 'within_group'}
    groups : np.ndarray or None
    edges : dict of {str: (int, int)}
        ``name -> (source, target)``.
    dyad_alpha, alpha : float
    rng : numpy.random.Generator
    n_perm : int

    Returns
    -------
    dict of {str: dict}
        `edge_summary` per edge name.
    """
    delta, z = delta_and_z_all(real, pair_values, mode, groups)
    p_dyad = dyad_pvalues(real, pair_values, mode, groups)
    table = pair_values.copy()
    index = np.arange(real.shape[0])
    table[index, index] = diagonal
    return {name: edge_summary(table, delta, z, p_dyad, edge, groups, mode == "within_group", dyad_alpha, alpha, rng, n_perm)
            for name, edge in edges.items()}


def run_study(designs, nodes, child_nodes, est, edges, modes, groups, rng, stability_max_root,
              dyad_alpha, alpha, n_perm, max_pairs=None):
    """Analyse one simulated study end to end and return one record per (mode, edge).

    Dyads whose real fit is unstable are dropped together with their surrogate pairs.

    Parameters
    ----------
    designs : np.ndarray, shape (n_dyads, k_total, n_samples)
        z-scored designs.
    nodes : sequence of int
        Rows analysed; edge positions refer to this order.
    child_nodes : sequence of int
        Row indices of the child role.
    est : dict
        Keyword arguments of `edge_matrix` (except `design`).
    edges : dict of {str: (int, int)}
        ``name -> (source, target)`` positions in the analysed node order.
    modes : sequence of str
        Null scopes to evaluate ('pooled', 'exclude', 'within_group').
    groups : np.ndarray or None
    rng : numpy.random.Generator
    stability_max_root : float
    dyad_alpha, alpha : float
        Dyad-level and study/group-level significance levels.
    n_perm : int
        Permutations of the study-level and group-level tests.
    max_pairs : int, optional

    Returns
    -------
    list of dict
        Each record has `mode`, `edge`, the `edge_summary` fields, `real_mean` (mean real value),
        `null_median_mean` (mean null median) and bookkeeping (`n_real_unstable`,
        `n_pairs_attempted`, `n_pairs_unstable`).
    """
    real, roots, _ = real_matrices(designs, nodes, est)
    values, n_attempted, n_unstable = surrogate_values(designs, nodes, child_nodes, est, stability_max_root, rng, max_pairs)
    keep = roots < stability_max_root
    real, values = real[keep], values[keep][:, keep]
    kept_groups = None if groups is None else groups[keep]
    records = []
    for mode in modes:
        delta, _ = delta_and_z_all(real, values, mode, kept_groups)
        summaries = score_study(real, values, real, mode, kept_groups, edges, dyad_alpha, alpha, rng, n_perm)
        for name, (source, target) in edges.items():
            records.append(dict(summaries[name], mode=mode, edge=name,
                                real_mean=float(real[:, target, source].mean()),
                                null_median_mean=float(np.mean(real[:, target, source] - delta[:, target, source])),
                                n_real_unstable=int((~keep).sum()), n_pairs_attempted=n_attempted,
                                n_pairs_unstable=n_unstable))
    return records


def wilson_interval(successes, n, z=1.96):
    """Wilson score confidence interval of a proportion.

    Parameters
    ----------
    successes, n : int
    z : float, optional
        Normal quantile (1.96 = 95%).

    Returns
    -------
    low, high : float
    """
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)
