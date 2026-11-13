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
  With the within-group null, `renull=True` recomputes every dyad's null median under each relabelling
  (`renull_delta`), so the estimation error of the group null medians is permuted with the labels.

Multiplicity across edges: `pairing_permutation_maxT` and `group_permutation_maxT` score all edges
of a family on one shared set of permutations and return max-T (Westfall & Young, 1993), min-p and
Bonferroni adjusted p-values (`family_pvalues`); `score_study`/`run_study` add them on request.

Exact enumeration: when a test has no more distinct rearrangements than `n_perm` (group relabellings
C(n, n_1), within-group re-pairings, sign vectors 2^n), all of them are used instead of random draws
(`label_permutations`, `caregiver_permutations`, `sign_flips`). The observed arrangement is left out of
the enumerated set, so the usual ``(1 + #{T_b >= T_obs}) / (1 + B)`` with B = (number of arrangements - 1)
equals the exact p-value ``#{arrangements with T >= T_obs} / (number of arrangements)``. The random draws
are still made (and discarded) in that case, so the generator's stream, and every later test drawn from
it, is unchanged.
"""

import itertools
import math

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


def cross_film_values(designs, other_designs, nodes, child_nodes, est, stability_max_root):
    """Connectivity of mismatched-film pairings: child rows from one film's dyads, caregiver rows from another film's.

    Laid out like `surrogate_values` (diagonal left as NaN), so it can replace the same-film table in
    `score_study` to test what happens when the null comes from the wrong stimulus.

    Parameters
    ----------
    designs : np.ndarray, shape (n_dyads, k_total, n_samples)
        Dyads of the scored film (child rows are taken from these).
    other_designs : np.ndarray, shape (n_dyads, k_total, n_samples)
        Dyads of another film (caregiver rows are taken from these).
    nodes, child_nodes : sequence of int
    est : dict
        Keyword arguments of `edge_matrix` (except `design`).
    stability_max_root : float

    Returns
    -------
    values : np.ndarray, shape (n_dyads, n_dyads, k, k)
        ``values[b, c]`` = matrix of child b (scored film) with caregiver c (other film), b != c; unstable fits NaN.
    """
    n_dyads = designs.shape[0]
    both = np.concatenate([designs, other_designs])
    values = np.full((n_dyads, n_dyads, len(nodes), len(nodes)), np.nan)
    for child_dyad, cg_dyad in surrogate_pairs(list(range(n_dyads))):
        matrix, root, _ = edge_matrix(surrogate_design(both, child_dyad, n_dyads + cg_dyad, nodes, child_nodes), **est)
        if root < stability_max_root:
            values[child_dyad, cg_dyad] = matrix
    return values


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
    perms = caregiver_permutations(n_dyads, rng, n_perm, groups)
    permuted = np.nanmean(table[np.arange(n_dyads), perms], axis=1)
    return float((1 + np.sum(permuted >= observed)) / (1 + len(perms)))


def caregiver_permutations(n_dyads, rng, n_perm, groups=None):
    """Re-pairings of caregivers with children, optionally only within groups.

    Random re-pairings, or all of them except the identity when there are at most `n_perm`
    (exact enumeration, see the module docstring).

    Parameters
    ----------
    n_dyads : int
    rng : numpy.random.Generator
    n_perm : int
    groups : np.ndarray, shape (n_dyads,), optional
        If given, caregivers are permuted only among dyads of the same group.

    Returns
    -------
    np.ndarray of int, shape (n_perm, n_dyads), or (n_arrangements - 1, n_dyads) when enumerated
        ``perms[b, i]`` = dyad whose caregiver is paired with child `i` in permutation `b`.
    """
    perms = np.tile(np.arange(n_dyads), (n_perm, 1))
    blocks = [np.arange(n_dyads)] if groups is None else [np.flatnonzero(groups == label) for label in np.unique(groups)]
    for members in blocks:
        perms[:, members] = rng.permuted(perms[:, members], axis=1)
    if math.prod(math.factorial(len(m)) for m in blocks) > n_perm:
        return perms
    exact = []
    for orders in itertools.product(*[itertools.permutations(m) for m in blocks]):
        row = np.arange(n_dyads)
        for members, order in zip(blocks, orders):
            row[members] = order
        exact.append(row)
    exact = np.array(exact)
    return exact[~(exact == np.arange(n_dyads)).all(axis=1)]


def label_permutations(in_one, rng, n_perm):
    """Relabellings of a two-group design: random ones, or all except the observed one when there are at most `n_perm`.

    Parameters
    ----------
    in_one : np.ndarray of bool, shape (n,)
        Observed membership of group 1.
    rng : numpy.random.Generator
    n_perm : int

    Returns
    -------
    np.ndarray of bool, shape (n_perm, n), or (C(n, n_1) - 1, n) when enumerated (see the module docstring)
    """
    random = rng.permuted(np.tile(in_one, (n_perm, 1)), axis=1)
    n, n_one = in_one.size, int(in_one.sum())
    if math.comb(n, n_one) > n_perm:
        return random
    exact = np.zeros((math.comb(n, n_one), n), dtype=bool)
    for b, members in enumerate(itertools.combinations(range(n), n_one)):
        exact[b, list(members)] = True
    return exact[~(exact == in_one).all(axis=1)]


def sign_flips(n, rng, n_perm):
    """Sign vectors for a sign-flip test: random ones, or all except all-plus when 2**n <= `n_perm`.

    Parameters
    ----------
    n : int
        Number of units (dyads).
    rng : numpy.random.Generator
    n_perm : int

    Returns
    -------
    np.ndarray of float (+1/-1), shape (n_perm, n), or (2**n - 1, n) when enumerated (see the module docstring)
    """
    random = rng.choice([-1.0, 1.0], size=(n_perm, n))
    if 2 ** n > n_perm:
        return random
    exact = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    return exact[1:]


def maxT_adjust(observed, null, stepdown=True):
    """Family-wise adjusted p-values from a permutation null shared by all tests (max-T).

    Each test is standardized by its own permutation distribution,
    ``u = (T - mean_perm) / sd_perm``; the adjusted p-value of test e is the share of
    permutations whose maximum u over the family reaches ``u_obs[e]``:
    ``p = (1 + #{max_e u_b,e >= u_obs}) / (1 + n_perm)`` (Westfall & Young, 1993). With
    `stepdown`, the maximum runs only over the tests not yet rejected (tests ordered by
    decreasing ``u_obs``), and the p-values are made monotone in that order.

    Parameters
    ----------
    observed : np.ndarray, shape (n_tests,)
        Observed statistics; larger is more extreme.
    null : np.ndarray, shape (n_perm, n_tests)
        Statistics of the same permutations for every test.
    stepdown : bool, optional

    Returns
    -------
    np.ndarray, shape (n_tests,)
    """
    mean, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    u_obs, u_null = (observed - mean) / sd, (null - mean) / sd
    order = np.argsort(-u_obs)
    if stepdown:
        maxima = np.maximum.accumulate(u_null[:, order][:, ::-1], axis=1)[:, ::-1]   # max over tests j.. of the order
    else:
        maxima = np.repeat(u_null.max(axis=1, keepdims=True), observed.size, axis=1)
    p_sorted = (1 + np.sum(maxima >= u_obs[order], axis=0)) / (1 + null.shape[0])
    p = np.empty_like(p_sorted)
    p[order] = np.maximum.accumulate(p_sorted)
    return p


def minP_adjust(observed, null, stepdown=True):
    """Family-wise adjusted p-values from a shared permutation null (min-p, rank based).

    Every permutation statistic is turned into a per-test p-value by its rank among the
    observed and permuted values of that test, ``p_b,e = #{T >= T_b,e} / (1 + n_perm)``; the
    adjusted p-value of test e is ``(1 + #{min_e p_b,e <= p_obs[e]}) / (1 + n_perm)``. No
    assumption on the shape of the null (unlike the standardization of `maxT_adjust`).
    `stepdown` works as in `maxT_adjust`, with tests ordered by increasing ``p_obs``.

    Parameters
    ----------
    observed : np.ndarray, shape (n_tests,)
        Observed statistics; larger is more extreme.
    null : np.ndarray, shape (n_perm, n_tests)
    stepdown : bool, optional

    Returns
    -------
    np.ndarray, shape (n_tests,)
    """
    n_perm, n_tests = null.shape
    reference = np.sort(np.vstack([observed, null]), axis=0)
    rank_p = lambda values, e: (n_perm + 1 - np.searchsorted(reference[:, e], values, side="left")) / (n_perm + 1)
    p_obs = np.array([rank_p(observed[e], e) for e in range(n_tests)])
    p_null = np.stack([rank_p(null[:, e], e) for e in range(n_tests)], axis=1)
    order = np.argsort(p_obs, kind="stable")
    if stepdown:
        minima = np.minimum.accumulate(p_null[:, order][:, ::-1], axis=1)[:, ::-1]
    else:
        minima = np.repeat(p_null.min(axis=1, keepdims=True), n_tests, axis=1)
    p_sorted = (1 + np.sum(minima <= p_obs[order], axis=0)) / (1 + n_perm)
    p = np.empty_like(p_sorted)
    p[order] = np.maximum.accumulate(p_sorted)
    return p


def family_pvalues(observed, null, stepdown=True):
    """Unadjusted and family-wise adjusted p-values of tests that share one permutation null.

    Parameters
    ----------
    observed : np.ndarray, shape (n_tests,)
        Observed statistics; larger is more extreme.
    null : np.ndarray, shape (n_perm, n_tests)
    stepdown : bool, optional
        Step-down variant for 'maxT' and 'minP' (the single-step values are always returned too).

    Returns
    -------
    dict of {str: np.ndarray}
        'unadjusted' (``(1 + #{T_b >= T_obs}) / (1 + n_perm)`` per test), 'bonferroni'
        (unadjusted times the family size, capped at 1), 'maxT', 'maxT_single_step', 'minP',
        'minP_single_step'.
    """
    unadjusted = (1 + np.sum(null >= observed, axis=0)) / (1 + null.shape[0])
    return {
        "unadjusted": unadjusted,
        "bonferroni": np.minimum(1.0, unadjusted * observed.size),
        "maxT": maxT_adjust(observed, null, stepdown),
        "maxT_single_step": maxT_adjust(observed, null, False),
        "minP": minP_adjust(observed, null, stepdown),
        "minP_single_step": minP_adjust(observed, null, False),
    }


def pairing_permutation_maxT(tables, rng, n_perm, groups=None, stepdown=True):
    """Study-level pairing permutation test of several edges on the same re-pairings, with FWER control.

    One set of caregiver re-pairings is drawn and applied to every edge, so the joint
    null of the family is preserved; the statistic of an edge is the mean of its paired
    values (NaN pairs, i.e. unstable surrogate fits, left out of the mean).

    Parameters
    ----------
    tables : np.ndarray, shape (n_dyads, n_dyads, n_edges)
        ``tables[b, c, e]`` = value of edge e for child b with caregiver c; the diagonal holds the
        real dyads.
    rng : numpy.random.Generator
    n_perm : int
    groups : np.ndarray, shape (n_dyads,), optional
        Re-pair only within groups (within-group null).
    stepdown : bool, optional

    Returns
    -------
    dict of {str: np.ndarray}
        `family_pvalues` output (one value per edge).
    """
    n_dyads = tables.shape[0]
    index = np.arange(n_dyads)
    observed = np.nanmean(tables[index, index], axis=0)
    perms = caregiver_permutations(n_dyads, rng, n_perm, groups)
    null = np.nanmean(tables[index[None, :], perms], axis=1)
    return family_pvalues(observed, null, stepdown)


def group_null_medians(pair_values, labels):
    """Within-group null median of every column, for every labelling of the dyads.

    The null of a group is the set of surrogate pairs (child of dyad b, caregiver of dyad c) with b and c both in
    that group, as in `null_mask` with ``mode='within_group'``; NaN entries (the diagonal, unstable surrogate fits)
    are left out per column.

    Parameters
    ----------
    pair_values : np.ndarray, shape (n_dyads, n_dyads, n_columns)
        Surrogate values (``[child dyad, caregiver dyad, column]``; a column is an edge, or an edge of one film).
    labels : np.ndarray of bool, shape (n_labellings, n_dyads)
        Membership of group 1 in each labelling (every row has the same number of members).

    Returns
    -------
    np.ndarray, shape (n_labellings, 2, n_columns)
        ``[labelling, group (0/1), column]``.
    """
    n_labellings, n_dyads = labels.shape
    medians = []
    for in_group in (~labels, labels):
        members = np.nonzero(in_group)[1].reshape(n_labellings, -1)
        block = pair_values[members[:, :, None], members[:, None, :]]          # (labellings, m, m, columns)
        medians.append(np.nanmedian(block.reshape(n_labellings, -1, pair_values.shape[2]), axis=1))
    return np.stack(medians, axis=1)


def renull_delta(real, pair_values, labels):
    """Delta of every dyad against its within-group null, recomputed for every labelling.

    ``delta[b, d] = real[d] - median of the null of the group of d in labelling b`` (`group_null_medians`). For the
    observed labelling this equals `delta_and_z_all` with ``mode='within_group'``.

    Parameters
    ----------
    real : np.ndarray, shape (n_dyads, n_columns)
    pair_values : np.ndarray, shape (n_dyads, n_dyads, n_columns)
    labels : np.ndarray of bool, shape (n_labellings, n_dyads)

    Returns
    -------
    np.ndarray, shape (n_labellings, n_dyads, n_columns)
    """
    medians = group_null_medians(pair_values, labels)
    return real[None] - np.take_along_axis(medians, labels.astype(int)[:, :, None], axis=1)


def group_mean_difference(values, in_one):
    """Mean of group 1 minus mean of group 0 per column, NaN values left out.

    Parameters
    ----------
    values : np.ndarray, shape (n_dyads, n_columns) or (n_labellings, n_dyads, n_columns)
    in_one : np.ndarray of bool, shape (n_labellings, n_dyads)

    Returns
    -------
    np.ndarray, shape (n_labellings, n_columns)
    """
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    one = in_one[:, :, None].astype(float)
    return (one * filled).sum(1) / (one * valid).sum(1) - ((1 - one) * filled).sum(1) / ((1 - one) * valid).sum(1)


def renull_statistics(real, pair_values, valid, labels, combine=None):
    """Signed group differences of the re-nulled delta, one row per labelling (the `renull` statistic).

    Parameters
    ----------
    real : np.ndarray, shape (n_dyads, n_columns)
    pair_values : np.ndarray, shape (n_dyads, n_dyads, n_columns)
    valid : np.ndarray of bool, shape (n_dyads, n_tests)
        Dyad-test values that enter the statistic (the others are left out, as NaN in the fixed-delta test).
    labels : np.ndarray of bool, shape (n_labellings, n_dyads)
    combine : np.ndarray, shape (n_columns, n_tests), optional
        Linear map from the re-nulled delta columns to the tested quantities (e.g. a film contrast); default identity.

    Returns
    -------
    np.ndarray, shape (n_labellings, n_tests)
    """
    delta = renull_delta(real, pair_values, labels)
    if combine is not None:
        delta = delta @ combine
    return group_mean_difference(np.where(valid[None], delta, np.nan), labels)


def group_permutation_maxT(delta, groups, rng, n_perm, stepdown=True, renull=False, real=None, pair_values=None, combine=None):
    """Two-sided group contrast of several edges on the same label permutations, with FWER control.

    The statistic of an edge is ``|mean_1 - mean_0|`` of its delta (NaN values left out).

    Parameters
    ----------
    delta : np.ndarray, shape (n_dyads, n_edges)
    groups : np.ndarray of int, shape (n_dyads,)
        Labels 0/1.
    rng : numpy.random.Generator
    n_perm : int
    stepdown : bool, optional
    renull : bool, optional
        False (default): the labels are permuted on the fixed `delta`. True: for every permutation the within-group
        null median of every dyad is recomputed from `pair_values` with the permuted groups and delta is recomputed
        before the statistic (`renull_statistics`); `delta` then only marks, by its NaN entries, the dyads left out.
        The same permutations are drawn in both cases.
    real : np.ndarray, shape (n_dyads, n_columns), optional
        Real values (needed with `renull`).
    pair_values : np.ndarray, shape (n_dyads, n_dyads, n_columns), optional
        Surrogate values of all dyads of both groups, NaN on the diagonal (needed with `renull`).
    combine : np.ndarray, shape (n_columns, n_edges), optional
        With `renull`: tested quantities as linear combinations of the delta columns (e.g. a film contrast over
        columns stacked as film x edge); default identity.

    Returns
    -------
    dict of {str: np.ndarray}
        `family_pvalues` output (one value per edge).
    """
    labels = label_permutations(groups == 1, rng, n_perm)
    if renull:
        observed_labels = (groups == 1)[None]
        valid = np.isfinite(delta)
        observed = np.abs(renull_statistics(real, pair_values, valid, observed_labels, combine))[0]
        return family_pvalues(observed, np.abs(renull_statistics(real, pair_values, valid, labels, combine)), stepdown)
    valid = np.isfinite(delta)
    values = np.where(valid, delta, 0.0)

    def statistic(in_one):
        in_one = in_one.astype(float)
        mean_one = (in_one @ values) / (in_one @ valid)
        mean_zero = ((1 - in_one) @ values) / ((1 - in_one) @ valid)
        return np.abs(mean_one - mean_zero)

    return family_pvalues(statistic(groups[None, :] == 1)[0], statistic(labels), stepdown)



def signflip_maxT(contrast, rng, n_perm, groups=None, stepdown=True):
    """Two-sided sign-flip test of a within-dyad contrast on several edges, on shared sign vectors, with FWER control.

    Under H0 (no film effect) each dyad's contrast is symmetric about 0, so its sign is exchangeable. One matrix of
    signs (one sign per dyad, ``n_perm x n_dyads``) is drawn and applied to every edge; with fewer than `n_perm`
    non-trivial sign vectors all of them are used (exact test, see `sign_flips`). The statistic of an edge is
    ``|mean c|`` over its non-NaN dyads, or with `groups` ``|(mean_0 c + mean_1 c) / 2|`` (the film effect averaged
    over groups, so unequal group sizes do not weight it).

    Parameters
    ----------
    contrast : np.ndarray, shape (n_dyads, n_edges)
        Contrast per dyad and edge (e.g. delta in film 1 minus the mean of films 2 and 3); NaN for dyads that miss a
        film, left out of that edge only.
    rng : numpy.random.Generator
    n_perm : int
    groups : np.ndarray of int, shape (n_dyads,), optional
        Labels 0/1.
    stepdown : bool, optional

    Returns
    -------
    dict of {str: np.ndarray}
        `family_pvalues` output (one value per edge).
    """
    valid = np.isfinite(contrast)
    values = np.where(valid, contrast, 0.0)
    blocks = [np.ones(len(contrast), bool)] if groups is None else [groups == 0, groups == 1]

    def statistic(signs):
        means = [((signs * block) @ values) / (block.astype(float) @ valid) for block in blocks]
        return np.abs(np.mean(means, axis=0))

    flips = sign_flips(len(contrast), rng, n_perm)
    return family_pvalues(statistic(np.ones((1, len(contrast))))[0], statistic(flips), stepdown)

MULTIPLICITY_METHODS = ("unadjusted", "bonferroni", "maxT", "maxT_single_step", "minP", "minP_single_step")


def multiplicity_records(table, delta, z, groups, within_group, family, alpha, rng, n_perm, methods,
                         renull=False, real=None, pair_values=None):
    """Family-wise adjusted study-level and group-level decisions for one study and one null.

    Parameters
    ----------
    table : np.ndarray, shape (n_dyads, n_dyads, k, k)
        Pairing table (diagonal = real dyads).
    delta, z : np.ndarray, shape (n_dyads, k, k)
        From `delta_and_z_all`; dyads with non-finite z are left out of the group contrast,
        as in `edge_summary`.
    groups : np.ndarray or None
        Group labels 0/1; None skips the group contrast.
    within_group : bool
        Re-pair only within groups in the study-level test.
    family : dict of {str: (int, int)}
        Edges of the family, ``name -> (source, target)``.
    alpha : float
    rng : numpy.random.Generator
    n_perm : int
    methods : sequence of str
        Subset of `MULTIPLICITY_METHODS`.
    renull : bool, optional
        Re-null the group contrast under every permutation (`group_permutation_maxT`); needs `real` and `pair_values`.
    real : np.ndarray, shape (n_dyads, k, k), optional
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k), optional
        Surrogate values (NaN diagonal).

    Returns
    -------
    dict of {str: dict}
        Per edge name: ``study_p_adj_<m>``, ``study_reject_adj_<m>``, ``group_p_adj_<m>``,
        ``group_reject_adj_<m>`` for every method m.
    """
    sources = [s for s, _ in family.values()]
    targets = [t for _, t in family.values()]
    study = pairing_permutation_maxT(table[:, :, targets, sources], rng, n_perm,
                                     groups if within_group else None)
    group = None
    if groups is not None:
        renull_args = {}
        if renull:
            renull_args = dict(renull=True, real=real[:, targets, sources], pair_values=pair_values[:, :, targets, sources])
        group = group_permutation_maxT(np.where(np.isfinite(z[:, targets, sources]), delta[:, targets, sources], np.nan),
                                       groups, rng, n_perm, **renull_args)
    out = {}
    for i, name in enumerate(family):
        row = {}
        for m in methods:
            row[f"study_p_adj_{m}"] = float(study[m][i])
            row[f"study_reject_adj_{m}"] = bool(study[m][i] < alpha)
            row[f"group_p_adj_{m}"] = float(group[m][i]) if group is not None else np.nan
            row[f"group_reject_adj_{m}"] = bool(group[m][i] < alpha) if group is not None else np.nan
        out[name] = row
    return out


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
    pooled = np.concatenate([values_a, values_b])
    observed_labels = np.arange(pooled.size) < values_a.size
    in_a = label_permutations(observed_labels, rng, n_perm)
    statistic = lambda labels: np.abs((labels * pooled).sum(-1) / values_a.size - (~labels * pooled).sum(-1) / values_b.size)
    return float((1 + np.sum(statistic(in_a) >= statistic(observed_labels))) / (1 + len(in_a)))


def edge_summary(table, delta, z, p_dyad, edge, groups, within_group, dyad_alpha, alpha, rng, n_perm,
                 renull=False, real=None, pair_values=None):
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
    renull : bool, optional
        Group contrast with the within-group null recomputed under every label permutation (see
        `group_permutation_maxT`) instead of `permutation_pvalue` on the fixed delta; needs `real` and `pair_values`.
    real : np.ndarray, shape (n_dyads, k, k), optional
    pair_values : np.ndarray, shape (n_dyads, n_dyads, k, k), optional
        Surrogate values (NaN diagonal).

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
    if groups is not None and renull:
        group_p = float(group_permutation_maxT(np.where(ok, d_fwd, np.nan)[:, None], groups, rng, n_perm, renull=True,
                                               real=real[:, target, source][:, None],
                                               pair_values=pair_values[:, :, target, source][:, :, None])["unadjusted"][0])
    elif groups is not None:
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


def family_dict(edges, family):
    """The multiplicity family as ``name -> (source, target)``.

    Parameters
    ----------
    edges : dict of {str: (int, int)}
    family : None, list of str or dict of {str: (int, int)}
        None = all `edges`; a list names edges of `edges`; a dict is returned as is.

    Returns
    -------
    dict of {str: (int, int)}
    """
    if family is None:
        return dict(edges)
    if isinstance(family, dict):
        return dict(family)
    return {name: edges[name] for name in family}


def score_study(real, pair_values, diagonal, mode, groups, edges, dyad_alpha, alpha, rng, n_perm,
                family=None, multiplicity="none", multiplicity_rng=None, renull=False):
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
    family : list of str or dict of {str: (int, int)}, optional
        The multiplicity family: names of `edges` (default: all `edges`), or ``name -> (source,
        target)``, which may contain edges outside `edges` (those get only the family readouts).
    multiplicity : str or sequence of str, optional
        'none' (default: per-edge tests only) or one or more of `MULTIPLICITY_METHODS` ('maxT' and
        'minP' are step-down). The family is tested on one shared set of permutations
        (`pairing_permutation_maxT`, `group_permutation_maxT`).
    multiplicity_rng : numpy.random.Generator, optional
        Generator of the family permutations (default `rng`, drawn after the per-edge tests). A
        separate generator leaves every draw from `rng`, and so the per-edge results, unchanged.
    renull : bool, optional
        Only with ``mode='within_group'``: the group contrasts (per edge and family-wise) recompute the within-group
        null of every dyad under every label permutation (see `group_permutation_maxT`). Default False.

    Returns
    -------
    dict of {str: dict}
        `edge_summary` per edge name. With multiplicity, each family edge also has
        ``study_p_adj_<m>``, ``study_reject_adj_<m>``, ``group_p_adj_<m>``, ``group_reject_adj_<m>``
        (NaN for edges outside the family), and every edge has ``any_reject_study_<m>`` and
        ``any_reject_group_<m>`` (at least one rejection in the family) for each method m and for
        'none' (the per-edge tests, uncorrected; NaN if the family has edges outside `edges`).
    """
    delta, z = delta_and_z_all(real, pair_values, mode, groups)
    p_dyad = dyad_pvalues(real, pair_values, mode, groups)
    table = pair_values.copy()
    index = np.arange(real.shape[0])
    table[index, index] = diagonal
    assert not renull or mode == "within_group", "renull applies to the within-group null only"
    renull_args = dict(renull=renull, real=real, pair_values=pair_values)
    out = {name: edge_summary(table, delta, z, p_dyad, edge, groups, mode == "within_group", dyad_alpha, alpha, rng, n_perm,
                              **renull_args)
           for name, edge in edges.items()}
    methods = [multiplicity] if isinstance(multiplicity, str) else list(multiplicity)
    if methods == ["none"]:
        return out
    family_edges = family_dict(edges, family)
    adjusted = multiplicity_records(table, delta, z, groups, mode == "within_group", family_edges, alpha,
                                    rng if multiplicity_rng is None else multiplicity_rng, n_perm, methods, **renull_args)
    flags = {"any_reject_study_none": np.nan, "any_reject_group_none": np.nan}
    if set(family_edges) <= set(edges):
        flags["any_reject_study_none"] = any(out[n]["study_reject"] for n in family_edges)
        if groups is not None:
            flags["any_reject_group_none"] = any(out[n]["group_reject"] for n in family_edges)
    for name in family_edges:
        out.setdefault(name, {})
    for m in methods:
        flags[f"any_reject_study_{m}"] = any(adjusted[n][f"study_reject_adj_{m}"] for n in family_edges)
        flags[f"any_reject_group_{m}"] = any(adjusted[n][f"group_reject_adj_{m}"] for n in family_edges) if groups is not None else np.nan
    empty = {key: np.nan for key in next(iter(adjusted.values()))}
    for name in out:
        out[name].update(adjusted.get(name, empty), **flags)
    return out


def run_study(designs, nodes, child_nodes, est, edges, modes, groups, rng, stability_max_root,
              dyad_alpha, alpha, n_perm, max_pairs=None, family=None, multiplicity="none", multiplicity_rng=None,
              renull=False):
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
    family : list of str, optional
        Multiplicity family (edge names; default all `edges`), see `score_study`.
    multiplicity : str or sequence of str, optional
        See `score_study` (default 'none').
    multiplicity_rng : numpy.random.Generator, optional
        See `score_study`.
    renull : bool, optional
        Re-null the group contrasts of the 'within_group' mode under every permutation (see `score_study`); the other
        modes are unaffected. Default False.

    Returns
    -------
    list of dict
        Each record has `mode`, `edge`, the `score_study` fields, `real_mean` (mean real value),
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
        summaries = score_study(real, values, real, mode, kept_groups, edges, dyad_alpha, alpha, rng, n_perm,
                                family, multiplicity, multiplicity_rng, renull and mode == "within_group")
        for name, (source, target) in (edges | family_dict(edges, family)).items():
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
        Bounds kept within [0, p] and [p, 1] (rounding can otherwise put them a hair outside, e.g. 1 - 1e-16 for p = 1).
    """
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(min(max(centre - half, 0.0), p)), float(max(min(centre + half, 1.0), p))
