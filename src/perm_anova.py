"""Factorial permutation ANOVA for dyad-level repeated-measures designs (Freedman-Lane, whole-dyad rearrangements).

Design. Each dyad i has a within-dyad cell vector y_i (e.g. 3 films x 8 edges = 24 cells). The cells are laid out in
the order of the `levels` dict, first factor slowest (C order): for ``levels = {"film": 3, "edge": 8}`` cell
``f * 8 + e`` holds film f, edge e (film-major). A within-dyad term W is an orthonormal contrast matrix C_W
(n_cells x k_W, `rm_contrasts`); the dyads' data for the term are ``Y_W = Y @ C_W`` (n_dyads x k_W). The between-dyad
mean (for the group main effect) uses the term ``()`` (C = 1 / sqrt(n_cells)). This is the summary-statistic
reduction of a repeated-measures design: rearranging whole rows of Y_W moves whole dyads, so the dependence between
the cells of a dyad is kept without modelling it.

Statistic (`f_statistic`). Y_W ~ [X Z] by OLS, with X the effect of interest and Z the nuisance columns:
``F = (tr SS_hyp / df_hyp) / (tr SS_err / df_err)``, pooled over the k_W columns (the sphericity-assumed RM-ANOVA F;
the common factor k_W of both degrees of freedom cancels). The permutation distribution makes it valid without
sphericity.

Freedman-Lane (`freedman_lane`). Fit the reduced model Y_W ~ Z, rearrange its residuals across dyads (permutation of
rows, or sign flip of rows), add back the fitted values, refit the full model and recompute F. Permutation for
between-dyad effects of interest (group, group x W); sign flip for within-dyad terms whose effect of interest is the
intercept of Y_W (W main effects), with the group (sum-to-zero coded) in Z so that the intercept is the unweighted
mean of the group means.

Rearrangements (`rearrangement_matrix`) come from `src.sim_inference` (`caregiver_permutations`, `sign_flips`):
random ones, or all of them except the observed one when there are no more than `n_perm` (exact enumeration). With
``p = (1 + #{F_b >= F_obs}) / (1 + B)`` (`src.sim_inference.family_pvalues`) the enumerated case gives the exact
p-value. Family-wise control over several tests scored on the same rearrangements (e.g. the 8 edges) uses
`src.sim_inference.maxT_adjust` (max-T / max-F, standardized, step-down).

Statistics (`term_statistic`): 'F' (above), 'ss_hyp' (tr SS_hyp only, not studentized by SS_err; for a 1-df group
main effect a monotone function of the squared difference of group means) and 'beta' (sum of squared OLS coefficients
of X; for one X column and one Y column a monotone function of |beta|).

`relabel_renull_test` is the raw-label (Manly-type) permutation for group terms whose dependent variable itself
depends on the grouping (delta against a within-group null recomputed for every relabelling).
"""

import numpy as np

try:
    from .sim_inference import caregiver_permutations, sign_flips
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.sim_inference import caregiver_permutations, sign_flips


def helmert_contrasts(k):
    """Orthonormal Helmert contrasts of a factor with `k` levels.

    Column j (j = 1 .. k-1) compares level j+1 with the mean of levels 1 .. j.

    Parameters
    ----------
    k : int
        Number of levels (>= 2).

    Returns
    -------
    np.ndarray, shape (k, k - 1)
        Orthonormal columns, each orthogonal to the constant vector.
    """
    assert k >= 2, "a factor needs at least 2 levels"
    out = np.zeros((k, k - 1))
    for j in range(1, k):
        out[:j, j - 1] = 1.0
        out[j, j - 1] = -j
        out[:, j - 1] /= np.linalg.norm(out[:, j - 1])
    return out


def rm_contrasts(levels, term, custom=None):
    """Orthonormal contrast matrix of a within-dyad term over the cell layout.

    The cells are ordered by the factors of `levels` in dict order, first factor slowest (C order; e.g. film-major
    for ``{"film": 3, "edge": 8}``). Factors in `term` contribute Helmert contrasts (or the matrix of `custom`),
    the others the normalized constant vector (average over that factor); the result is their Kronecker product.
    ``term=()`` gives the overall mean, 1 / sqrt(n_cells).

    Parameters
    ----------
    levels : dict of {str: int}
        Factor name -> number of levels, in cell-layout order.
    term : tuple of str
        Factors of the term (a subset of `levels`).
    custom : dict of {str: np.ndarray}, optional
        Factor -> (levels x m) orthonormal contrast used instead of the Helmert contrasts (e.g. an edge-class
        contrast from `edge_factor_contrasts`); every key must be in `term`.

    Returns
    -------
    np.ndarray, shape (n_cells, k_term)
    """
    custom = custom or {}
    assert set(term) <= set(levels), f"term {term} has factors outside {list(levels)}"
    assert set(custom) <= set(term), "custom contrasts must belong to factors of the term"
    out = np.ones((1, 1))
    for factor, k in levels.items():
        if factor in custom:
            block = np.asarray(custom[factor], dtype=float)
            assert block.shape[0] == k, f"custom contrast of {factor} must have {k} rows"
        elif factor in term:
            block = helmert_contrasts(k)
        else:
            block = np.ones((k, 1)) / np.sqrt(k)
        out = np.kron(out, block)
    assert np.allclose(out.T @ out, np.eye(out.shape[1])), "contrast matrix is not orthonormal"
    return out


def edge_factor_contrasts(edge_names, factor):
    """8 x 1 contrast of a binary edge class of the cross-person edges.

    Edge names are ``"<role>:<signal>-><role>:<signal>"`` with roles ``cg``/``child`` and two signals (e.g. ROI and
    HRV). The classes are `direction` (cg -> child = +1, child -> cg = -1), `source` and `target` (first signal in
    sorted order of the observed signals = +1, e.g. HRV = +1, ROI = -1). The edges must form a complete
    2 x 2 x 2 layout (direction x source signal x target signal).

    Parameters
    ----------
    edge_names : list of str
    factor : {'direction', 'source', 'target'}

    Returns
    -------
    np.ndarray, shape (n_edges, 1)
        Unit-norm column of +1 / -1 entries.
    """
    parsed = []
    for name in edge_names:
        source, target = name.split("->")
        (s_role, s_signal), (t_role, t_signal) = source.split(":"), target.split(":")
        assert {s_role, t_role} == {"cg", "child"}, f"{name} is not a cross-person edge"
        parsed.append((s_role, s_signal, t_signal))
    signals = sorted({p[1] for p in parsed} | {p[2] for p in parsed})
    assert len(signals) == 2, f"expected two signals, got {signals}"
    assert len(edge_names) == 8 and len(set(parsed)) == 8, "edges do not form a complete 2 x 2 x 2 layout"
    column = {"direction": [1.0 if p[0] == "cg" else -1.0 for p in parsed],
              "source": [1.0 if p[1] == signals[0] else -1.0 for p in parsed],
              "target": [1.0 if p[2] == signals[0] else -1.0 for p in parsed]}[factor]
    return np.asarray(column)[:, None] / np.sqrt(len(edge_names))


def residual_ss(Y, M):
    """Trace of the residual sum-of-squares matrix of the OLS fit Y ~ M (batched over leading axes).

    Parameters
    ----------
    Y : np.ndarray, shape (..., n, k)
    M : np.ndarray, shape (..., n, q)
        Design; q may be 0 (no columns: the residual is Y itself).

    Returns
    -------
    np.ndarray, shape (...)
    """
    fitted = M @ (np.linalg.pinv(M) @ Y)
    return ((Y - fitted) ** 2).sum(axis=(-2, -1))


def full_design(X, Z):
    """Full design [X Z] with X and Z broadcast to a common batch shape.

    Parameters
    ----------
    X : np.ndarray, shape (..., n, q_x)
    Z : np.ndarray, shape (..., n, q_z)

    Returns
    -------
    np.ndarray, shape (..., n, q_x + q_z)
    """
    assert X.shape[-2] == Z.shape[-2], "X and Z must have one row per dyad"
    batch = np.broadcast_shapes(X.shape[:-2], Z.shape[:-2])
    return np.concatenate([np.broadcast_to(X, batch + X.shape[-2:]), np.broadcast_to(Z, batch + Z.shape[-2:])], axis=-1)


def ss_components(Y, X, Z):
    """Trace hypothesis and error sums of squares of the columns X in Y ~ [X Z] (batched over leading axes).

    ``SS_hyp = SS_err(Y ~ Z) - SS_err(Y ~ [X Z])``, ``SS_err = SS_err(Y ~ [X Z])``, pooled over the columns of Y.

    Parameters
    ----------
    Y : np.ndarray, shape (..., n, k)
    X : np.ndarray, shape (..., n, q_x)
    Z : np.ndarray, shape (..., n, q_z)

    Returns
    -------
    ss_hyp, ss_err : np.ndarray, shape (...)
    """
    assert X.shape[-2] == Y.shape[-2], "X must have one row per dyad"
    ss_full = residual_ss(Y, full_design(X, Z))
    return residual_ss(Y, Z) - ss_full, ss_full


def f_statistic(Y, X, Z):
    """Pooled (trace) F of the columns X in the OLS model Y ~ [X Z] (batched over leading axes).

    ``F = ((SS_err(Y ~ Z) - SS_err(Y ~ [X Z])) / q_x) / (SS_err(Y ~ [X Z]) / (n - q_x - q_z))``, sums of squares
    pooled over the columns of Y (the sphericity-assumed RM-ANOVA F of the term).

    Parameters
    ----------
    Y : np.ndarray, shape (..., n, k)
    X : np.ndarray, shape (..., n, q_x)
        Effect of interest.
    Z : np.ndarray, shape (..., n, q_z)
        Nuisance columns (q_z may be 0).

    Returns
    -------
    np.ndarray, shape (...)
    """
    n, q_x, q_z = Y.shape[-2], X.shape[-1], Z.shape[-1]
    df_err = n - q_x - q_z
    assert df_err > 0, "no residual degrees of freedom"
    ss_hyp, ss_err = ss_components(Y, X, Z)
    return (ss_hyp / q_x) / (ss_err / df_err)


def coefficient_ss(Y, X, Z):
    """Sum of squared OLS coefficients of the columns X in Y ~ [X Z], over all columns of Y (batched).

    Parameters
    ----------
    Y : np.ndarray, shape (..., n, k)
    X : np.ndarray, shape (..., n, q_x)
    Z : np.ndarray, shape (..., n, q_z)

    Returns
    -------
    np.ndarray, shape (...)
    """
    beta = np.linalg.pinv(full_design(X, Z)) @ Y
    return (beta[..., :X.shape[-1], :] ** 2).sum(axis=(-2, -1))


def term_statistic(Y, X, Z, statistic):
    """Statistic of the columns X in Y ~ [X Z]: 'F' (`f_statistic`), 'ss_hyp' (SS_hyp of `ss_components`) or 'beta'
    (`coefficient_ss`); batched over leading axes.

    Parameters
    ----------
    Y : np.ndarray, shape (..., n, k)
    X : np.ndarray, shape (..., n, q_x)
    Z : np.ndarray, shape (..., n, q_z)
    statistic : {'F', 'ss_hyp', 'beta'}

    Returns
    -------
    np.ndarray, shape (...)
    """
    assert statistic in ("F", "ss_hyp", "beta"), f"unknown statistic {statistic}"
    if statistic == "F":
        return f_statistic(Y, X, Z)
    if statistic == "ss_hyp":
        return ss_components(Y, X, Z)[0]
    return coefficient_ss(Y, X, Z)


def rearrangement_matrix(n, n_perm, mode, rng, groups=None):
    """Whole-dyad rearrangements: row permutations or sign vectors, all of them when there are at most `n_perm`.

    Wraps `src.sim_inference.caregiver_permutations` (permute; exact when n! <= n_perm, or the product of the
    within-group factorials with `groups`) and `src.sim_inference.sign_flips` (signflip; exact when 2**n <= n_perm).
    The observed arrangement is left out of an enumerated set.

    Parameters
    ----------
    n : int
        Number of dyads.
    n_perm : int
    mode : {'permute', 'signflip'}
    rng : numpy.random.Generator
    groups : np.ndarray, shape (n,), optional
        Permute only within groups (permute mode only).

    Returns
    -------
    np.ndarray, shape (n_rearrangements, n)
        Integer row indices (permute; row b of the rearranged data is ``data[perms[b]]``) or +1/-1 signs (signflip).
    """
    assert mode in ("permute", "signflip"), f"unknown mode {mode}"
    if mode == "permute":
        return caregiver_permutations(n, rng, n_perm, groups)
    assert groups is None, "sign flips are not restricted to groups"
    return sign_flips(n, rng, n_perm)


def freedman_lane(Y, X, Z, rearrangements, mode, statistic="F"):
    """Freedman-Lane permutation (or sign-flip) distribution of a statistic of the columns X (default the pooled F).

    The reduced model Y ~ Z is fitted; its residuals are rearranged across dyads (rows permuted, or rows multiplied
    by a sign) and added back to its fitted values; the full model Y ~ [X Z] is refitted on every rearranged data set.

    Parameters
    ----------
    Y : np.ndarray, shape (n, k)
        Dyad-level data of one term (``Y_cells @ C_W``).
    X : np.ndarray, shape (n, q_x)
    Z : np.ndarray, shape (n, q_z)
    rearrangements : np.ndarray, shape (B, n)
        From `rearrangement_matrix` (integer indices for 'permute', +1/-1 for 'signflip').
    mode : {'permute', 'signflip'}
    statistic : {'F', 'ss_hyp', 'beta'}, optional
        See `term_statistic`.

    Returns
    -------
    F_obs : float
        Observed statistic.
    F_perm : np.ndarray, shape (B,)
    """
    n = Y.shape[0]
    assert Y.ndim == 2 and X.shape[0] == n and Z.shape[0] == n, "Y, X, Z must be (n, .) matrices"
    assert rearrangements.ndim == 2 and rearrangements.shape[1] == n, "rearrangements must be (B, n)"
    fitted = Z @ (np.linalg.pinv(Z) @ Y)
    resid = Y - fitted
    if mode == "permute":
        assert np.issubdtype(rearrangements.dtype, np.integer), "permute mode needs integer row indices"
        rearranged = resid[rearrangements]
    else:
        assert mode == "signflip" and set(np.unique(rearrangements)) <= {-1.0, 1.0}, "signflip mode needs +1/-1 signs"
        rearranged = rearrangements[:, :, None] * resid[None]
    return float(term_statistic(Y, X, Z, statistic)), term_statistic(fitted[None] + rearranged, X, Z, statistic)


def labels_from_permutations(in_one, perms):
    """Group labels implied by row permutations of the data with the labels kept in place.

    Permuting the data rows (``data[perm]``) against fixed labels is the same as giving dyad ``perm[i]`` the label of
    position i, i.e. labels ``in_one[argsort(perm)]``.

    Parameters
    ----------
    in_one : np.ndarray of bool, shape (n,)
    perms : np.ndarray of int, shape (B, n)

    Returns
    -------
    np.ndarray of bool, shape (B, n)
    """
    return in_one[np.argsort(perms, axis=1)]


def relabel_renull_test(y_fn, labels, rearrangements, term_fn, statistic="F"):
    """Raw-label permutation of group terms whose dependent variable is recomputed for every relabelling.

    For every rearrangement the dyads get the labels `labels_from_permutations` (the same relabelling that a
    Freedman-Lane permutation with an intercept-only Z implies), `y_fn` recomputes the dyad data under those labels
    (e.g. delta against the within-group nulls of the relabelled groups), and `term_fn` computes the full model's
    statistic (`statistic`, see `term_statistic`) of every tested term. Not Freedman-Lane: there is no nuisance covariate to residualize.

    Parameters
    ----------
    y_fn : callable
        ``y_fn(labels (L, n) bool) -> (L, n, n_cells)``.
    labels : np.ndarray of bool, shape (n,)
        Observed membership of group 1.
    rearrangements : np.ndarray of int, shape (B, n)
        Row permutations (`rearrangement_matrix`, mode 'permute').
    term_fn : callable
        ``term_fn(Y (L, n, n_cells), labels (L, n) bool, statistic) -> (L, m)`` statistic of m tests per labelling.
    statistic : {'F', 'ss_hyp', 'beta'}, optional
        Passed to `term_fn`.

    Returns
    -------
    F_obs : np.ndarray, shape (m,)
    F_perm : np.ndarray, shape (B, m)
    """
    all_labels = np.vstack([labels[None], labels_from_permutations(labels, rearrangements)])
    Y = y_fn(all_labels)
    assert Y.shape[:2] == all_labels.shape, "y_fn must return one (n, n_cells) table per labelling"
    F = term_fn(Y, all_labels, statistic)
    assert F.ndim == 2 and F.shape[0] == all_labels.shape[0], "term_fn must return (labellings, tests)"
    return F[0], F[1:]
