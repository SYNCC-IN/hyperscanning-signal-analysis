"""Tests of src/perm_anova.py (factorial permutation ANOVA, Freedman-Lane)."""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, f_oneway
from statsmodels.stats.anova import AnovaRM

from src.perm_anova import (edge_factor_contrasts, f_statistic, freedman_lane, labels_from_permutations,
                            rearrangement_matrix, relabel_renull_test, rm_contrasts)
from src.sim_inference import family_pvalues

EDGES = ["cg:ROI->child:ROI", "child:ROI->cg:ROI", "cg:HRV->child:HRV", "child:HRV->cg:HRV",
         "cg:HRV->child:ROI", "child:HRV->cg:ROI", "cg:ROI->child:HRV", "child:ROI->cg:HRV"]


def test_rm_contrasts_layout_and_orthonormality():
    """Film-major layout: the film contrast is constant within a film, the mean contrast is 1/sqrt(n_cells)."""
    levels = {"film": 3, "edge": 8}
    mean = rm_contrasts(levels, ())
    assert mean.shape == (24, 1) and np.allclose(mean, 1 / np.sqrt(24))
    film = rm_contrasts(levels, ("film",))
    assert film.shape == (24, 2)
    assert np.allclose(film.reshape(3, 8, 2), film.reshape(3, 8, 2)[:, :1, :])      # constant over edges within film
    assert rm_contrasts(levels, ("edge",)).shape == (24, 7)
    assert rm_contrasts(levels, ("film", "edge")).shape == (24, 14)
    full = np.hstack([mean, film, rm_contrasts(levels, ("edge",)), rm_contrasts(levels, ("film", "edge"))])
    assert np.allclose(full.T @ full, np.eye(24))


def test_edge_factor_contrasts():
    """Direction/source/target contrasts are orthonormal, mutually orthogonal and orthogonal to the edge mean."""
    cols = np.hstack([edge_factor_contrasts(EDGES, f) for f in ("direction", "source", "target")])
    assert np.allclose(cols.T @ cols, np.eye(3))
    assert np.allclose(cols.sum(axis=0), 0)
    assert edge_factor_contrasts(EDGES, "direction")[0, 0] > 0 and edge_factor_contrasts(EDGES, "direction")[1, 0] < 0
    with pytest.raises(AssertionError):
        edge_factor_contrasts(EDGES[:7] + [EDGES[0]], "direction")


def test_f_matches_anovarm_within_design():
    """Within-only design (film x edge): F of every term equals statsmodels AnovaRM."""
    rng = np.random.default_rng(0)
    n, levels = 8, {"film": 3, "edge": 2}
    y = rng.normal(size=(n, 6)) + np.array([0.0, 0.5, 0.2, 0.9, 0.1, 0.3])
    long = pd.DataFrame([{"subj": i, "film": f, "edge": e, "y": y[i, f * 2 + e]}
                         for i in range(n) for f in range(3) for e in range(2)])
    ref = AnovaRM(long, "y", "subj", within=["film", "edge"]).fit().anova_table["F Value"]
    ones, empty = np.ones((n, 1)), np.zeros((n, 0))
    for term, label in ((("film",), "film"), (("edge",), "edge"), (("film", "edge"), "film:edge")):
        assert f_statistic(y @ rm_contrasts(levels, term), ones, empty) == pytest.approx(ref[label], rel=1e-10)


def test_f_matches_mixed_anova_by_hand():
    """Group x film mixed design: group, film and group x film F equal the classical sphericity-assumed formulas."""
    rng = np.random.default_rng(1)
    n_g, n_f = 7, 3
    y = rng.normal(size=(2 * n_g, n_f)) + np.r_[np.zeros(n_g), np.ones(n_g)][:, None] * np.array([0.0, 0.4, 1.0])
    g = np.r_[-np.ones(n_g), np.ones(n_g)][:, None]
    ones = np.ones((2 * n_g, 1))
    levels = {"film": n_f}
    # group: one-way ANOVA on the subject means
    ref_group = f_oneway(y[:n_g].mean(1), y[n_g:].mean(1)).statistic
    assert f_statistic(y @ rm_contrasts(levels, ()), g, ones) == pytest.approx(ref_group, rel=1e-10)
    # within-subject error of the mixed design
    groups = [y[:n_g], y[n_g:]]
    cell = np.stack([b.mean(0) for b in groups])                        # (group, film)
    ss_err = sum(((b - b.mean(1, keepdims=True) - cell[j] + cell[j].mean()) ** 2).sum() for j, b in enumerate(groups))
    df_err = (2 * n_g - 2) * (n_f - 1)
    grand, film_mean = y.mean(), y.mean(0)
    ss_film = 2 * n_g * ((film_mean - grand) ** 2).sum()
    ss_int = n_g * ((cell - cell.mean(1, keepdims=True) - film_mean + grand) ** 2).sum()
    C_film = rm_contrasts(levels, ("film",))
    assert f_statistic(y @ C_film, ones, g) == pytest.approx((ss_film / (n_f - 1)) / (ss_err / df_err), rel=1e-10)
    assert f_statistic(y @ C_film, g, ones) == pytest.approx((ss_int / (n_f - 1)) / (ss_err / df_err), rel=1e-10)


def test_freedman_lane_intercept_only_equals_label_permutation():
    """With Z = intercept, Freedman-Lane F (and p) equal plain label permutation of F on the same rearrangements."""
    rng = np.random.default_rng(2)
    n = 20
    in_one = np.r_[np.zeros(10, bool), np.ones(10, bool)]
    Y = rng.normal(size=(n, 3)) + 0.5 * in_one[:, None]
    ones = np.ones((n, 1))
    perms = rearrangement_matrix(n, 500, "permute", rng)
    F_obs, F_perm = freedman_lane(Y, np.where(in_one, 1.0, -1.0)[:, None], ones, perms, "permute")
    labels = labels_from_permutations(in_one, perms)
    F_label = f_statistic(Y, np.where(labels, 1.0, -1.0)[:, :, None], ones)
    assert np.allclose(F_perm, F_label, rtol=1e-12, atol=0)
    assert np.array_equal(family_pvalues(np.array([F_obs]), F_perm[:, None])["unadjusted"],
                          family_pvalues(np.array([F_obs]), F_label[:, None])["unadjusted"])


def test_relabel_renull_with_fixed_y_equals_freedman_lane():
    """`relabel_renull_test` with a label-independent y_fn reproduces the intercept-only Freedman-Lane distribution."""
    rng = np.random.default_rng(3)
    n = 12
    in_one = np.r_[np.zeros(6, bool), np.ones(6, bool)]
    Y = rng.normal(size=(n, 4))
    ones = np.ones((n, 1))
    perms = rearrangement_matrix(n, 300, "permute", rng)
    F_obs, F_perm = freedman_lane(Y, np.where(in_one, 1.0, -1.0)[:, None], ones, perms, "permute")
    obs, null = relabel_renull_test(lambda labels: np.broadcast_to(Y, labels.shape + (4,)), in_one, perms,
                                    lambda y, labels: f_statistic(y, np.where(labels, 1.0, -1.0)[:, :, None], ones)[:, None])
    assert obs[0] == pytest.approx(F_obs) and np.allclose(null[:, 0], F_perm)


def test_exhaustive_enumeration_gives_exact_p():
    """With fewer rearrangements than n_perm all are used (identity left out) and p is the exact p-value."""
    rng = np.random.default_rng(4)
    n = 5
    in_one = np.array([False, False, True, True, True])
    Y = rng.normal(size=(n, 2)) + in_one[:, None]
    X, ones = np.where(in_one, 1.0, -1.0)[:, None], np.ones((n, 1))
    perms = rearrangement_matrix(n, 1000, "permute", rng)
    assert perms.shape == (math.factorial(n) - 1, n)
    F_obs, F_perm = freedman_lane(Y, X, ones, perms, "permute")
    every = np.array([f_statistic(Y[list(p)], X, ones) for p in itertools.permutations(range(n))])
    tol = 1e-9 * F_obs                      # mathematically tied arrangements (e.g. all signs flipped) differ in rounding
    p = (1 + np.sum(F_perm >= F_obs - tol)) / (1 + len(F_perm))
    assert p == pytest.approx(np.mean(every >= F_obs - tol))

    n = 6
    g = np.r_[-np.ones(3), np.ones(3)][:, None]
    Y = rng.normal(size=(n, 2)) + 0.7
    flips = rearrangement_matrix(n, 1000, "signflip", rng)
    assert flips.shape == (2 ** n - 1, n)
    F_obs, F_perm = freedman_lane(Y, np.ones((n, 1)), g, flips, "signflip")
    fitted = g @ (np.linalg.pinv(g) @ Y)
    every = np.array([f_statistic(fitted + np.array(s)[:, None] * (Y - fitted), np.ones((n, 1)), g)
                      for s in itertools.product([1.0, -1.0], repeat=n)])
    tol = 1e-9 * F_obs                      # mathematically tied arrangements (e.g. all signs flipped) differ in rounding
    p = (1 + np.sum(F_perm >= F_obs - tol)) / (1 + len(F_perm))
    assert p == pytest.approx(np.mean(every >= F_obs - tol))


@pytest.mark.parametrize("mode", ["permute", "signflip"])
def test_null_rejection_rate_heavy_tails(mode):
    """Toy null with t3 errors, 2000 replicates: rejection rate inside the binomial 95% interval around 0.05.

    permute: group effect with a nuisance covariate that is correlated with group and drives the outcome;
    signflip: within-dyad intercept test with a group shift in Z (the mean of the group means is 0).
    """
    rng = np.random.default_rng(5 if mode == "permute" else 6)
    n, k, n_rep, n_perm, alpha = 30, 3, 2000, 999, 0.05
    g = np.r_[-np.ones(15), np.ones(15)][:, None]
    ones = np.ones((n, 1))
    rejected = 0
    for _ in range(n_rep):
        noise = rng.standard_t(3, size=(n, k))
        if mode == "permute":
            c = rng.normal(0.5 * g[:, 0], 1.0)[:, None]
            Y = 0.8 * c + noise
            X, Z = g, np.hstack([ones, c])
        else:
            Y = 0.8 * g + noise
            X, Z = ones, g
        F_obs, F_perm = freedman_lane(Y, X, Z, rearrangement_matrix(n, n_perm, mode, rng), mode)
        rejected += (1 + np.sum(F_perm >= F_obs)) / (1 + n_perm) <= alpha
    low, high = binom.interval(0.95, n_rep, alpha)
    assert low <= rejected <= high, f"{rejected} rejections out of {n_rep}, interval [{low}, {high}]"
