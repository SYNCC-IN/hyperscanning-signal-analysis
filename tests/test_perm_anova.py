"""Tests of src/perm_anova.py (factorial permutation ANOVA, Freedman-Lane)."""

import importlib
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, f_oneway
from statsmodels.stats.anova import AnovaRM

from src.perm_anova import (all_pairs_mean, balanced_mean, balanced_mean_maxT, bootstrap_null_medians, edge_factor_contrasts,
                            f_statistic, freedman_lane, intercept_ss, labels_from_permutations, rearrangement_matrix,
                            relabel_renull_test, repaired_values, rm_contrasts, ss_components, term_statistic)
from src.sim_inference import (caregiver_permutations, family_pvalues, group_null_medians, renull_delta, renull_statistics,
                               sign_flips, signflip_maxT)

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
                                    lambda y, labels, statistic: f_statistic(y, np.where(labels, 1.0, -1.0)[:, :, None], ones)[:, None])
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


# ---------------------------------------------------------------------------
# Contrast invariance with the S13 cell layout and flattening (imported, not re-implemented)
# ---------------------------------------------------------------------------
S13_DIR = Path(__file__).resolve().parents[1] / "simulations_multimodal_MVAR" / "dDTF"


@pytest.fixture(scope="module")
def s13():
    """The S13 script as a module (it reads MODE / ESTIMATOR from sys.argv at import, so argv is set for the import)."""
    saved = sys.argv
    sys.argv = ["S13_group_film_bayes.py", "full", "dDTF"]
    sys.path.insert(0, str(S13_DIR))
    module = importlib.import_module("S13_group_film_bayes")
    sys.argv = saved
    return module


def s13_terms(s13):
    """Within-dyad contrasts of S13 (F, E, FxE and the 1-df edge classes), keyed by term name."""
    terms = {t: C for t, (C, kind) in s13.FL_OMNIBUS.items() if kind == "within"}
    for cls in ("direction", "source", "target"):
        terms[cls] = s13.rm_contrasts(s13.LEVELS, ("edge",), {"edge": s13.EDGE_CLASS[cls]})
    return terms


def s13_patterns(s13, rng):
    """Constant (film x edge) patterns, each in exactly one subspace: mean, film, edge classes, other edge, FxE."""
    n_f, n_e = len(s13.FILMS), len(s13.EDGES)
    film = rng.normal(size=(n_f, 1)); film -= film.mean()
    cls = {c: s13.EDGE_CLASS[c][:, 0] for c in ("direction", "source", "target")}
    other = rng.normal(size=n_e)
    basis = np.column_stack([np.ones(n_e)] + list(cls.values()))
    other -= basis @ np.linalg.lstsq(basis, other, rcond=None)[0]                    # edge pattern outside the classes
    inter = rng.normal(size=(n_f, n_e)); inter -= inter.mean(0) + inter.mean(1, keepdims=True) - inter.mean()
    out = {"mean": np.full((n_f, n_e), 1.7), "film": np.repeat(film, n_e, axis=1), "edge_other": np.tile(other, (n_f, 1)),
           "FxE": inter}
    out.update({f"edge_{c}": np.tile(v * 3.0, (n_f, 1)) for c, v in cls.items()})
    return out


OWN = {"F": {"film"}, "E": {"edge_other", "edge_direction", "edge_source", "edge_target"}, "FxE": {"FxE"},
       "direction": {"edge_direction"}, "source": {"edge_source"}, "target": {"edge_target"}}


def flatten_like_s13(s13, values):
    """Long delta table of `values` (n, film, edge) flattened with the S13 code (`fl_cells`)."""
    n = values.shape[0]
    rows = [{"dyad": d, "film": film, "edge": edge, "delta": values[d, f, e]}
            for d in range(n) for f, film in enumerate(s13.FILMS) for e, edge in enumerate(s13.EDGES)]
    used, Y, cells = s13.fl_cells(pd.DataFrame(rows), n)
    assert used.all() and np.array_equal(Y, values)
    return cells


def test_s13_contrasts_orthogonality(s13):
    """C_A' C_A = I; distinct terms orthogonal except the edge classes, which are nested in E."""
    terms = dict(s13_terms(s13), mean=s13.FL_OMNIBUS["G"][0])
    for a, Ca in terms.items():
        assert np.allclose(Ca.T @ Ca, np.eye(Ca.shape[1]))
        for b, Cb in terms.items():
            if a == b:
                continue
            if {a, b} & {"direction", "source", "target"} and {a, b} & {"E"}:
                nested = Cb if a == "E" else Ca                                     # class contrast inside span(E)
                E = terms["E"]
                assert np.allclose(E @ (E.T @ nested), nested)
                continue
            assert np.allclose(Ca.T @ Cb, 0), (a, b)


@pytest.mark.parametrize("term", ["F", "E", "FxE", "direction", "source", "target"])
def test_s13_within_term_invariance(s13, term):
    """Within-dyad term statistic (sign-flip design X = 1, Z = group) is unchanged by every pattern except its own."""
    rng = np.random.default_rng(7)
    n = 20
    noise = rng.normal(size=(n, len(s13.FILMS), len(s13.EDGES)))
    C = s13_terms(s13)[term]
    X, Z = s13.fl_design(np.r_[np.zeros(10, int), np.ones(10, int)], None)["within"]
    base = f_statistic(flatten_like_s13(s13, noise) @ C, X, Z)
    for name, pattern in s13_patterns(s13, rng).items():
        F = f_statistic(flatten_like_s13(s13, noise + pattern[None]) @ C, X, Z)
        if name in OWN[term]:
            assert abs(F - base) > 1e-3, (term, name)
        else:
            assert F == pytest.approx(base, rel=1e-10, abs=1e-10), (term, name)


@pytest.mark.parametrize("term", ["GxF", "GxE", "Gxdirection", "Gxsource", "Gxtarget", "GxFxE"])
def test_s13_group_term_invariance(s13, term):
    """Group x within term (X = group, Z = 1) is unchanged by every pattern added equally to both groups, and by
    patterns added to one group outside its within-dyad subspace; it changes for its own pattern in one group."""
    rng = np.random.default_rng(8)
    n = 20
    groups = np.r_[np.zeros(10, int), np.ones(10, int)]
    noise = rng.normal(size=(n, len(s13.FILMS), len(s13.EDGES)))
    C = s13.FL_OMNIBUS[term][0]
    X, Z = s13.fl_design(groups, None)["between"]
    base = f_statistic(flatten_like_s13(s13, noise) @ C, X, Z)
    own = {"GxF": OWN["F"], "GxE": OWN["E"], "GxFxE": OWN["FxE"], "Gxdirection": OWN["direction"],
           "Gxsource": OWN["source"], "Gxtarget": OWN["target"]}[term]
    for name, pattern in s13_patterns(s13, rng).items():
        both = f_statistic(flatten_like_s13(s13, noise + pattern[None]) @ C, X, Z)
        assert both == pytest.approx(base, rel=1e-10, abs=1e-10), (term, name, "both groups")
        one = f_statistic(flatten_like_s13(s13, noise + (groups == 1)[:, None, None] * pattern[None]) @ C, X, Z)
        if name in own:
            assert abs(one - base) > 1e-3, (term, name)
        else:
            assert one == pytest.approx(base, rel=1e-10, abs=1e-10), (term, name, "one group")


def test_ss_hyp_renull_equals_existing_mean_difference_test():
    """Renull G with 'ss_hyp' gives the p-values of the existing S13 renull |difference of means| test
    (`renull_statistics`) on the same relabellings: per edge (film mean) and on the edge mean."""
    rng = np.random.default_rng(9)
    n, n_films, n_edges = 20, 3, 8
    in_one = np.r_[np.zeros(10, bool), np.ones(10, bool)]
    real = rng.normal(size=(n, n_films * n_edges)) + 0.4 * in_one[:, None]
    pair_values = rng.normal(size=(n, n, n_films * n_edges))
    pair_values[np.arange(n), np.arange(n)] = np.nan
    perms = rearrangement_matrix(n, 500, "permute", rng)
    labels = np.vstack([in_one[None], labels_from_permutations(in_one, perms)])
    ones = np.ones((n, 1))
    film_mean = rm_contrasts({"film": n_films}, ())
    edge_mean = rm_contrasts({"film": n_films, "edge": n_edges}, ())

    def term_fn(y, lab, statistic):
        X = np.where(lab, 1.0, -1.0)[:, :, None]
        per_edge = y.reshape(y.shape[:2] + (n_films, n_edges))
        cols = [term_statistic(per_edge[..., e] @ film_mean, X, ones, statistic) for e in range(n_edges)]
        return np.stack(cols + [term_statistic(y @ edge_mean, X, ones, statistic)], axis=1)

    obs, null = relabel_renull_test(lambda lab: renull_delta(real, pair_values, lab), in_one, perms, term_fn, "ss_hyp")
    combine = np.hstack([np.vstack([np.eye(n_edges) / n_films] * n_films), np.full((n_films * n_edges, 1), 1 / (n_films * n_edges))])
    old = np.abs(renull_statistics(real, pair_values, np.ones((n, n_edges + 1), bool), labels, combine))
    assert np.array_equal(family_pvalues(obs, null)["unadjusted"], family_pvalues(old[0], old[1:])["unadjusted"])


def test_balanced_mean_maxT_equals_signflip_maxT():
    """Sign flips passed as a null of `balanced_mean_maxT` reproduce `signflip_maxT` (same signs, unequal groups, NaN)."""
    rng = np.random.default_rng(3)
    groups = np.r_[np.zeros(9, int), np.ones(12, int)]
    contrast = rng.normal(0.2, 1.0, (21, 8))
    contrast[4, 2] = np.nan
    expected = signflip_maxT(contrast, np.random.default_rng(5), 500, groups)
    signs = sign_flips(21, np.random.default_rng(5), 500)
    got = balanced_mean_maxT(contrast, signs[:, :, None] * contrast[None], groups)
    for key in ("unadjusted", "maxT"):
        assert np.allclose(got[key], expected[key], rtol=0, atol=1e-12)


def test_intercept_ss_equals_term_statistic():
    """`intercept_ss` is tr SS_hyp of the intercept of Y ~ [1, group] (sum-to-zero), batched, unequal groups."""
    rng = np.random.default_rng(4)
    groups = np.r_[np.zeros(7, int), np.ones(11, int)]
    Y = rng.normal(size=(5, 18, 3))
    X, Z = np.ones((18, 1)), np.where(groups == 1, 1.0, -1.0)[:, None]
    assert np.allclose(intercept_ss(Y, groups), term_statistic(Y, X, Z, "ss_hyp"))


def test_all_pairs_mean_centres_exhaustive_repairing():
    """Averaged over all within-group re-pairings, the re-paired group means equal `all_pairs_mean`."""
    rng = np.random.default_rng(6)
    groups = np.r_[np.zeros(4, int), np.ones(4, int)]
    table = rng.normal(size=(8, 8, 2))
    perms = caregiver_permutations(8, rng, 10_000, groups)
    perms = np.vstack([np.arange(8)[None], perms])                     # enumeration leaves out the identity
    assert len(perms) == math.factorial(4) ** 2
    repaired = repaired_values(table, perms)
    for label in (0, 1):
        assert np.allclose(repaired[:, groups == label].mean(axis=(0, 1)), all_pairs_mean(table, groups)[label])
    assert np.allclose(balanced_mean(repaired, groups).mean(axis=0), all_pairs_mean(table, groups).mean(axis=0))


def test_bootstrap_null_medians_shape_and_constant_table():
    """A table that is constant within each group gives that constant in every bootstrap replicate; the diagonal is skipped."""
    groups = np.r_[np.zeros(5, int), np.ones(5, int)]
    table = np.where(groups[:, None, None] == 1, 2.0, -1.0) * np.ones((10, 10, 3))
    table[np.arange(10), np.arange(10)] = np.nan
    out = bootstrap_null_medians(table, groups, np.random.default_rng(0), 50)
    assert out.shape == (50, 2, 3)
    assert np.allclose(out[:, 0], -1.0) and np.allclose(out[:, 1], 2.0)


def test_bootstrap_null_medians_centre():
    """The bootstrap medians scatter around the observed within-group median."""
    rng = np.random.default_rng(1)
    groups = np.r_[np.zeros(15, int), np.ones(15, int)]
    table = rng.normal(size=(30, 30, 2)) + rng.normal(size=(30, 1, 1)) + rng.normal(size=(1, 30, 1))
    table[np.arange(30), np.arange(30)] = np.nan
    observed = group_null_medians(table, (groups == 1)[None])[0]
    boot = bootstrap_null_medians(table, groups, rng, 400)
    assert np.all(np.abs(boot.mean(axis=0) - observed) < 3 * boot.std(axis=0) / np.sqrt(400) + 0.05)
    assert np.all(boot.std(axis=0) > 0.05)


def test_freedman_lane_zero_shifts():
    """Zero shifts leave the Freedman-Lane distribution unchanged."""
    rng = np.random.default_rng(2)
    Y, X, Z = rng.normal(size=(12, 2)), np.ones((12, 1)), np.r_[-np.ones(6), np.ones(6)][:, None]
    flips = sign_flips(12, rng, 200)
    a = freedman_lane(Y, X, Z, flips, "signflip", "ss_hyp")
    b = freedman_lane(Y, X, Z, flips, "signflip", "ss_hyp", shifts=np.zeros((len(flips), 12, 2)))
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
