"""Checks of the re-nulled group contrast (`renull=True`) of `src.sim_inference`."""

import numpy as np
from scipy.stats import kstest

from src.sim_inference import (delta_and_z_all, group_mean_difference, group_permutation_maxT, pairing_permutation_pvalue, permutation_pvalue,
                               renull_delta, renull_statistics, score_study)

N_PER_GROUP = 8
N_REPLICATES = 2000
N_PERM = 200


def exchangeable_study(rng, n, k=1):
    """Real values (n, k) and a surrogate table (n, n, k), all entries i.i.d. N(0, 1), NaN on the diagonal."""
    pair_values = rng.normal(size=(n, n, k))
    pair_values[np.arange(n), np.arange(n)] = np.nan
    return rng.normal(size=(n, k)), pair_values


def test_renull_delta_with_observed_labels_equals_within_group_delta():
    """For the observed labelling the re-nulled delta is the delta of `delta_and_z_all` with the within-group null."""
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1], [5, 6])
    real, pair_values = exchangeable_study(rng, 11, 4)
    pair_values[2, 7] = np.nan                                       # an unstable surrogate fit
    delta, _ = delta_and_z_all(real.reshape(11, 2, 2), pair_values.reshape(11, 11, 2, 2), "within_group", groups)
    np.testing.assert_allclose(renull_delta(real, pair_values, (groups == 1)[None])[0], delta.reshape(11, 4))


def test_renull_contrast_combines_films_linearly():
    """With films stacked as columns, `combine` gives the group difference of the film contrast of the deltas."""
    rng = np.random.default_rng(1)
    groups = np.repeat([0, 1], 6)
    real, pair_values = exchangeable_study(rng, 12, 6)               # 3 films x 2 edges
    eye = np.eye(2)
    combine = np.vstack([eye, -eye / 2, -eye / 2])
    labels = (groups == 1)[None]
    expected = group_mean_difference(renull_delta(real, pair_values, labels)[0] @ combine, labels)
    got = renull_statistics(real, pair_values, np.ones((12, 2), bool), labels, combine)
    np.testing.assert_allclose(got, expected)


def test_renull_and_fixed_delta_draw_the_same_permutations():
    """Both versions consume the same draws, so the generator is in the same state afterwards."""
    groups = np.repeat([0, 1], 8)
    real, pair_values = exchangeable_study(np.random.default_rng(2), 16)
    delta, _ = delta_and_z_all(real.reshape(16, 1, 1), pair_values.reshape(16, 16, 1, 1), "within_group", groups)
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    group_permutation_maxT(delta.reshape(16, 1), groups, rng_a, 100)
    group_permutation_maxT(delta.reshape(16, 1), groups, rng_b, 100, renull=True, real=real, pair_values=pair_values)
    assert rng_a.random() == rng_b.random()


def test_score_study_renull_reports_the_re_nulled_group_p():
    """`score_study(..., renull=True)` gives the per-edge group p of `group_permutation_maxT` with renull."""
    rng = np.random.default_rng(4)
    groups = np.repeat([0, 1], 6)
    real, pair_values = exchangeable_study(rng, 12, 4)
    real, pair_values = real.reshape(12, 2, 2), pair_values.reshape(12, 12, 2, 2)
    out = score_study(real, pair_values, real, "within_group", groups, {"e": (1, 0)}, 0.05, 0.05,
                      np.random.default_rng(5), 300, renull=True)
    reference_rng = np.random.default_rng(5)
    table = pair_values[:, :, 0, 1].copy()
    table[np.arange(12), np.arange(12)] = real[:, 0, 1]
    pairing_permutation_pvalue(table, reference_rng, 300, groups)    # the study-level test is drawn first
    delta, _ = delta_and_z_all(real, pair_values, "within_group", groups)
    expected = group_permutation_maxT(delta[:, 0, 1][:, None], groups, reference_rng, 300, renull=True,
                                      real=real[:, 0, 1][:, None], pair_values=pair_values[:, :, 0, 1][:, :, None])
    assert np.isclose(out["e"]["group_p"], expected["unadjusted"][0])


def test_renull_group_contrast_is_uniform_under_exchangeable_values():
    """Both groups from one distribution (i.i.d. real and surrogate values): renull p-values are uniform (KS p > 0.01).

    The fixed-delta version (current default) is run on the same data; its rate at 0.05 is printed for the report.
    """
    rng = np.random.default_rng(20261113)
    n = 2 * N_PER_GROUP
    groups = np.repeat([0, 1], N_PER_GROUP)
    p_renull, p_fixed = [], []
    for _ in range(N_REPLICATES):
        real, pair_values = exchangeable_study(rng, n)
        delta, _ = delta_and_z_all(real.reshape(n, 1, 1), pair_values.reshape(n, n, 1, 1), "within_group", groups)
        d = delta.reshape(n)
        p_fixed.append(permutation_pvalue(d[groups == 0], d[groups == 1], rng, N_PERM))
        p_renull.append(group_permutation_maxT(d[:, None], groups, rng, N_PERM, renull=True, real=real,
                                               pair_values=pair_values)["unadjusted"][0])
    p_renull, p_fixed = np.array(p_renull), np.array(p_fixed)
    ks_renull, ks_fixed = kstest(p_renull, "uniform").pvalue, kstest(p_fixed, "uniform").pvalue
    print(f"\n{N_PER_GROUP} vs {N_PER_GROUP}, {N_REPLICATES} replicates, {N_PERM} permutations: "
          f"rate(p < 0.05) renull = {np.mean(p_renull < 0.05):.4f} (KS p = {ks_renull:.3f}), "
          f"fixed delta = {np.mean(p_fixed < 0.05):.4f} (KS p = {ks_fixed:.2g})")
    assert ks_renull > 0.01
