"""Checks of the dyad-level, study-level and group-level tests of `src.sim_inference`."""

import numpy as np

from src.sim_inference import dyad_pvalues, pairing_permutation_pvalue, permutation_pvalue


def test_dyad_pvalues_count_null_values_at_or_above_real():
    """p = (1 + #{null >= real}) / (1 + n_null), with the pooled null of all foreign pairs."""
    rng = np.random.default_rng(0)
    n = 6
    pair_values = rng.normal(size=(n, n, 2, 2))
    pair_values[np.arange(n), np.arange(n)] = np.nan
    real = rng.normal(size=(n, 2, 2))
    p = dyad_pvalues(real, pair_values, "pooled")
    null = pair_values[np.isfinite(pair_values[:, :, 0, 0])][:, 0, 1]
    assert np.isclose(p[2, 0, 1], (1 + np.sum(null >= real[2, 0, 1])) / (1 + null.size))


def test_pairing_permutation_is_calibrated_for_a_skewed_null():
    """Under exchangeable (iid, strongly skewed) values the false-positive rate is nominal."""
    rng = np.random.default_rng(1)
    p = [pairing_permutation_pvalue(rng.lognormal(sigma=1.0, size=(20, 20)), rng, 200) for _ in range(2000)]
    assert 0.035 <= np.mean(np.array(p) < 0.05) <= 0.065


def test_pairing_permutation_detects_coupled_real_dyads():
    """A raised diagonal (real dyads) gives a small p-value."""
    rng = np.random.default_rng(2)
    table = rng.normal(size=(20, 20))
    table[np.arange(20), np.arange(20)] += 1.5
    assert pairing_permutation_pvalue(table, rng, 1000) < 0.01


def test_within_group_pairing_removes_a_group_effect():
    """Pairs within a group are higher than across groups: significant with the pooled re-pairing, not within groups."""
    rng = np.random.default_rng(3)
    groups = np.repeat([0, 1], 10)
    p_pooled, p_within = [], []
    for _ in range(300):
        table = rng.normal(size=(20, 20)) + 2.0 * (groups[:, None] == groups[None, :])
        p_pooled.append(pairing_permutation_pvalue(table, rng, 200))
        p_within.append(pairing_permutation_pvalue(table, rng, 200, groups))
    assert np.mean(np.array(p_pooled) < 0.05) > 0.9
    assert np.mean(np.array(p_within) < 0.05) < 0.09


def test_group_permutation_reaches_small_p_with_five_per_group():
    """Completely separated groups of 5 give p near the exact minimum 2/252."""
    rng = np.random.default_rng(4)
    p = permutation_pvalue(np.arange(5.0), np.arange(5.0) + 10, rng, 5000)
    assert p < 0.02
