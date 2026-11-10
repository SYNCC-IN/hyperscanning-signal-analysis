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


# ---------------------------------------------------------------------------
# Multiplicity across edges (max-T, min-p, Bonferroni)
# ---------------------------------------------------------------------------
from src.sim_inference import (family_pvalues, group_permutation_maxT, pairing_permutation_maxT, score_study,
                               wilson_interval)

ADJUSTED = ("bonferroni", "maxT", "maxT_single_step", "minP", "minP_single_step")


def test_one_edge_family_adjusted_equals_unadjusted():
    """With a single test, every adjustment returns the unadjusted p-value."""
    rng = np.random.default_rng(10)
    out = pairing_permutation_maxT(rng.normal(size=(15, 15, 1)), rng, 500)
    for m in ADJUSTED:
        np.testing.assert_allclose(out[m], out["unadjusted"])


def test_one_edge_family_matches_the_single_edge_test():
    """On the same re-pairings, the unadjusted family p equals `pairing_permutation_pvalue`."""
    table = np.random.default_rng(11).normal(size=(15, 15))
    table[np.arange(15), np.arange(15)] += 0.5
    p_single = pairing_permutation_pvalue(table, np.random.default_rng(5), 500)
    p_family = pairing_permutation_maxT(table[:, :, None], np.random.default_rng(5), 500)["unadjusted"][0]
    assert np.isclose(p_single, p_family)


def test_adjusted_p_not_below_unadjusted():
    """Adjusted p-values are never smaller than the unadjusted ones (study level and group level)."""
    rng = np.random.default_rng(12)
    tables = rng.normal(size=(20, 20, 8))
    tables[np.arange(20), np.arange(20), :3] += 0.6
    delta = rng.normal(size=(20, 8))
    delta[10:, :2] += 1.0
    for out in (pairing_permutation_maxT(tables, rng, 500), group_permutation_maxT(delta, np.repeat([0, 1], 10), rng, 500)):
        for m in ADJUSTED:
            assert np.all(out[m] >= out["unadjusted"] - 1e-12)


def test_identical_edges_adjusted_equals_unadjusted():
    """Perfectly dependent tests: max-T and min-p reduce to the unadjusted test, Bonferroni does not."""
    rng = np.random.default_rng(13)
    table = rng.normal(size=(15, 15))
    table[np.arange(15), np.arange(15)] += 0.4
    out = pairing_permutation_maxT(np.repeat(table[:, :, None], 6, axis=2), rng, 500)
    for m in ("maxT", "maxT_single_step", "minP", "minP_single_step"):
        np.testing.assert_allclose(out[m], out["unadjusted"])
    assert np.all(out["bonferroni"] >= out["unadjusted"])


def test_stepdown_rejects_a_superset_of_single_step():
    """Step-down p-values are at most the single-step ones, so every single-step rejection survives."""
    rng = np.random.default_rng(14)
    for _ in range(50):
        observed, null = rng.normal(size=8) + rng.uniform(0, 3, 8), rng.normal(size=(300, 8))
        out = family_pvalues(observed, null)
        for m in ("maxT", "minP"):
            assert np.all(out[m] <= out[f"{m}_single_step"] + 1e-12)
            assert set(np.flatnonzero(out[f"{m}_single_step"] < 0.05)) <= set(np.flatnonzero(out[m] < 0.05))


def test_maxT_controls_fwer_under_the_global_null():
    """Independent edges, no planted structure: max-T FWER within the 95% Wilson interval of 0.05; Bonferroni not above."""
    rng = np.random.default_rng(15)
    n_runs, n_perm = 500, 400
    any_reject = {m: 0 for m in ("unadjusted", "bonferroni", "maxT", "minP")}
    for _ in range(n_runs):
        out = pairing_permutation_maxT(rng.normal(size=(20, 20, 8)), rng, n_perm)
        for m in any_reject:
            any_reject[m] += bool(np.any(out[m] < 0.05))
    low, high = wilson_interval(int(0.05 * n_runs), n_runs)
    assert low <= any_reject["maxT"] / n_runs <= high
    assert low <= any_reject["minP"] / n_runs <= high
    assert any_reject["bonferroni"] / n_runs <= high
    assert any_reject["unadjusted"] / n_runs > 0.2           # 1 - 0.95**8 = 0.34 without correction


def test_group_maxT_controls_fwer_under_the_global_null():
    """Group contrast of 8 independent null edges: max-T FWER within the Wilson interval of 0.05."""
    rng = np.random.default_rng(16)
    groups = np.repeat([0, 1], 10)
    n_runs = 500
    hits = sum(bool(np.any(group_permutation_maxT(rng.lognormal(size=(20, 8)), groups, rng, 400)["maxT"] < 0.05))
               for _ in range(n_runs))
    low, high = wilson_interval(int(0.05 * n_runs), n_runs)
    assert low <= hits / n_runs <= high


def test_score_study_adds_family_columns_and_keeps_per_edge_results():
    """multiplicity adds adjusted columns for family edges only; the per-edge readouts are unchanged."""
    rng = np.random.default_rng(17)
    n, k = 8, 3
    real = rng.normal(size=(n, k, k))
    pair_values = rng.normal(size=(n, n, k, k))
    pair_values[np.arange(n), np.arange(n)] = np.nan
    groups = np.repeat([0, 1], 4)
    edges = {"a": (0, 1), "b": (1, 0), "c": (2, 0)}
    plain = score_study(real, pair_values, real, "pooled", groups, edges, 0.05, 0.05, np.random.default_rng(1), 200)
    multi = score_study(real, pair_values, real, "pooled", groups, edges, 0.05, 0.05, np.random.default_rng(1), 200,
                        family=["a", "b"], multiplicity=["maxT", "bonferroni"])
    for name in edges:
        for key, value in plain[name].items():
            assert multi[name][key] == value or (np.isnan(value) and np.isnan(multi[name][key]))
    assert np.isfinite(multi["a"]["study_p_adj_maxT"]) and np.isnan(multi["c"]["study_p_adj_maxT"])
    assert multi["c"]["any_reject_study_maxT"] == multi["a"]["any_reject_study_maxT"]


def test_family_outside_scored_edges_and_separate_rng():
    """A dict family may add edges scored only by the family test; a separate generator keeps the main stream intact."""
    rng = np.random.default_rng(18)
    n, k = 8, 3
    real = rng.normal(size=(n, k, k))
    pair_values = rng.normal(size=(n, n, k, k))
    pair_values[np.arange(n), np.arange(n)] = np.nan
    edges = {"a": (0, 1)}
    family = {"a": (0, 1), "b": (1, 0), "c": (2, 0)}
    main = np.random.default_rng(2)
    out = score_study(real, pair_values, real, "pooled", None, edges, 0.05, 0.05, main, 200,
                      family=family, multiplicity="maxT", multiplicity_rng=np.random.default_rng(3))
    reference = np.random.default_rng(2)
    score_study(real, pair_values, real, "pooled", None, edges, 0.05, 0.05, reference, 200)
    assert main.random() == reference.random()
    assert set(out) == {"a", "b", "c"} and "study_p" not in out["c"] and np.isfinite(out["c"]["study_p_adj_maxT"])
    assert np.isnan(out["a"]["any_reject_study_none"])


# ---------------------------------------------------------------------------
# Exact enumeration of small permutation sets
# ---------------------------------------------------------------------------
import itertools

from src.sim_inference import caregiver_permutations, label_permutations, sign_flips


def test_exact_relabelling_uses_all_252_once_and_gives_the_exact_p():
    """5 vs 5 dyads: the 251 relabellings other than the observed one are each used once; p is the exact p."""
    rng = np.random.default_rng(30)
    in_one = np.repeat([False, True], 5)
    labels = label_permutations(in_one, rng, 1000)
    assert labels.shape == (251, 10)
    assert len({row.tobytes() for row in labels}) == 251
    assert not (labels == in_one).all(axis=1).any()
    a, b = rng.normal(size=5), rng.normal(size=5) + 0.8
    pooled = np.concatenate([a, b])
    stat = lambda members: abs(pooled[list(members)].mean() - np.delete(pooled, list(members)).mean())
    observed = stat(range(5))
    exact = np.mean([stat(m) >= observed - 1e-12 for m in itertools.combinations(range(10), 5)])
    assert np.isclose(permutation_pvalue(a, b, rng, 1000), exact)
    groups = in_one.astype(int)
    p_group = group_permutation_maxT(pooled[:, None], groups[::-1], rng, 1000)["unadjusted"][0]
    assert np.isclose(p_group, exact)


def test_random_relabelling_when_there_are_more_arrangements_than_permutations():
    """C(20, 10) > 1000: the usual random draws."""
    assert label_permutations(np.repeat([False, True], 10), np.random.default_rng(0), 1000).shape == (1000, 20)


def test_enumeration_keeps_the_random_stream():
    """The random draws are made in both branches, so later draws from the generator are unchanged."""
    rng_a, rng_b = np.random.default_rng(31), np.random.default_rng(31)
    label_permutations(np.repeat([False, True], 5), rng_a, 1000)            # enumerated
    rng_b.permuted(np.tile(np.repeat([False, True], 5), (1000, 1)), axis=1)  # what the random branch draws
    assert rng_a.random() == rng_b.random()


def test_exact_repairings_and_sign_flips():
    """4 dyads: 23 non-identity re-pairings; 2 groups of 2: 3; 5 dyads: 31 non-trivial sign vectors."""
    rng = np.random.default_rng(32)
    perms = caregiver_permutations(4, rng, 1000)
    assert perms.shape == (23, 4) and len({p.tobytes() for p in perms}) == 23
    within = caregiver_permutations(4, rng, 1000, groups=np.array([0, 0, 1, 1]))
    assert within.shape == (3, 4) and set(within[:, :2].ravel()) <= {0, 1}
    flips = sign_flips(5, rng, 1000)
    assert flips.shape == (31, 5) and len({f.tobytes() for f in flips}) == 31 and not (flips == 1).all(axis=1).any()
    assert sign_flips(12, rng, 1000).shape == (1000, 12)
