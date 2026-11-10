"""Checks of the simulation generators in `src.sim_generators`."""

import numpy as np

from src.sim_generators import coupling_window, simulate_dyads, simulate_dyads_partial

RADII = np.array([[0.45, 0.65], [0.45, 0.65], [0.80, 0.92], [0.80, 0.92]])
F0 = np.array([[0.20, 0.50], [0.20, 0.50], [0.25, 0.45], [0.25, 0.45]])
EDGE = [(1, 0, 1, 0.4)]


def test_partial_coupling_with_all_dyads_and_all_time_reproduces_design_a():
    """q = 1 and coupling on 100% of the time: identical to `simulate_dyads` for the same seed."""
    reference = simulate_dyads(np.random.default_rng(7), 6, 150, 2.5, RADII, F0, EDGE, 3)
    all_dyads = simulate_dyads_partial(np.random.default_rng(7), 6, 150, 2.5, RADII, F0, EDGE, 3, np.ones(6, bool))
    all_time = simulate_dyads_partial(np.random.default_rng(7), 6, 150, 2.5, RADII, F0, EDGE, 3, np.ones(6, bool),
                                      np.ones((6, 150), bool))
    np.testing.assert_array_equal(all_dyads, reference)
    np.testing.assert_array_equal(all_time, reference)


def test_uncoupled_dyads_and_off_time_have_no_coupling():
    """An uncoupled dyad equals the same dyad simulated with gain 0; coupling off all the time likewise."""
    no_edge = simulate_dyads(np.random.default_rng(8), 3, 150, 2.5, RADII, F0, [], 3)
    uncoupled = simulate_dyads_partial(np.random.default_rng(8), 3, 150, 2.5, RADII, F0, EDGE, 3, np.zeros(3, bool))
    off = simulate_dyads_partial(np.random.default_rng(8), 3, 150, 2.5, RADII, F0, EDGE, 3, np.ones(3, bool),
                                 np.zeros((3, 150), bool))
    np.testing.assert_allclose(uncoupled, no_edge)
    np.testing.assert_allclose(off, no_edge)


def test_coupling_window_is_one_contiguous_episode_of_the_requested_length():
    """75% of 150 samples: one block of 112 or 113 samples."""
    mask = coupling_window(np.random.default_rng(9), 150, 0.75)
    assert mask.sum() == round(0.75 * 150)
    assert np.count_nonzero(np.diff(mask.astype(int)) != 0) <= 2


def test_node_noise_zero_is_unchanged_and_sets_the_noise_level():
    """Noise SD 0 returns the input; SD 1 doubles the variance of the noisy node only."""
    from src.sim_generators import add_node_noise
    raw = simulate_dyads(np.random.default_rng(10), 4, 2000, 2.5, RADII, F0, EDGE, 3)
    np.testing.assert_array_equal(add_node_noise(raw, [3], 0.0, np.random.default_rng(1)), raw)
    noisy = add_node_noise(raw, [3], 1.0, np.random.default_rng(1))
    np.testing.assert_array_equal(noisy[:, :3], raw[:, :3])
    ratio = noisy[:, 3].var(axis=1) / raw[:, 3].var(axis=1)
    assert np.all(np.abs(ratio - 2.0) < 0.2)
