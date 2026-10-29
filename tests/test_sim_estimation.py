"""Checks that the fast simulation estimator reproduces the repository's own estimator."""

import numpy as np
import pytest

from src.connectivity import Granger_estimator
from src.design import window_geometry
from src.sim_estimation import connectivity_cube, fit_mvar
from src.surrogate import delta_and_z


@pytest.mark.parametrize("estimator", ["dDTF", "ffDTF", "GPDC"])
def test_fast_cube_matches_granger_estimator(estimator):
    """The vectorized cube equals `Granger_estimator` on the same windowed fit."""
    rng = np.random.default_rng(1)
    design = rng.normal(size=(4, 150))
    design[1, 1:] += 0.6 * design[0, :-1]
    design = (design - design.mean(1, keepdims=True)) / design.std(1, keepdims=True)
    fs, p = 2.5, 4
    win_len, step = window_geometry(10.0, 0.5, fs)
    freqs = np.linspace(0.02, fs / 2 - 0.02, 100)
    reference, _ = Granger_estimator(design, freqs, fs, p, win_len, step, "linear", ESTIMATOR=estimator)
    coeffs, variance, _ = fit_mvar(design, p, win_len, step)
    fast = connectivity_cube(coeffs, variance, freqs, fs, estimator)
    np.testing.assert_allclose(fast, reference, rtol=1e-8, atol=1e-12)


def test_dtf_rows_sum_to_one():
    """Plain DTF is normalized so that inflows to each target sum to 1 at every frequency."""
    rng = np.random.default_rng(2)
    design = rng.normal(size=(3, 120))
    coeffs, variance, _ = fit_mvar(design, 2, 20, 10)
    cube = connectivity_cube(coeffs, variance, np.linspace(0.05, 1.2, 40), 2.5, "DTF")
    np.testing.assert_allclose(cube.sum(axis=1), 1.0, rtol=1e-10)


def test_delta_and_z_vectorized_matches_repository():
    """The vectorized null statistics equal `src.surrogate.delta_and_z`."""
    from src.sim_inference import null_stats
    rng = np.random.default_rng(3)
    pool = rng.normal(size=(3, 3, 2, 2)) + 5.0
    np.fill_diagonal(pool[:, :, 0, 0], np.nan)
    pair_values = np.full((6, 6, 2, 2), np.nan)
    for b in range(6):
        for c in range(6):
            if b != c:
                pair_values[b, c] = rng.normal(size=(2, 2))
    med, mad, _ = null_stats(pair_values, 0, "pooled")
    off = pair_values[~np.isnan(pair_values[:, :, 0, 0])]
    reference = delta_and_z(1.0, off[:, 0, 1])
    assert np.isclose(med[0, 1], reference["null_median"])
    assert np.isclose(mad[0, 1], reference["null_std"])
