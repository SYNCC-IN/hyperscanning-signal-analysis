"""Checks that the fast simulation estimator reproduces the repository's own estimator."""

import numpy as np
import pytest

from src.connectivity import Granger_estimator
from src.design import window_geometry
from src.mtmvar import ols_coeff
from src.sim_estimation import connectivity_cube, fit_mvar, true_conditional_gc
from src.sim_generators import build_coupling, simulate_var
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
    coeffs, variance, _ = fit_mvar(design, p, win_len, step, "linear", method="yw")   # Granger_estimator: Yule-Walker, linear detrending
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


def test_ols_recovers_coefficients_on_long_data():
    """On a long recording, the stacked-window OLS fit recovers the generating coefficients and orientation."""
    rng = np.random.default_rng(4)
    coupling = build_coupling([0.55, 0.86], [0.35, 0.35], 2.5, [(1, 0, 1, 0.4)], 3)
    x = simulate_var(coupling, 40000, rng)
    coeffs, variance = ols_coeff(x, 3)
    np.testing.assert_allclose(coeffs, coupling, atol=0.03)
    np.testing.assert_allclose(variance, np.eye(2), atol=0.05)


def test_ols_does_not_use_lags_across_windows():
    """Windows are separate realizations: swapping the order of windows leaves the OLS fit unchanged."""
    rng = np.random.default_rng(5)
    stack = rng.normal(size=(3, 25, 6))
    a1, v1 = ols_coeff(stack, 2)
    a2, v2 = ols_coeff(stack[:, :, ::-1], 2)
    np.testing.assert_allclose(a1, a2, atol=1e-12)
    np.testing.assert_allclose(v1, v2, atol=1e-12)


def test_ols_less_shrunk_than_yule_walker_in_short_windows():
    """In 25-sample windows, the Yule-Walker self-term of a smooth node is shrunk; OLS is not."""
    rng = np.random.default_rng(6)
    coupling = build_coupling([0.9], [0.3], 2.5, [], 2)
    x = simulate_var(coupling, 25 * 400, rng)
    windows = x.reshape(1, 400, 25).transpose(0, 2, 1)
    windows = windows - windows.mean(axis=1, keepdims=True)
    a_ols, _ = ols_coeff(windows, 2)
    from src.mtmvar import ar_coeff
    a_yw, _ = ar_coeff(windows, 2)
    true = coupling[0, 0, 0]
    assert abs(a_ols[0, 0, 0] - true) < abs(a_yw[0, 0, 0] - true)


def test_true_conditional_gc_of_a_white_source():
    """x0_t = g x1_{t-1} + e0_t with white x1: without x1 the prediction-error variance is 1 + g^2, so F = ln(1 + g^2)."""
    coupling = np.zeros((2, 2, 1))
    coupling[0, 1, 0] = 0.5
    gc, improvement = true_conditional_gc(coupling, 1, 0)
    assert np.isclose(gc, np.log(1.25), atol=1e-10)
    assert np.isclose(improvement, 0.2, atol=1e-10)
    assert abs(true_conditional_gc(coupling, 0, 1)[0]) < 1e-10


def test_regressor_diagnostics_flags_a_near_duplicate_node():
    """Independent white nodes: VIF ~ 1 and a small condition number; a near-copy of the scored node inflates both."""
    from src.sim_estimation import regressor_diagnostics
    rng = np.random.default_rng(0)
    design = rng.normal(size=(5, 300))
    clean = regressor_diagnostics(design, 4, 25, 12, 3)
    design[4] = design[3] + 0.1 * rng.normal(size=300)
    collinear = regressor_diagnostics(design, 4, 25, 12, 3)
    assert clean["vif_max"] < 1.5 and clean["condition_number"] < 3
    assert collinear["vif_max"] > 50 and collinear["condition_number"] > 10
