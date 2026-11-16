"""The real-data pipeline estimates connectivity on the same code path as the simulations.

`src.connectivity.Granger_estimator` (stage 04), `src.surrogate.pair_tables` (stage 05) and
`src.sim_estimation.edge_matrices` must give the band-averaged edge matrix of `src.sim_estimation.edge_matrix`
(the simulations' estimator) in the manuscript setting: OLS on stacked 10 s windows, no detrending, p = 4,
Box-Cox lambda 0.25, coupling band 0.15-0.5 Hz, at 2.5 Hz.
"""

import numpy as np
import pytest
import xarray as xr

from src.connectivity import Granger_estimator
from src.design import assemble_design_matrix, detrend_windows, window_geometry, window_stack
from src.mtmvar import dtf_estimator, multivariate_spectra
from src.sim_estimation import edge_matrices, edge_matrix
from src.surrogate import band_average_cube, pair_tables, windowed_ar_stability

FS = 2.5
FREQS = np.linspace(0.02, FS / 2 - 0.02, 100)
BAND = (0.15, 0.5)
P, LAMBDA = 4, 0.25
WIN_LEN, STEP = window_geometry(10.0, 0.5, FS)
NODES = [{"name": "child:ROI", "role": "child"}, {"name": "cg:ROI", "role": "caregiver"},
         {"name": "child:HRV", "role": "child"}, {"name": "cg:HRV", "role": "caregiver"}]
NAMES = [n["name"] for n in NODES]


def random_design(seed, n=150):
    """z-scored 4 x n design with a lag-1 coupling cg:ROI -> child:ROI."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, n))
    for t in range(1, n):
        x[0, t] += 0.5 * x[1, t - 1] + 0.3 * x[0, t - 1]
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


@pytest.mark.parametrize("estimator", ["GPDC", "dDTF"])
def test_pipeline_edge_matrix_equals_simulation(estimator):
    """Stage 04/05 band-averaged matrix == `edge_matrix` to 1e-12 (manuscript setting)."""
    design = random_design(0)
    reference, root, _ = edge_matrix(design, P, WIN_LEN, STEP, FREQS, FS, estimator, BAND, LAMBDA, "none", method="ols")
    cube, _ = Granger_estimator(design, FREQS, FS, P, WIN_LEN, STEP, "none", estimator, LAMBDA, method="ols")
    np.testing.assert_allclose(band_average_cube(cube, FREQS, BAND), reference, rtol=0, atol=1e-12)
    matrices, root_2 = edge_matrices(design, P, WIN_LEN, STEP, FREQS, FS, [estimator], BAND, LAMBDA, "none", "ols")
    np.testing.assert_allclose(matrices[estimator], reference, rtol=0, atol=1e-12)
    assert root_2 == root == windowed_ar_stability(design, WIN_LEN, STEP, P, "none", "ols")[0]


def test_pair_tables_diagonal_and_pseudo_dyads_equal_simulation():
    """`pair_tables`: diagonal = real dyads, cell (b, c) = child rows of b with caregiver rows of c, both via `edge_matrix`."""
    designs = [random_design(seed) for seed in (1, 2, 3)]
    envelopes = [xr.DataArray(d, dims=("variable", "time"), coords={"variable": NAMES, "time": np.arange(150) / FS},
                              attrs={"fs": FS}) for d in designs]
    tables, roots = pair_tables(envelopes, NODES, P, WIN_LEN, STEP, FREQS, FS, ["GPDC", "dDTF"], BAND, LAMBDA,
                                "none", "ols", 1.0)
    for est in ("GPDC", "dDTF"):
        for b in range(3):
            for c in range(3):
                rows = [designs[b][r] if NODES[r]["role"] == "child" else designs[c][r] for r in range(4)]
                design = assemble_design_matrix(xr.DataArray(np.stack(rows), dims=("variable", "time"),
                                                             coords={"variable": NAMES}), NAMES, zscore=True)
                reference, root, _ = edge_matrix(design, P, WIN_LEN, STEP, FREQS, FS, est, BAND, LAMBDA, "none", method="ols")
                np.testing.assert_allclose(tables[est][b, c], reference, rtol=0, atol=1e-12)
                assert roots[b, c] == root


@pytest.mark.parametrize("estimator", ["GPDC", "dDTF", "ffDTF"])
def test_legacy_yule_walker_path_unchanged(estimator):
    """With its legacy defaults (YW, linear detrending) `Granger_estimator` equals the old `src.mtmvar` path."""
    design = random_design(4)
    stack = detrend_windows(window_stack(design, WIN_LEN, STEP), dtype="linear")
    old = dtf_estimator(stack, FREQS, FS, optimal_model_order=P, ESTIMATOR=estimator)
    old_spectra = multivariate_spectra(stack, FREQS, FS, optimal_model_order=P)
    cube, spectra = Granger_estimator(design, FREQS, FS, P, WIN_LEN, STEP, "linear", estimator)
    np.testing.assert_allclose(cube, old, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(spectra, old_spectra, rtol=1e-9, atol=1e-12)


def test_detrend_none_is_identity():
    """``detrend_windows(..., 'none')`` returns the windows as cut (mirrors `fit_mvar`)."""
    stack = window_stack(random_design(5), WIN_LEN, STEP)
    assert detrend_windows(stack, "none") is stack
