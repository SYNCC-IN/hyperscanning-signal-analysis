"""Directed-connectivity estimation for the real-data pipeline (dDTF, GPDC, ...).

One code path with the simulations: the MVAR fit is `src.sim_estimation.fit_mvar` (least squares on the stacked
windows, ``method='ols'``, windows fitted as cut, ``detrend_type='none'``, in the manuscript setting) and the
estimator cube is `src.sim_estimation.connectivity_cube`. `Granger_estimator` keeps its historical name and
signature for older callers; its `detrend_type` / `method` defaults ('linear', 'yw') reproduce the original
Yule-Walker path, which `simulations_multimodal_MVAR/dDTF/S01_recovery_orientation.py` still uses as a reference.
The pipeline always passes `DETREND_TYPE` and `FIT_METHOD` from its config.
"""

try:
    from .mtmvar import box_cox_transform
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.mtmvar import box_cox_transform


def Granger_estimator(design, freqs, fs, p, win_len, step, detrend_type="linear", ESTIMATOR="dDTF", box_cox_lambda=-1,
                      method="yw"):
    """Connectivity cube and spectral matrix of one design from one windowed MVAR fit.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix in config-driven node order (`src.design.assemble_design_matrix`).
    freqs : np.ndarray
        Frequency axis (Hz).
    fs : float
        Sampling frequency (Hz) of `design`.
    p : int
        MVAR model order.
    win_len, step : int
        Window length / step in samples.
    detrend_type : {'none', 'linear', 'constant'}, optional
        Per-window detrending (`src.sim_estimation.fit_mvar`). Default 'linear' only for older callers.
    ESTIMATOR : {'dDTF', 'GPDC', 'ffDTF', 'DTF'}, optional
        See `src.sim_estimation.connectivity_cube`.
    box_cox_lambda : float, optional
        Box-Cox exponent applied to the connectivity cube (-1 = no transform); spectra are never transformed.
    method : {'ols', 'yw'}, optional
        MVAR fitting method (`src.sim_estimation.fit_mvar`). Default 'yw' only for older callers.

    Returns
    -------
    cube : np.ndarray, shape (k, k, n_freqs)
        ``cube[target, source, f]`` = flow source -> target.
    spectra : np.ndarray, shape (k, k, n_freqs), complex
        Spectral matrix of the same fit (`src.sim_estimation.spectral_matrix`).
    """
    try:
        from .sim_estimation import connectivity_cube, fit_mvar, spectral_matrix  # deferred: circular import via src.surrogate
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from src.sim_estimation import connectivity_cube, fit_mvar, spectral_matrix
    assert win_len > p, f"win_len={win_len} must exceed model order p={p}"
    coeffs, variance, _ = fit_mvar(design, p, win_len, step, detrend_type, method)
    cube = box_cox_transform(connectivity_cube(coeffs, variance, freqs, fs, ESTIMATOR), box_cox_lambda)
    return cube, spectral_matrix(coeffs, variance, freqs, fs)


def read_edge_value(cube, source, target, names=None, orientation="target_source"):
    """Read one directed edge's value out of a connectivity cube/matrix.

    Handles both raw integer indexing (`names=None`) and named lookup
    (`names` is the ordered list `source`/`target` are drawn from, e.g.
    `src.design.node_names(nodes)`). `orientation` says which axis is target
    vs source: `"target_source"` (default) matches `Granger_estimator`'s
    `[target, source, freq]` convention (row = target/driven, column =
    source/driving); `"source_target"` is the reverse. A 3-D `cube`
    (frequency-resolved) is read as its mean across the trailing frequency
    axis; a plain 2-D matrix (already band- or frequency-averaged) is read
    as-is.

    Parameters
    ----------
    cube : np.ndarray, shape (k, k) or (k, k, n_freqs)
        Connectivity matrix or frequency-resolved cube.
    source, target : int or str
        Raw row/column indices (`names=None`) or entries of `names`.
    names : list of str, optional
        Ordered channel/variable names `cube`'s rows/columns correspond to.
        Default None (source/target are already integer indices).
    orientation : {"target_source", "source_target"}, optional
        Default "target_source".

    Returns
    -------
    float
        The edge value (frequency-averaged, for a 3-D `cube`).
    """
    if names is not None:
        source, target = names.index(source), names.index(target)
    row, col = (target, source) if orientation == "target_source" else (source, target)
    value = cube[row, col]
    return float(value.mean()) if cube.ndim == 3 else float(value)


def edge_value(design, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz, method="yw"):
    """Band-averaged connectivity value of one directed edge of a design (via `Granger_estimator`).

    Convenience wrapper combining `Granger_estimator` + `band_average_cube` +
    `read_edge_value` in one call, for callers that only need a single edge's
    band-averaged value (e.g. a synthetic-validation harness scoring one
    edge per simulated dyad).

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix.
    edge : tuple of int
        `(row, target)` raw indices into the band-averaged matrix
        (`orientation="target_source"`, i.e. `edge = (target, source)`, to
        match the historical caller convention).
    freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, method :
        Passed through to `Granger_estimator`.
    band_hz : tuple of float
        `(low, high)` band edges in Hz, passed to `src.surrogate.band_average_cube`.

    Returns
    -------
    float
        Band-averaged edge value.
    """
    try:
        from .surrogate import band_average_cube  # deferred: src.surrogate imports this module
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from src.surrogate import band_average_cube
    cube, _ = Granger_estimator(design, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, method)
    band_avg = band_average_cube(cube, freqs, band_hz)
    return float(band_avg[edge[0], edge[1]])
