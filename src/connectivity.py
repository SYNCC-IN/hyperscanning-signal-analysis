"""Granger_estimator connectivity estimation (Stage 4).

The Stage 4 estimator reuses Stage 3's windowed-ACF-averaged MVAR core
(`src.design.window_stack`/`detrend_windows` -> `src.mtmvar.full_freq_dtf`/
`multivariate_spectra` with an explicit `optimal_model_order`) rather than a
single global 2-D fit. Stage 3 established empirically that this averaged
estimator is more stable and whiter-residualed than a global fit, primarily
for the non-stationary HRV variables -- see
`DTF_analysis_notes/pipeline_plan.md` Stage 4 and `scripts/stage03_mvar_order.py`.

This is the default estimation path, not a deferred swap-in: `Granger_estimator`
is the stable interface Stage 4 (and later stages) call, so a future
Bayesian-MVAR core could replace the internals without changing callers.
"""

try:
    from .design import detrend_windows, window_stack
    from .mtmvar import dtf_estimator, multivariate_spectra
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.design import detrend_windows, window_stack
    from src.mtmvar import dtf_estimator, multivariate_spectra


def Granger_estimator(design, freqs, fs, p, win_len, step, detrend_type="linear", ESTIMATOR="dDTF", box_cox_lambda=-1):
    """Windowed-ACF-averaged Granger_estimator and multivariate spectra at a fixed order.

    Cuts the per-film design matrix into overlapping, per-window-detrended
    windows (the same geometry Stage 3 selected and recorded) and fits ONE
    MVAR from the block-autocovariance averaged across those windows -- the
    Kaminski sDTF core, which Stage 3 established as more stable and
    whiter-residualed than a single global fit, primarily on the
    non-stationary HRV variables. Granger_estimator and spectra are then read off that one
    averaged fit.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix in `src.design.DESIGN_VARIABLES` order
        (from `src.design.assemble_design_matrix`).
    freqs : np.ndarray
        Frequency axis (Hz) for the Granger_estimator/spectra cubes.
    fs : float
        Sampling frequency (Hz) of `design`.
    p : int
        MVAR model order (Stage 3's `p_used`; never None on the 3-D stack).
    win_len, step : int
        Window length / step in samples (Stage 3's `win_len` / `step`).
    detrend_type : {'linear', 'constant'}, optional
        Per-window detrend type (Stage 3's `detrend_type`).
    ESTIMATOR : {'dDTF', 'ffDTF', 'GPDC'}, optional
        Estimator type for the directed transfer function calculation (default is "dDTF").
    box_cox_lambda : float, optional
        Box-Cox exponent `(x**box_cox_lambda - 1) / box_cox_lambda` applied to the
        Granger_estimator cube (see `src.mtmvar.box_cox_transform`). Default -1 = no
        transform, preserving prior behaviour; spectra are never transformed.

    Returns
    -------
    granger_estimator : np.ndarray, shape (k, k, n_freqs)
        Granger_estimator cube. `granger_estimator[target, source, f]` = flow source -> target.
    spectra : np.ndarray, shape (k, k, n_freqs), complex
        Multivariate spectra on the same averaged fit.
    """
    assert win_len > p, f"win_len={win_len} must exceed model order p={p}"
    stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    granger_estimator = dtf_estimator(stack, freqs, fs, optimal_model_order=p, ESTIMATOR=ESTIMATOR, box_cox_lambda=box_cox_lambda)
    spectra = multivariate_spectra(stack, freqs, fs, optimal_model_order=p)
    return granger_estimator, spectra


def read_edge_value(cube, source, target, names=None, orientation="target_source"):
    """Read one directed edge's value out of a connectivity cube/matrix.

    Handles both raw integer indexing (`names=None`) and named lookup
    (`names` is the ordered list `source`/`target` are drawn from, e.g.
    `src.design.DESIGN_VARIABLES`). `orientation` says which axis is target
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


def edge_value(design, edge, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda, band_hz):
    """Band-averaged Granger_estimator value for one directed edge of a 2-channel design.

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
    freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda :
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
    granger_estimator, _ = Granger_estimator(design, freqs, fs, p, win_len, step, detrend_type, estimator, box_cox_lambda)
    band_avg = band_average_cube(granger_estimator, freqs, band_hz)
    return float(band_avg[edge[0], edge[1]])
