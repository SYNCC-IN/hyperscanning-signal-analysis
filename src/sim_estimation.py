"""Fast, vectorized connectivity estimation for simulation studies.

The simulations need tens of thousands of MVAR fits (every replication fits one
model per surrogate pair), so the frequency-domain quantities are computed here
from ONE fit, with batched linear algebra, instead of the three separate fits
per call of `src.mtmvar.direct_dtf`. The MVAR fit itself is the repository's own
(`src.mtmvar.ar_coeff` on windows from `src.design`), and the formulas are the
repository's (checked against `src.connectivity.Granger_estimator` in
`tests/test_sim_estimation.py`). Plain DTF (normalized per frequency), which the
repository does not expose, is added for the estimator comparisons.
"""

import numpy as np

try:
    from .design import detrend_windows, window_stack
    from .mtmvar import ar_coeff, box_cox_transform, mvar_criterion, ols_coeff
    from .mvar_diag import ar_root_stability
    from .surrogate import band_average_cube
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.design import detrend_windows, window_stack
    from src.mtmvar import ar_coeff, box_cox_transform, mvar_criterion, ols_coeff
    from src.mvar_diag import ar_root_stability
    from src.surrogate import band_average_cube


def freq_matrices(ar_coeffs, freqs, fs):
    """Frequency-domain AR matrix ``A(f) = I - sum_r A(r) exp(-i 2 pi f r / fs)``.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (k, k, p)
    freqs : np.ndarray, shape (n_freqs,)
    fs : float

    Returns
    -------
    np.ndarray, shape (n_freqs, k, k), complex
    """
    k, _, p = ar_coeffs.shape
    z = np.exp(-2j * np.pi * np.outer(freqs, np.arange(1, p + 1)) / fs)
    return np.eye(k)[None, :, :] - np.einsum("ijm,fm->fij", ar_coeffs, z)


def connectivity_cube(ar_coeffs, variance, freqs, fs, estimator):
    """Directed connectivity cube ``[target, source, freq]`` from one fitted MVAR model.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (k, k, p)
    variance : np.ndarray, shape (k, k)
        Residual covariance.
    freqs : np.ndarray
    fs : float
    estimator : {'DTF', 'ffDTF', 'dDTF', 'GPDC'}
        'DTF' is normalized per frequency (Eq. 6 of the paper); 'ffDTF' and 'dDTF' follow
        `src.mtmvar.full_freq_dtf` / `direct_dtf` (dDTF = ffDTF * |partial coherence|^2); 'GPDC' follows `gen_partial_directed_coherence`.

    Returns
    -------
    np.ndarray, shape (k, k, n_freqs)
    """
    a_mat = freq_matrices(ar_coeffs, freqs, fs)
    if estimator == "GPDC":
        sigma2 = np.diag(variance)
        numerator = np.abs(a_mat) / np.sqrt(sigma2)[None, :, None]
        denominator = np.sqrt((np.abs(a_mat) ** 2 / sigma2[None, :, None]).sum(axis=1))
        return (numerator / denominator[:, None, :]).transpose(1, 2, 0)
    h2 = np.abs(np.linalg.inv(a_mat)) ** 2
    if estimator == "DTF":
        cube = h2 / h2.sum(axis=2, keepdims=True)
    else:
        cube = h2 / h2.sum(axis=(0, 2))[None, :, None]
        if estimator == "dDTF":
            s_inv = np.conj(a_mat.transpose(0, 2, 1)) @ np.linalg.inv(variance) @ a_mat
            diag = np.real(np.einsum("fii->fi", s_inv))
            # chi^2 = F^2 * |kappa|^2 (Korzeniewska et al., 2003; squared DTF convention)
            cube = cube * (np.abs(s_inv) ** 2 / (diag[:, :, None] * diag[:, None, :]))
        elif estimator != "ffDTF":
            raise ValueError(f"Unknown estimator: {estimator}")
    return cube.transpose(1, 2, 0)


def fit_mvar(design, p, win_len, step, detrend_type="linear", method="ols"):
    """Fit one MVAR model, from window-averaged autocovariance or from the whole segment.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix.
    p : int
        Model order.
    win_len, step : int or None
        Window length / step in samples. ``win_len=None`` fits the whole segment as one
        un-detrended window (the global fit).
    detrend_type : {'linear', 'constant'}, optional
    method : {'ols', 'yw'}, optional
        'ols' (default): least squares on the stacked windows (`src.mtmvar.ols_coeff`), no
        edge-truncation bias. 'yw': Yule-Walker on the window-averaged biased autocovariance
        (`src.mtmvar.ar_coeff`), the original choice; its coefficients are shrunk in short windows.

    Returns
    -------
    ar_coeffs : np.ndarray, shape (k, k, p)
    variance : np.ndarray, shape (k, k)
    max_abs_root : float
        Largest companion-matrix eigenvalue modulus (< 1 = stable).
    """
    if win_len is None:
        stack = design
    else:
        stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    if method == "ols":
        coeffs, variance = ols_coeff(stack, p)
    elif method == "yw":
        coeffs, variance = ar_coeff(stack, p)
    else:
        raise ValueError(f"Unknown fitting method: {method}")
    _, max_abs_root, _ = ar_root_stability(coeffs)
    return coeffs, variance, max_abs_root


def select_order(design, p_max, crit_type):
    """Model order minimizing an information criterion, on the global 2-D signal.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
    p_max : int
    crit_type : {'AIC', 'HQ', 'SC'}

    Returns
    -------
    int
    """
    return int(mvar_criterion(design, p_max, crit_type, plot=False)[2])


def edge_matrix(design, p, win_len, step, freqs, fs, estimator, band_hz,
                box_cox_lambda=-1, detrend_type="linear", p_max=8, crit_type="SC", method="ols"):
    """Band-averaged connectivity matrix ``[target, source]`` of one design.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
    p : int or None
        Model order; None selects it per fit by `crit_type` up to `p_max`.
    win_len, step : int or None
        See `fit_mvar`.
    freqs : np.ndarray
    fs : float
    estimator : {'DTF', 'ffDTF', 'dDTF', 'GPDC'} or callable
        A callable maps the design (k, n) to a matrix [target, source] and bypasses the MVAR (competitors in S11).
    band_hz : tuple of float
        Coupling band (low, high) in Hz.
    box_cox_lambda : float, optional
        Box-Cox exponent applied before band-averaging (-1 = none).
    detrend_type : {'linear', 'constant'}, optional
    p_max : int, optional
        Largest order considered when `p` is None.
    crit_type : {'AIC', 'HQ', 'SC'}, optional
    method : {'ols', 'yw'}, optional
        MVAR fitting method, see `fit_mvar`.

    Returns
    -------
    matrix : np.ndarray, shape (k, k)
    max_abs_root : float
    p_used : int
    """
    if callable(estimator):               # competitor statistic (S11): estimator(design) -> matrix[target, source]
        return np.asarray(estimator(design), dtype=float), 0.0, 0
    p_used = p if p is not None else select_order(design, p_max, crit_type)
    coeffs, variance, max_abs_root = fit_mvar(design, p_used, win_len, step, detrend_type, method)
    cube = box_cox_transform(connectivity_cube(coeffs, variance, freqs, fs, estimator), box_cox_lambda)
    return band_average_cube(cube, freqs, band_hz), max_abs_root, p_used
