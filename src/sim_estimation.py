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
from scipy.linalg import solve_discrete_lyapunov

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


def fit_mvar(design, p, win_len, step, detrend_type="none", method="ols"):
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
    detrend_type : {'none', 'linear', 'constant'}, optional
        Per-window detrending. 'none' (default since 2026-11-13) fits the windows as cut; 'linear' (the earlier
        default) biases conditional estimates of slow, collinear nodes (check A4, REPORT_checks_2026-11-13.md).
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
    elif detrend_type == "none":
        stack = window_stack(design, win_len, step)
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
                box_cox_lambda=-1, detrend_type="none", p_max=8, crit_type="SC", method="ols"):
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
    detrend_type : {'none', 'linear', 'constant'}, optional
        See `fit_mvar` (default 'none').
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



def regressor_diagnostics(design, p, win_len, step, node, detrend_type="none"):
    """Collinearity of the lagged regressors of the least-squares MVAR fit on stacked windows.

    The regressor matrix is the one `src.mtmvar.ols_coeff` solves: for every window (from `src.design`), the rows
    ``[x(t-1), ..., x(t-p)]`` for t = p, ..., n-1, stacked over windows. It is the same in every equation of the
    model, so the VIF of a regressor does not depend on the target equation.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
    p : int
        Model order.
    win_len, step : int
        Window length / step in samples.
    node : int
        Row of `design` whose lags are scored by the VIF (e.g. the caregiver heart).
    detrend_type : {'none', 'linear', 'constant'}, optional
        As in `fit_mvar` (default 'none').

    Returns
    -------
    dict
        condition_number (2-norm condition number of the regressor matrix), vif_max (largest variance inflation
        factor ``1 / (1 - R^2)`` over the p lags of `node`, from the inverse correlation matrix of the regressors).
    """
    stack = window_stack(design, win_len, step)
    if detrend_type != "none":
        stack = detrend_windows(stack, dtype=detrend_type)
    k, n, n_windows = stack.shape
    x = np.vstack([np.hstack([stack[:, p - lag:n - lag, w].T for lag in range(1, p + 1)]) for w in range(n_windows)])
    vif = np.diag(np.linalg.inv(np.corrcoef(x, rowvar=False)))
    return {"condition_number": float(np.linalg.cond(x)), "vif_max": float(vif[[lag * k + node for lag in range(p)]].max())}

def var_autocovariance(coupling, noise_cov, n_lags):
    """Exact autocovariances Gamma(0..n_lags) of a stable VAR process.

    Parameters
    ----------
    coupling : np.ndarray, shape (k, k, p)
        ``coupling[target, source, lag - 1]`` (the convention of `src.sim_generators.simulate_var`).
    noise_cov : np.ndarray, shape (k, k)
        Innovation covariance.
    n_lags : int
        Largest lag returned.

    Returns
    -------
    np.ndarray, shape (n_lags + 1, k, k)
        ``gamma[h] = E[x_t x_{t-h}^T]``.
    """
    assert ar_root_stability(coupling)[2], "coupling is unstable: the process has no stationary autocovariance"
    k, _, p = coupling.shape
    companion = np.zeros((k * p, k * p))
    companion[:k] = np.concatenate([coupling[:, :, lag] for lag in range(p)], axis=1)
    companion[k:, :-k] = np.eye(k * (p - 1))
    q = np.zeros((k * p, k * p))
    q[:k, :k] = noise_cov
    state_cov = solve_discrete_lyapunov(companion, q)
    gamma = np.zeros((max(n_lags, p - 1) + 1, k, k))
    for h in range(p):
        gamma[h] = state_cov[:k, h * k:(h + 1) * k]
    for h in range(p, n_lags + 1):
        gamma[h] = sum(coupling[:, :, r - 1] @ (gamma[h - r] if h >= r else gamma[r - h].T) for r in range(1, p + 1))
    return gamma[:n_lags + 1]


def prediction_error_cov(gamma, order):
    """Innovation covariance of the order-`order` linear predictor of a process with autocovariances `gamma`.

    Solves the multivariate Yule-Walker equations on exact autocovariances, so the result has no sampling error.

    Parameters
    ----------
    gamma : np.ndarray, shape (>= order + 1, m, m)
        ``gamma[h] = E[x_t x_{t-h}^T]``.
    order : int

    Returns
    -------
    np.ndarray, shape (m, m)
    """
    m = gamma.shape[1]
    toeplitz = np.block([[gamma[j - i] if j >= i else gamma[i - j].T for j in range(order)] for i in range(order)])
    rhs = np.concatenate([gamma[h] for h in range(1, order + 1)], axis=1)
    coeffs = np.linalg.solve(toeplitz.T, rhs.T).T                    # [A_1 ... A_order], from A R = rhs
    return gamma[0] - coeffs @ rhs.T


def true_conditional_gc(coupling, source, target, noise_var=1.0, order=60):
    """Population conditional Granger causality F of `source` -> `target` and the prediction improvement 1 - exp(-F).

    F = ln(sigma2_restricted / sigma2_full) (Geweke, 1984): sigma2_full is the target's innovation variance in the
    true model, sigma2_restricted its one-step prediction-error variance when the past of `source` is removed and
    the past of all other nodes is kept. The restricted predictor is a VAR(`order`) on the exact autocovariances of
    the remaining nodes (the restricted process is not a finite VAR; `order` controls the truncation).

    Parameters
    ----------
    coupling : np.ndarray, shape (k, k, p)
        True coefficient tensor ``coupling[target, source, lag - 1]``.
    source, target : int
    noise_var : float, optional
        Innovation variance of every node (independent innovations).
    order : int, optional

    Returns
    -------
    gc : float
    prediction_improvement : float
    """
    k = coupling.shape[0]
    gamma = var_autocovariance(coupling, noise_var * np.eye(k), order)
    keep = [node for node in range(k) if node != source]
    restricted = prediction_error_cov(gamma[:, keep][:, :, keep], order)[keep.index(target), keep.index(target)]
    gc = float(np.log(restricted / noise_var))
    return gc, float(1.0 - np.exp(-gc))


def conditional_gc(design, source, target, p, win_len, step, detrend_type="none"):
    """Estimated conditional Granger causality F of `source` -> `target` from a least-squares MVAR fit.

    ``F = ln(sigma2_restricted / sigma2_full)``: sigma2_full is the residual variance of the `target` equation of the
    full model (the regression `src.mtmvar.ols_coeff` solves: lags 1..p of every node, rows stacked over windows),
    sigma2_restricted that of the same equation without the lags of `source`. Population value: `true_conditional_gc`.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
    source, target : int
        Rows of `design`.
    p : int
        Model order.
    win_len, step : int or None
        Window length / step in samples; ``win_len=None`` uses the whole segment as one window, not detrended.
    detrend_type : {'none', 'linear', 'constant'}, optional
        As in `fit_mvar` (default 'none').

    Returns
    -------
    float
    """
    if win_len is None:
        stack = design[:, :, None]
    elif detrend_type == "none":
        stack = window_stack(design, win_len, step)
    else:
        stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    k, n, n_windows = stack.shape
    x = np.vstack([np.hstack([stack[:, p - lag:n - lag, w].T for lag in range(1, p + 1)]) for w in range(n_windows)])
    y = np.concatenate([stack[target, p:, w] for w in range(n_windows)])
    keep = [col for col in range(k * p) if col % k != source]

    def residual_variance(columns):
        coef, *_ = np.linalg.lstsq(columns, y, rcond=None)
        return np.mean((y - columns @ coef) ** 2)

    return float(np.log(residual_variance(x[:, keep]) / residual_variance(x)))
