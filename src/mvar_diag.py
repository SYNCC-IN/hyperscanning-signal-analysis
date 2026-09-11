"""MVAR model-order selection and fit-quality diagnostics (Stage 3).

Stage 3's default estimation path is the **windowed-ACF-averaged** fit: a
design matrix is cut into short windows (`src.design.window_stack`,
`detrend_windows`) stacked on a trials axis, and `src.mtmvar.ar_coeff`'s
existing `count_corr`-based averaging (over that trials axis) turns them into
one MVAR estimate -- the Kaminski sDTF core. `fit_mvar_avg_acf` is a thin,
documented wrapper over that path; `residual_whiteness` and `ar_root_stability`
are the two diagnostics that ask whether the fitted VAR is trustworthy. No
ffDTF here (Stage 4); this module only asks "is the fit well-behaved", not
"what does it say about coupling".

`residual_whiteness`/`ar_root_stability` work identically on a 3-D windowed
stack or on a plain `(k, n_samples)` design reshaped to one window
(`design[:, :, np.newaxis]`) -- the same functions serve both the windowed fit
and a global single-window comparison fit, which is what the Stage 3 gate
uses to show the windowed method's improvement.

All functions preserve whatever channel order the caller passes in -- see
`scripts/stage03_mvar_order.py` for how the fixed `src.design.DESIGN_VARIABLES`
order is threaded through.
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

try:
    from .mtmvar import ar_coeff, mvar_criterion
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.mtmvar import ar_coeff, mvar_criterion


def fit_mvar_avg_acf(design_3d, p):
    """Fit one MVAR from the autocovariance averaged across windows.

    Thin wrapper over `src.mtmvar.ar_coeff`: the averaging across the trials
    axis (here, windows) happens inside `src.mtmvar.count_corr`, which
    `ar_coeff` already calls when given 3-D input -- this function only
    asserts the expected shape and documents the averaged-ACF semantics that
    make it correct to call this way (the Kaminski sDTF core).

    Parameters
    ----------
    design_3d : np.ndarray, shape (k, win_len, n_windows)
        Windowed, per-window-detrended design matrix
        (``detrend_windows(window_stack(design, win_len, step))``).
    p : int
        Model order.

    Returns
    -------
    ar_coeffs : np.ndarray, shape (k, k, p)
        ``ar_coeffs[:, :, m]`` is the lag-``(m + 1)`` coefficient matrix
        (row = target, column = source).
    variance : np.ndarray, shape (k, k)
        Residual covariance matrix, averaged across windows.
    """
    if design_3d.ndim != 3:
        raise ValueError(f"design_3d must be 3-D (k, win_len, n_windows), got shape {design_3d.shape}")
    return ar_coeff(design_3d, p)


def _windowed_residuals(design_3d, ar_coeffs):
    """One-step-ahead residuals computed independently within each window (no cross-window prediction)."""
    k, win_len, n_windows = design_3d.shape
    p = ar_coeffs.shape[2]

    residuals = np.zeros((k, win_len - p, n_windows))
    for window in range(n_windows):
        for t in range(p, win_len):
            predicted = np.zeros(k)
            for lag in range(1, p + 1):
                predicted += ar_coeffs[:, :, lag - 1] @ design_3d[:, t - lag, window]
            residuals[:, t - p, window] = design_3d[:, t, window] - predicted
    return residuals


def residual_whiteness(design_3d, ar_coeffs, max_lag):
    """Per-variable residual autocorrelation, averaged across windows.

    Reconstructs one-step-ahead residuals independently within each window
    (never predicting across a window boundary), computes each window's ACF
    up to `max_lag`, and averages the ACF curves across windows -- the same
    averaging philosophy as the fit itself, rather than concatenating windows
    into one pseudo-series (which would fabricate boundary autocorrelation).

    Parameters
    ----------
    design_3d : np.ndarray, shape (k, win_len, n_windows)
        The same (detrended, windowed) design matrix `ar_coeffs` was fit on
        (via `fit_mvar_avg_acf`) -- or a global design reshaped to one window
        (``design[:, :, np.newaxis]``) for a single-window comparison fit.
    ar_coeffs : np.ndarray, shape (k, k, p)
        AR coefficient tensor from `fit_mvar_avg_acf` (or `src.mtmvar.ar_coeff`).
    max_lag : int
        Number of lags to compute/average (must be well below
        ``win_len - p`` for the per-window ACF to be meaningful).

    Returns
    -------
    residual_acf : np.ndarray, shape (k, max_lag + 1)
        Per-variable ACF (lag 0 to `max_lag`), averaged across windows.
    whiteness_summary : dict
        ``{channel_index: fraction_of_lags_within_band}`` -- for each
        variable, the fraction of lags 1..max_lag whose averaged ACF falls
        within the ``+/- 1.96 / sqrt(n_eff)`` Bartlett white-noise band
        (``n_eff = win_len - p``). 1.0 = fully white by this summary.
    """
    residuals = _windowed_residuals(design_3d, ar_coeffs)
    n_channels, n_eff, n_windows = residuals.shape

    acf_per_window = np.stack([
        np.stack([acf(residuals[channel, :, window], nlags=max_lag, fft=False) for window in range(n_windows)])
        for channel in range(n_channels)
    ])  # (n_channels, n_windows, max_lag + 1)
    residual_acf = acf_per_window.mean(axis=1)

    bartlett_band = 1.96 / np.sqrt(n_eff)
    within_band = np.abs(residual_acf[:, 1:]) <= bartlett_band
    whiteness_summary = {channel: float(within_band[channel].mean()) for channel in range(n_channels)}

    return residual_acf, whiteness_summary


def ar_root_stability(ar_coeffs):
    """Companion-matrix eigenvalues and stability of a fitted AR coefficient tensor.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (k, k, p)
        AR coefficient tensor from `fit_mvar_avg_acf` (or `src.mtmvar.ar_coeff`).

    Returns
    -------
    roots : np.ndarray, shape (k * p,), complex
        Eigenvalues of the ``(k*p, k*p)`` companion matrix (top block-row
        ``[A_1 ... A_p]``, sub-diagonal identity blocks) -- for plotting on
        the complex unit circle.
    max_abs_root : float
        The largest eigenvalue modulus (the headline stability number).
    is_stable : bool
        Whether every eigenvalue modulus is strictly below 1 (inside the unit
        circle).
    """
    k, _, p = ar_coeffs.shape
    top_row = np.concatenate([ar_coeffs[:, :, lag] for lag in range(p)], axis=1)

    if p > 1:
        sub_identity = np.eye(k * (p - 1))
        sub_zeros = np.zeros((k * (p - 1), k))
        bottom_rows = np.concatenate([sub_identity, sub_zeros], axis=1)
        companion = np.vstack([top_row, bottom_rows])
    else:
        companion = top_row

    roots = np.linalg.eigvals(companion)
    max_abs_root = float(np.abs(roots).max())
    return roots, max_abs_root, bool(max_abs_root < 1.0)


def select_order(system, max_model_order, crit_types):
    """Select the MVAR model order under each of several information criteria.

    Thin wrapper over `src.mtmvar.mvar_criterion`, run once per criterion.
    `mvar_criterion` only accepts a 2-D `(m, n_samples)` array, so this always
    runs on the global (non-windowed) design -- see the Stage 3 script's
    documented caveat on why order selection stays 2-D.

    Parameters
    ----------
    system : np.ndarray, shape (m, n_samples)
        Any 2-D design matrix -- the full multivariate system, or a sub-block
        (e.g. just the EEG or just the HRV rows).
    max_model_order : int
        Maximum model order to evaluate.
    crit_types : list of str
        Criteria to evaluate, each one of ``'AIC'``, ``'HQ'``, ``'SC'``.

    Returns
    -------
    optimal_orders : dict
        ``{crit_type: optimal_model_order}``.
    curves : dict
        ``{crit_type: criterion values array, length max_model_order}``.
    model_order_range : np.ndarray
        The evaluated order range (``1..max_model_order``), shared by every
        criterion's curve.
    """
    optimal_orders = {}
    curves = {}
    model_order_range = None
    for crit_type in crit_types:
        crit, model_order_range, optimal_model_order = mvar_criterion(
            system, max_model_order, crit_type=crit_type, plot=False,
        )
        optimal_orders[crit_type] = int(optimal_model_order)
        curves[crit_type] = crit
    return optimal_orders, curves, model_order_range


def select_p_used(design, max_model_order, crit_types, primary_crit, eeg_rows, hrv_rows):
    """Select the shared model order for the joint system, plus diagnostic sub-block orders.

    `p_used` is fit on the joint 4-variable system (required for exploratory
    cross-block edges, which only exist in the joint model); `p_eeg`/`p_hrv`
    are reported only to expose an EEG/HRV order mismatch, not to justify
    splitting the fit.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        Global (non-windowed) z-scored design matrix.
    max_model_order, crit_types, primary_crit : see `select_order`.
    eeg_rows, hrv_rows : list of int
        Row indices for the EEG-only and HRV-only sub-blocks.

    Returns
    -------
    dict
        ``{p_used, p_eeg, p_hrv, orders_full, orders_eeg, orders_hrv,
        curves_full, curves_eeg, curves_hrv, order_range, order_at_cap}``.
    """
    orders_full, curves_full, order_range = select_order(design, max_model_order, crit_types)
    orders_eeg, curves_eeg, _ = select_order(design[eeg_rows], max_model_order, crit_types)
    orders_hrv, curves_hrv, _ = select_order(design[hrv_rows], max_model_order, crit_types)
    return {
        "p_used": orders_full[primary_crit], "p_eeg": orders_eeg[primary_crit], "p_hrv": orders_hrv[primary_crit],
        "orders_full": orders_full, "orders_eeg": orders_eeg, "orders_hrv": orders_hrv,
        "curves_full": curves_full, "curves_eeg": curves_eeg, "curves_hrv": curves_hrv,
        "order_range": order_range, "order_at_cap": any(o == max_model_order for o in orders_full.values()),
    }


def plot_order_curves(order_range, curves_by_block, orders_by_block, max_model_order, title):
    """Plot AIC/HQ/SC criterion curves for the full system and each sub-block."""
    figure, axes = plt.subplots(ncols=len(curves_by_block), figsize=(4 * len(curves_by_block), 4))
    for axis, block_label in zip(axes, curves_by_block):
        for crit_type, curve in curves_by_block[block_label].items():
            line, = axis.plot(order_range, curve, label=crit_type)
            axis.axvline(orders_by_block[block_label][crit_type], color=line.get_color(), linestyle=":", alpha=0.6)
        axis.axvline(max_model_order, color="black", linestyle="--", label="cap")
        axis.set_title(block_label)
        axis.set_xlabel("model order p")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("criterion value")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_model_order_histogram(manifest_df):
    """Grouped bar chart of model orders (`p_used`) used, by group.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Manifest with `p_used` and `group` columns.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered histogram figure.
    """
    order_counts = manifest_df.groupby(["p_used", "group"]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(5, 3.5))
    order_counts.plot(kind="bar", ax=axis)
    axis.set_xlabel("model order p_used")
    axis.set_ylabel("n cases")
    axis.set_title("Model orders used (from Stage 3)")
    axis.legend(title="group")
    figure.tight_layout()
    return figure


def plot_roots_comparison(roots_global, roots_windowed, max_abs_root_global, max_abs_root_windowed, title):
    """Plot AR companion eigenvalues for the global vs windowed fit on one unit circle."""
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    axis.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    axis.scatter(roots_global.real, roots_global.imag, color="steelblue",
                 label=f"global (max={max_abs_root_global:.3f})", zorder=3)
    axis.scatter(roots_windowed.real, roots_windowed.imag, color="crimson", marker="x",
                 label=f"windowed (max={max_abs_root_windowed:.3f})", zorder=4)
    axis.set_xlabel("Re")
    axis.set_ylabel("Im")
    axis.set_aspect("equal")
    axis.legend(fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_acf_comparison(acf_global, acf_windowed, band_global, band_windowed, variable_names, title):
    """Plot pooled residual ACF, global vs windowed fit, one panel per variable."""
    figure, axes = plt.subplots(ncols=len(variable_names), figsize=(3.2 * len(variable_names), 3), sharey=True)
    lags = np.arange(acf_global.shape[1])
    for channel, (axis, name) in enumerate(zip(axes, variable_names)):
        width = 0.35
        axis.bar(lags[1:] - width / 2, acf_global[channel, 1:], width=width, color="steelblue", label="global")
        axis.bar(lags[1:] + width / 2, acf_windowed[channel, 1:], width=width, color="crimson", label="windowed")
        axis.axhline(band_global, color="steelblue", linestyle="--", linewidth=0.8)
        axis.axhline(-band_global, color="steelblue", linestyle="--", linewidth=0.8)
        axis.axhline(band_windowed, color="crimson", linestyle="--", linewidth=0.8)
        axis.axhline(-band_windowed, color="crimson", linestyle="--", linewidth=0.8)
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("lag")
    axes[0].set_ylabel("residual ACF")
    axes[0].legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_detrend_example(stack, stack_detrended, variable_names, win_len_s, coupling_band_hz, title, n_examples=3):
    """Plot a few example windows, pre- vs post-detrend, one row per variable."""
    figure, axes = plt.subplots(nrows=len(variable_names), figsize=(8, 2.2 * len(variable_names)), sharex=True)
    for channel, (axis, name) in enumerate(zip(axes, variable_names)):
        for window in range(min(n_examples, stack.shape[2])):
            offset = window * stack.shape[1]
            time = offset + np.arange(stack.shape[1])
            axis.plot(time, stack[channel, :, window], color="steelblue", alpha=0.6,
                      label="raw" if window == 0 else None)
            axis.plot(time, stack_detrended[channel, :, window], color="crimson", alpha=0.8,
                      label="detrended" if window == 0 else None)
        axis.set_ylabel(name, fontsize=9)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("sample (example windows concatenated for display)")
    figure.suptitle(
        f"{title}\nlinear detrend attenuates below ~1/win_len = {1 / win_len_s:.2f} Hz "
        f"(coupling band {coupling_band_hz[0]}-{coupling_band_hz[1]} Hz)"
    )
    figure.tight_layout()
    return figure
