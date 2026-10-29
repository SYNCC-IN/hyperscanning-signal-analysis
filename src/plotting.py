"""Figures of the synchronisation pipeline (no computation beyond display scaling)."""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from .dsp import zscore


@dataclass
class QcParams:
    """Display settings of the alignment quality-control figures.

    Attributes
    ----------
    zoom_lag_ms : float
        One-sided lag range of the zoomed global cross-correlation panel.
    uncorrected_zoom_ms : float
        One-sided lag range of the peak-centred comparison of fitted and
        uncorrected alignment.
    zoom_window_s : float
        Length of each zoomed time-series panel.
    zoom_fractions : tuple of float
        Positions of the time-series zooms inside the EEG period (0 = start, 1 = end).
    dpi : int
        Resolution of the saved figures.
    """

    zoom_lag_ms: float
    uncorrected_zoom_ms: float
    zoom_window_s: float
    zoom_fractions: tuple
    dpi: int


@dataclass
class OverlayParams:
    """Display settings of the ECG overlay figure.

    Attributes
    ----------
    reference_name, comparison_name : str
        Device names used in labels.
    rolling_window_s : float
        Rolling-correlation window (for the axis label).
    min_correlation : float
        Correlation threshold drawn in the lower panel.
    max_lag_s : float
        Lag range of the rolling peak correlation (for the axis label).
    zoom_window_s : float
        Length of each zoomed trace.
    zoom_fractions : tuple of float
        Zoom positions inside the run (0 = start, 1 = end).
    dpi : int
        Resolution of the saved figure.
    """

    reference_name: str
    comparison_name: str
    rolling_window_s: float
    min_correlation: float
    max_lag_s: float
    zoom_window_s: float
    zoom_fractions: tuple
    dpi: int


def plot_offset_tracks(fits, output_path, dpi):
    """Plot tracked offsets and fitted lines of all devices.

    Parameters
    ----------
    fits : dict of str to clocksync.SyncFit
        Clock fit per device.
    output_path : str or Path
        PNG file to write.
    dpi : int
        Resolution of the saved figure.
    """
    fig, axes = plt.subplots(len(fits), 1, figsize=(8, 6), sharex=True, squeeze=False)
    for axis, (name, fit) in zip(axes[:, 0], fits.items()):
        tracks = fit.tracks
        used = fit.used_mask
        offset_ms = 1e3 * (tracks['offset'] - fit.intercept_s)
        fitted_ms = 1e3 * (np.polyval([fit.slope, fit.intercept_s], tracks['eeg_t']) - fit.intercept_s)
        axis.plot(tracks['eeg_t'][used], offset_ms[used], 'o', label='used')
        axis.plot(tracks['eeg_t'][~used], offset_ms[~used], 'rx', label='rejected')
        axis.plot(tracks['eeg_t'], fitted_ms, 'k-')
        axis.set_ylabel(f'{name}: offset-a (ms)')
        axis.legend()
    axes[-1, 0].set_xlabel('EEG time (s)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_ecg_overlay(grid_s, reference_ecg, run, params, output_path):
    """Plot zoomed overlays of both ECGs and the rolling correlation.

    Top row: short zooms spread across the "same wearer" run. Bottom:
    rolling correlation over the whole common span with the run highlighted.

    Parameters
    ----------
    grid_s : numpy.ndarray
        Common EEG-time grid.
    reference_ecg : numpy.ndarray
        Aligned reference ECG on ``grid_s``.
    run : ecg_compare.SharedWearerRun
        Run bounds, statistics, rolling correlation and the polarity-matched
        comparison ECG.
    params : OverlayParams
        Labels, thresholds and zoom settings.
    output_path : str or Path
        PNG file to write.
    """
    reference_z = zscore(reference_ecg)
    comparison_z = zscore(run.comparison_ecg)

    fig = plt.figure(figsize=(15, 7))
    grid_spec = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

    for column, fraction in enumerate(params.zoom_fractions):
        axis = fig.add_subplot(grid_spec[0, column])
        zoom_start_s = run.start_s + fraction * (run.stop_s - run.start_s) - params.zoom_window_s / 2
        in_zoom = (grid_s >= zoom_start_s) & (grid_s < zoom_start_s + params.zoom_window_s)
        axis.plot(grid_s[in_zoom], reference_z[in_zoom], lw=1.2, label=f'{params.reference_name} ECG')
        axis.plot(grid_s[in_zoom], comparison_z[in_zoom], lw=1.2, alpha=0.8, label=f'{params.comparison_name} ECG')
        axis.set_xlabel('EEG time (s)')
        axis.set_title(f't ≈ {zoom_start_s + params.zoom_window_s / 2:.0f} s')
        if column == 0:
            axis.set_ylabel('z-score')
            axis.legend(loc='upper right', fontsize=8)

    axis = fig.add_subplot(grid_spec[1, :])
    axis.plot(run.rolling_centres_s, run.rolling_correlations, 'o-', ms=3)
    lag_axis = axis.twinx()
    above = run.rolling_correlations > params.min_correlation
    lag_axis.plot(run.rolling_centres_s[above], run.rolling_lags_ms[above], 's', color='darkorange', ms=4, label='lag (ms)')
    lag_axis.set_ylabel('peak lag (ms), windows above threshold', color='darkorange')
    axis.axhline(params.min_correlation, color='gray', ls=':')
    axis.axvspan(run.start_s, run.stop_s, color='g', alpha=0.15, label=f'first run |r|>{params.min_correlation}')
    axis.axvline(0, color='r', ls='--', label='EEG start')
    axis.set_xlabel('EEG time (s)')
    axis.set_ylabel(f'peak |r| within ±{1e3 * params.max_lag_s:.0f} ms ({params.rolling_window_s}s windows)')
    axis.legend()

    fig.suptitle(
        f'Aligned BI-Vital ECGs ({params.reference_name} vs {params.comparison_name}), '
        f'run |r|>{params.min_correlation} ({run.start_s:.0f}..{run.stop_s:.0f} s): '
        f'peak r={run.correlation:.3f}, r(0)={run.correlation_zero_lag:.3f}, residual lag={run.lag_ms:.1f} ms'
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=params.dpi)


def plot_xcorr_qc(device_name, window_correlations, global_xcorr, uncorrected_xcorr, params, output_path):
    """Plot cross-correlation functions that check the alignment of one device.

    Panels: (top left) correlation curve of every tracking window around the
    fitted alignment with the peak selected by the tracking marked; (top
    right) cross-correlation of the aligned ECG and EEG proxy over the whole
    EEG period; (bottom left) zoom of that curve around 0 ms; (bottom right)
    selected peak lag of each window versus EEG time. These four panels use
    lag axes in ms relative to the fitted alignment (0 = perfectly aligned;
    positive = ECG later than the EEG proxy). The bottom row shows the
    uncorrected alignment (start-time lag, no drift): its wide
    cross-correlation (0 = start-time alignment) and a peak-centred
    comparison with the fitted alignment, which shows how much correlation
    the uncorrected offset and drift cost.

    Parameters
    ----------
    device_name : str
        Device name for titles.
    window_correlations : clocksync.WindowCorrelations
        Per-window curves and selected peaks.
    global_xcorr : clocksync.AlignmentXcorr
        Whole-period cross-correlation of the fitted alignment and its peak.
    uncorrected_xcorr : clocksync.AlignmentXcorr
        Whole-period cross-correlation with the uncorrected alignment (wide
        lag range) and its peak.
    params : QcParams
        Zoom range and resolution.
    output_path : str or Path
        PNG file to write.
    """
    windows = window_correlations
    used = windows.used_mask
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    ax = axes[0, 0]
    colours = plt.cm.viridis((windows.centres_s - windows.centres_s.min()) / max(np.ptp(windows.centres_s), 1))
    for i, curve in enumerate(windows.curves):
        ax.plot(windows.lags_ms, curve, color=colours[i] if used[i] else 'red',
                ls='-' if used[i] else '--', lw=0.9, alpha=0.8)
    ax.plot(windows.selected_lag_ms[used], windows.selected_value[used], 'o', mfc='none', mec='k', ms=7,
            label='selected peak (used)')
    ax.plot(windows.selected_lag_ms[~used], windows.selected_value[~used], 'rx', ms=7, label='selected peak (rejected)')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('lag relative to fitted alignment (ms)')
    ax.set_ylabel('correlation')
    ax.set_title(f'{device_name}: window cross-correlations (colour = EEG time)')
    ax.legend(fontsize=8, loc='upper right')

    for ax, half_range_ms, title in (
        (axes[0, 1], None, 'whole EEG period, aligned ECG vs EEG proxy'),
        (axes[1, 0], params.zoom_lag_ms, f'zoom ±{params.zoom_lag_ms:.0f} ms'),
    ):
        lags, correlation = global_xcorr.lags_ms, global_xcorr.correlation
        if half_range_ms is not None:
            inside = np.abs(lags) <= half_range_ms
            lags, correlation = lags[inside], correlation[inside]
        ax.plot(lags, correlation, '.-' if half_range_ms is not None else '-', lw=1.0)
        ax.plot(global_xcorr.peak_lag_ms, global_xcorr.peak_correlation, 'ro', ms=8,
                label=f'peak {global_xcorr.peak_lag_ms:+.1f} ms, r={global_xcorr.peak_correlation:.3f}')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('lag (ms)  [+ : ECG later than EEG proxy]')
        ax.set_ylabel('correlation')
        ax.set_title(f'{device_name}: {title}')
        ax.legend(fontsize=9, loc='upper right')

    ax = axes[1, 1]
    ax.plot(windows.centres_s[used], windows.selected_lag_ms[used], 'o', label='used')
    ax.plot(windows.centres_s[~used], windows.selected_lag_ms[~used], 'rx', label='rejected')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('EEG time (s)')
    ax.set_ylabel('selected peak lag (ms)')
    ax.set_title(f'{device_name}: residual of each tracking window')
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(uncorrected_xcorr.lags_ms, uncorrected_xcorr.correlation, lw=0.8)
    ax.plot(uncorrected_xcorr.peak_lag_ms, uncorrected_xcorr.peak_correlation, 'ro', ms=8,
            label=f'peak {uncorrected_xcorr.peak_lag_ms:+.0f} ms ({uncorrected_xcorr.peak_lag_ms / 1e3:+.2f} s), '
                  f'r={uncorrected_xcorr.peak_correlation:.3f}')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('lag relative to the start-time alignment (ms)')
    ax.set_ylabel('correlation')
    ax.set_title(f'{device_name}: uncorrected alignment (start-time lag, no drift)')
    ax.legend(fontsize=9, loc='upper right')

    ax = axes[2, 1]
    half_range_ms = params.uncorrected_zoom_ms
    for xcorr, label, colour in (
        (global_xcorr, 'fitted alignment (offset + drift)', 'tab:blue'),
        (uncorrected_xcorr, 'start-time lag, no drift', 'tab:orange'),
    ):
        centred_lags = xcorr.lags_ms - xcorr.peak_lag_ms
        inside = np.abs(centred_lags) <= half_range_ms
        ax.plot(centred_lags[inside], xcorr.correlation[inside], '.-', color=colour, lw=1.0,
                label=f'{label}: peak r={xcorr.peak_correlation:.3f}')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('lag relative to each curve\'s own peak (ms)')
    ax.set_ylabel('correlation')
    ax.set_title(f'{device_name}: peak-centred comparison, ±{half_range_ms:.0f} ms')
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    fig.savefig(output_path, dpi=params.dpi)
    plt.close(fig)


def plot_aligned_traces(device_name, eeg_time_s, eeg_proxy, ecg_on_eeg, global_xcorr, params, output_path):
    """Overlay the aligned ECG and the EEG-derived cardiac proxy.

    Short zooms at several positions of the EEG period show both
    standardised signals on the EEG time axis; the zero-lag correlation of
    each zoom is given in its title. The ECG is flipped for display if the
    global correlation peak is negative.

    Parameters
    ----------
    device_name : str
        Device name for titles.
    eeg_time_s : numpy.ndarray
        EEG time axis (s) shared by both signals.
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy.
    ecg_on_eeg : numpy.ndarray
        ECG resampled onto ``eeg_time_s``.
    global_xcorr : clocksync.AlignmentXcorr
        Whole-period cross-correlation (supplies the polarity).
    params : QcParams
        Zoom positions, length and resolution.
    output_path : str or Path
        PNG file to write.
    """
    proxy_z = zscore(eeg_proxy)
    ecg_z = zscore(ecg_on_eeg) * np.sign(global_xcorr.peak_correlation)
    fs_hz = 1 / (eeg_time_s[1] - eeg_time_s[0])
    window_len = int(params.zoom_window_s * fs_hz)

    fig, axes = plt.subplots(len(params.zoom_fractions), 1, figsize=(14, 3.2 * len(params.zoom_fractions)))
    axes = np.atleast_1d(axes)
    for ax, fraction in zip(axes, params.zoom_fractions):
        start_idx = int(fraction * (len(eeg_time_s) - window_len))
        window = slice(start_idx, start_idx + window_len)
        zero_lag_r = np.corrcoef(proxy_z[window], ecg_z[window])[0, 1]
        ax.plot(eeg_time_s[window], ecg_z[window], lw=1.2, label='BI-Vital ECG (aligned)')
        ax.plot(eeg_time_s[window], proxy_z[window], lw=1.0, alpha=0.8, label='ECG proxy from EEG')
        ax.set_ylabel('z-score')
        ax.set_title(f'{device_name}: t ≈ {eeg_time_s[start_idx] + params.zoom_window_s / 2:.0f} s, zero-lag r = {zero_lag_r:.2f}')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('EEG time (s)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=params.dpi)
    plt.close(fig)
