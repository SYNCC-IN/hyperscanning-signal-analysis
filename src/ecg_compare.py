"""Comparison of two ECGs aligned to a common EEG time axis."""
from dataclasses import dataclass

import numpy as np

from .dsp import normalized_xcorr, zscore


@dataclass
class SharedRunParams:
    """Parameters for locating the period in which two devices see the same heart.

    Attributes
    ----------
    fs_hz : int
        Sampling rate of the common grid.
    rolling_window_s : float
        Window of the rolling correlation.
    rolling_step_s : float
        Hop of the rolling correlation.
    min_correlation : float
        Threshold on the peak ``|correlation|`` defining the "same wearer" run.
    max_lag_s : float
        Search range (+/-) for the inter-device lag in every window.
    """

    fs_hz: int
    rolling_window_s: float
    rolling_step_s: float
    min_correlation: float
    max_lag_s: float


@dataclass
class SharedWearerRun:
    """First contiguous period with high correlation between two ECGs.

    Attributes
    ----------
    comparison_ecg : numpy.ndarray
        Comparison ECG with its polarity matched to the reference.
    rolling_centres_s : numpy.ndarray
        Centres (EEG s) of the rolling windows.
    rolling_correlations : numpy.ndarray
        Peak ``|correlation|`` within the allowed lag range per window.
    rolling_lags_ms : numpy.ndarray
        Lag (ms) of that peak per window (positive: reference delayed).
    start_s, stop_s : float
        Bounds of the run in EEG time.
    correlation : float
        Peak ``|correlation|`` of the run at its best lag.
    correlation_zero_lag : float
        Correlation of the polarity-matched signals at zero lag in the run.
    lag_ms : float
        Residual lag (ms) of the reference relative to the comparison
        (positive: reference delayed).
    """

    comparison_ecg: np.ndarray
    rolling_centres_s: np.ndarray
    rolling_correlations: np.ndarray
    rolling_lags_ms: np.ndarray
    start_s: float
    stop_s: float
    correlation: float
    correlation_zero_lag: float
    lag_ms: float


def build_common_time_grid(eeg_time_by_device, fs_hz):
    """Build a uniform EEG-time grid covering the span common to all devices.

    Parameters
    ----------
    eeg_time_by_device : dict of str to numpy.ndarray
        ECG sample times in EEG time per device (may be negative).
    fs_hz : float
        Grid sampling rate.

    Returns
    -------
    numpy.ndarray
        Grid from the latest first sample to the earliest last sample.
    """
    start_s = max(times[0] for times in eeg_time_by_device.values())
    stop_s = min(times[-1] for times in eeg_time_by_device.values())
    first_idx = np.ceil(start_s * fs_hz)
    last_idx = np.floor(stop_s * fs_hz)
    return np.arange(first_idx, last_idx) / fs_hz


def align_to_common_grid(eeg_time_by_device, ecg_by_device, fs_hz):
    """Interpolate every device ECG onto one common EEG-time grid.

    The grid is not cropped to the EEG recording: the part recorded before
    the EEG started has negative times.

    Parameters
    ----------
    eeg_time_by_device : dict of str to numpy.ndarray
        ECG sample times in EEG time per device.
    ecg_by_device : dict of str to numpy.ndarray
        ECG samples per device.
    fs_hz : float
        Grid sampling rate.

    Returns
    -------
    grid_s : numpy.ndarray
        Common EEG-time grid.
    aligned : dict of str to numpy.ndarray
        ECG per device on ``grid_s``.
    """
    grid_s = build_common_time_grid(eeg_time_by_device, fs_hz)
    aligned = {
        name: np.interp(grid_s, eeg_time_by_device[name], ecg_by_device[name])
        for name in ecg_by_device
    }
    return grid_s, aligned


def peak_xcorr(signal_a, signal_b, fs_hz, max_lag_s):
    """Peak of the normalised cross-correlation within a lag range.

    Both signals are standardised, so the values are correlation coefficients.
    The peak is chosen by ``|correlation|`` (polarity independent) and its
    signed value is returned.

    Parameters
    ----------
    signal_a, signal_b : numpy.ndarray
        Equally long signals.
    fs_hz : float
        Sampling rate.
    max_lag_s : float
        Search range (+/-) for the lag.

    Returns
    -------
    peak_correlation : float
        Signed correlation at the peak.
    lag_ms : float
        Lag of the peak in ms; positive means ``signal_a`` is delayed
        relative to ``signal_b``.
    """
    lags_ms, correlation = normalized_xcorr(signal_a, signal_b, fs_hz, max_lag_s)
    best = int(np.argmax(np.abs(correlation)))
    return correlation[best], lags_ms[best]


def rolling_peak_correlation(signal_a, signal_b, grid_s, fs_hz, window_s, step_s, max_lag_s):
    """Lag-tolerant correlation of two aligned signals in sliding windows.

    In every window the peak of the cross-correlation within ``+/- max_lag_s``
    is taken, so a small residual misalignment does not hide a shared signal.

    Parameters
    ----------
    signal_a, signal_b : numpy.ndarray
        Signals sampled on ``grid_s``.
    grid_s : numpy.ndarray
        Uniform time grid (seconds).
    fs_hz : float
        Grid sampling rate.
    window_s, step_s : float
        Window length and hop.
    max_lag_s : float
        Search range (+/-) for the lag.

    Returns
    -------
    centres_s : numpy.ndarray
        Window centres (EEG s).
    peak_correlations : numpy.ndarray
        Signed correlation at the peak per window.
    lags_ms : numpy.ndarray
        Lag of the peak per window (ms).
    """
    window_len = int(window_s * fs_hz)
    step_len = int(step_s * fs_hz)
    centres, peaks, lags_ms = [], [], []
    for start_idx in range(0, len(grid_s) - window_len, step_len):
        window = slice(start_idx, start_idx + window_len)
        peak, lag_ms = peak_xcorr(signal_a[window], signal_b[window], fs_hz, max_lag_s)
        centres.append(grid_s[start_idx] + window_s / 2)
        peaks.append(peak)
        lags_ms.append(lag_ms)
    return np.array(centres), np.array(peaks), np.array(lags_ms)


def first_true_run(mask):
    """Find the first contiguous run of ``True`` values.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array.

    Returns
    -------
    first, last : int
        Indices of the first and last element of the run (inclusive).

    Raises
    ------
    ValueError
        If ``mask`` contains no ``True``.
    """
    if not mask.any():
        raise ValueError('mask contains no True value')
    first = int(np.argmax(mask))
    last = first
    while last + 1 < len(mask) and mask[last + 1]:
        last += 1
    return first, last


def find_shared_wearer_run(grid_s, reference_ecg, comparison_ecg, params):
    """Locate the period in which two ECGs correlate strongly.

    The lag-tolerant rolling correlation defines the first contiguous run
    with peak ``|correlation|`` above ``params.min_correlation``. Over that
    run the peak correlation, its lag and the zero-lag correlation are
    computed, and the comparison ECG is sign-matched to the reference
    (polarity taken from the run itself).

    Parameters
    ----------
    grid_s : numpy.ndarray
        Common EEG-time grid.
    reference_ecg, comparison_ecg : numpy.ndarray
        Aligned ECGs on ``grid_s``.
    params : SharedRunParams
        Thresholds and window settings.

    Returns
    -------
    SharedWearerRun
        Run bounds, statistics and the polarity-matched comparison ECG.

    Raises
    ------
    ValueError
        If no window exceeds the correlation threshold.
    """
    centres_s, peaks, lags_ms = rolling_peak_correlation(
        reference_ecg, comparison_ecg, grid_s, params.fs_hz,
        params.rolling_window_s, params.rolling_step_s, params.max_lag_s,
    )
    correlations = np.abs(peaks)
    first, last = first_true_run(correlations > params.min_correlation)
    start_s = centres_s[first] - params.rolling_window_s / 2
    stop_s = centres_s[last] + params.rolling_window_s / 2
    in_run = (grid_s >= start_s) & (grid_s < stop_s)

    run_peak, run_lag_ms = peak_xcorr(
        reference_ecg[in_run], comparison_ecg[in_run], params.fs_hz, params.max_lag_s
    )
    polarity = np.sign(run_peak)
    matched_comparison = comparison_ecg * polarity
    zero_lag_correlation = np.mean(zscore(reference_ecg[in_run]) * zscore(matched_comparison[in_run]))
    return SharedWearerRun(
        comparison_ecg=matched_comparison,
        rolling_centres_s=centres_s,
        rolling_correlations=correlations,
        rolling_lags_ms=lags_ms,
        start_s=start_s,
        stop_s=stop_s,
        correlation=abs(run_peak),
        correlation_zero_lag=zero_lag_correlation,
        lag_ms=run_lag_ms,
    )
