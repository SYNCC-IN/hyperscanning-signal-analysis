"""Generic signal-processing helpers (no knowledge of the synchronisation project)."""
import numpy as np
from scipy import signal


def zscore(values):
    """Standardise an array to zero mean and unit standard deviation.

    Parameters
    ----------
    values : numpy.ndarray
        Input signal.

    Returns
    -------
    numpy.ndarray
        ``(values - mean) / std``.
    """
    return (values - values.mean()) / values.std()


def design_bandpass(fs_hz, band_hz, order):
    """Design a Butterworth band-pass filter.

    Parameters
    ----------
    fs_hz : float
        Sampling rate of the signal to be filtered.
    band_hz : tuple of float
        Lower and upper cut-off frequencies.
    order : int
        Filter order (per direction when used with ``sosfiltfilt``).

    Returns
    -------
    numpy.ndarray
        Second-order sections.
    """
    return signal.butter(order, band_hz, 'bp', fs=fs_hz, output='sos')


def refine_peak_parabolic(values, peak_index):
    """Refine a peak position to sub-sample precision.

    Fits a parabola through the peak and its two neighbours and returns the
    position of its vertex. Peaks on the array edge are returned unchanged.

    Parameters
    ----------
    values : numpy.ndarray
        Non-negative curve containing the peak (e.g. ``|xcorr|``).
    peak_index : int
        Index of the discrete maximum.

    Returns
    -------
    float
        Fractional index of the peak.
    """
    if not 0 < peak_index < len(values) - 1:
        return float(peak_index)
    left, centre, right = values[peak_index - 1:peak_index + 2]
    return peak_index + 0.5 * (left - right) / (left - 2 * centre + right)


def normalized_xcorr(signal_a, signal_b, fs_hz, max_lag_s):
    """Normalised cross-correlation of two equally long signals within a lag range.

    Both signals are standardised, so the values are correlation coefficients.

    Parameters
    ----------
    signal_a, signal_b : numpy.ndarray
        Equally long signals.
    fs_hz : float
        Sampling rate.
    max_lag_s : float
        Lag range (+/-) to return.

    Returns
    -------
    lags_ms : numpy.ndarray
        Lags in ms; positive means ``signal_a`` is delayed relative to ``signal_b``.
    correlation : numpy.ndarray
        Correlation coefficient at each lag.
    """
    full = signal.correlate(zscore(signal_a), zscore(signal_b), mode='full') / len(signal_a)
    lags = signal.correlation_lags(len(signal_a), len(signal_b))
    allowed = np.abs(lags) <= int(max_lag_s * fs_hz)
    return 1e3 * lags[allowed] / fs_hz, full[allowed]


def curve_peak(lags_ms, correlation, fs_hz):
    """Locate the peak of a correlation curve with sub-sample precision.

    The peak is chosen by ``|correlation|`` and refined by a parabola.

    Parameters
    ----------
    lags_ms : numpy.ndarray
        Lag axis in ms (uniform spacing of ``1000 / fs_hz``).
    correlation : numpy.ndarray
        Correlation values.
    fs_hz : float
        Sampling rate.

    Returns
    -------
    peak_lag_ms : float
        Refined lag of the peak.
    peak_value : float
        Signed correlation at the integer-sample peak.
    """
    abs_correlation = np.abs(correlation)
    integer_peak = int(np.argmax(abs_correlation))
    refined_peak = refine_peak_parabolic(abs_correlation, integer_peak)
    peak_lag_ms = lags_ms[integer_peak] + 1e3 * (refined_peak - integer_peak) / fs_hz
    return peak_lag_ms, correlation[integer_peak]
