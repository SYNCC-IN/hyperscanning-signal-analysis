"""ECG, IBI and RMSSD of BI-Vital recordings on the EEG time grid.

The BI-Vital ECG (200 Hz, device clock) is filtered like the Warsaw ECG
(high-pass 0.5 Hz, 50 Hz notch), low-passed for the target rate, mapped to EEG
time with the fitted clock model and interpolated onto the common EEG grid.
R-peaks are detected on the native 200 Hz signal (NeuroKit2, refined to
sub-sample precision) and converted to EEG time; IBI and RMSSD are then
derived from the peak times and evaluated on the same grid.

Differences from the Warsaw implementation (``MultimodalData._set_ibi`` /
``_set_RMSSD_from_ECG``), needed because BI-Vital data contain dropouts:

* all steps are NaN-aware: dropouts are never bridged, IBI/RMSSD stay NaN there;
* IBI outside a physiological range is rejected (separate ranges per member);
* the RMSSD is computed in time units, not in samples, requires a minimum
  number of successive IBI differences in the window and a first beat close to
  the window start (no window that starts inside a dropout).

Conventions kept from Warsaw: IBI in ms, time-stamped at the later R-peak and
interpolated with a cubic spline; RMSSD over a window of 30 s aligned to the
window start.
"""
from dataclasses import dataclass, field

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import CubicSpline

from .clocksync import device_time_to_eeg_time
from .dsp import refine_peak_parabolic
from .synced_export import interpolate_to_grid


@dataclass
class HrvParams:
    """Parameters of the ECG / IBI / RMSSD computation."""

    ecg_fs_hz: float = 200.0                # BI-Vital ECG sampling rate
    ecg_max_gap_s: float = 0.1              # longer gaps in the sample stream invalidate the ECG
    min_run_s: float = 5.0                  # shorter continuous ECG runs are discarded
    highpass_hz: float = 0.5                # as in Warsaw (5th-order Butterworth)
    highpass_order: int = 5
    notch_hz: float = 50.0
    notch_q: float = 30.0
    lowpass_hz: float = 51.0                # anti-alias for the 128 Hz grid (0.8 * Nyquist of 128 Hz)
    lowpass_order: int = 4
    grid_max_gap_s: float = 0.05            # longest gap bridged when mapping ECG onto the grid
    peak_snap_samples: int = 2              # NeuroKit peaks are snapped to the local extremum within +/- this
    ibi_range_ms: dict = field(default_factory=lambda: {'ch': (300.0, 1200.0), 'cg': (400.0, 1500.0)})
    ibi_median_tol: float | None = None     # optional: reject IBI deviating more than this fraction from the local median
    ibi_median_beats: int = 11
    max_ibi_gap_s: float = 3.0              # IBI spline is not bridged over longer gaps
    rmssd_window_s: float = 30.0
    rmssd_min_diffs: int = 10               # minimum IBI differences in a window, else NaN


def uniform_ecg_with_mask(table, fs_hz, max_gap_s):
    """Put the raw BI-Vital ECG on a uniform device-time grid and mark dropouts.

    Parameters
    ----------
    table : pandas.DataFrame
        BI-Vital table with ``t`` (device time, s) and ``ecg``.
    fs_hz : float
    max_gap_s : float
        Gaps between valid samples longer than this are marked invalid
        (shorter ones, i.e. a few lost samples, are bridged linearly).

    Returns
    -------
    time_s, ecg, valid : numpy.ndarray
        Device time of every sample, raw ECG (gaps interpolated) and a boolean
        mask that is False inside long gaps.
    """
    valid_rows = table.dropna(subset=['ecg'])
    t_valid = valid_rows['t'].to_numpy()
    n_samples = int(np.floor((t_valid[-1] - t_valid[0]) * fs_hz)) + 1
    time_s = t_valid[0] + np.arange(n_samples) / fs_hz
    ecg = np.interp(time_s, t_valid, valid_rows['ecg'].to_numpy())
    mask = np.ones(n_samples, dtype=bool)
    for i in np.where(np.diff(t_valid) > max_gap_s)[0]:
        first = int(np.floor((t_valid[i] - t_valid[0]) * fs_hz)) + 1
        last = int(np.ceil((t_valid[i + 1] - t_valid[0]) * fs_hz))
        mask[first:last] = False
    return time_s, ecg, mask


def mask_runs(mask, min_len):
    """Return ``(start, stop)`` index pairs of contiguous True runs of at least ``min_len`` samples."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.diff(padded)
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(a, b) for a, b in zip(starts, stops) if b - a >= min_len]


def filter_ecg(ecg, mask, params):
    """Filter every valid run: median removal, high-pass, 50 Hz notch, low-pass (all zero-phase).

    Returns
    -------
    numpy.ndarray
        Filtered ECG with NaN outside the valid runs.
    """
    fs = params.ecg_fs_hz
    sos_high = signal.butter(params.highpass_order, params.highpass_hz, 'high', fs=fs, output='sos')
    sos_notch = signal.tf2sos(*signal.iirnotch(params.notch_hz, params.notch_q, fs=fs))
    sos_low = signal.butter(params.lowpass_order, params.lowpass_hz, 'low', fs=fs, output='sos')
    filtered = np.full(len(ecg), np.nan)
    for start, stop in mask_runs(mask, int(params.min_run_s * fs)):
        run = ecg[start:stop] - np.median(ecg[start:stop])
        for sos in (sos_high, sos_notch, sos_low):
            run = signal.sosfiltfilt(sos, run)
        filtered[start:stop] = run
    return filtered


def detect_r_peaks(time_s, ecg_filtered, params):
    """Detect R-peaks per valid run (NeuroKit2) and refine them to sub-sample precision.

    Parameters
    ----------
    time_s : numpy.ndarray
        Device time of every ECG sample (uniform grid).
    ecg_filtered : numpy.ndarray
        Filtered ECG, NaN outside valid runs (`filter_ecg`).

    Returns
    -------
    peak_time_s : numpy.ndarray
        Device time of the R-peaks.
    run_id : numpy.ndarray of int
        Index of the continuous run each peak belongs to.
    """
    fs = params.ecg_fs_hz
    runs = mask_runs(~np.isnan(ecg_filtered), int(params.min_run_s * fs))
    if not runs:
        return np.array([]), np.array([], dtype=int)
    all_valid = np.concatenate([ecg_filtered[a:b] for a, b in runs])
    polarity = -1.0 if stats.skew(all_valid) < 0 else 1.0     # make R-waves positive
    peak_times, run_ids = [], []
    for run_index, (start, stop) in enumerate(runs):
        y = polarity * ecg_filtered[start:stop]
        try:
            peaks = nk.ecg_peaks(y, sampling_rate=fs, method='neurokit')[1]['ECG_R_Peaks']
        except Exception:                                     # run too short / no detectable beats
            continue
        snapped = []
        for peak in np.asarray(peaks, dtype=int):
            lo, hi = max(peak - params.peak_snap_samples, 0), min(peak + params.peak_snap_samples + 1, len(y))
            snapped.append(lo + int(np.argmax(y[lo:hi])))
        for peak in np.unique(snapped):
            refined = refine_peak_parabolic(y - y.min(), int(peak))
            peak_times.append(time_s[start] + refined / fs)
            run_ids.append(run_index)
    return np.asarray(peak_times), np.asarray(run_ids, dtype=int)


def ibi_from_peaks(peak_time_eeg_s, run_id, range_ms, median_tol=None, median_beats=11):
    """Interbeat intervals from R-peak times.

    The IBI of beat ``k`` is ``t_k - t_{k-1}`` (ms) and is stamped at ``t_k``.
    It is NaN across a dropout (different run), outside ``range_ms`` or, if
    ``median_tol`` is given, when it deviates by more than this fraction from the
    local median.

    Returns
    -------
    ibi_ms : numpy.ndarray
        Same length as the peaks; first element NaN.
    """
    ibi = np.full(len(peak_time_eeg_s), np.nan)
    if len(peak_time_eeg_s) < 2:
        return ibi
    ibi[1:] = np.diff(peak_time_eeg_s) * 1000.0
    ibi[1:][np.diff(run_id) != 0] = np.nan
    ibi[(ibi < range_ms[0]) | (ibi > range_ms[1])] = np.nan
    if median_tol is not None:
        local = pd.Series(ibi).rolling(median_beats, center=True, min_periods=3).median().to_numpy()
        ibi[np.abs(ibi - local) > median_tol * local] = np.nan
    return ibi


def interpolate_ibi(stamps_s, ibi_ms, grid_s, max_gap_s):
    """Cubic-spline IBI onto ``grid_s``; NaN outside data and across gaps longer than ``max_gap_s``."""
    out = np.full(len(grid_s), np.nan)
    valid = ~np.isnan(ibi_ms)
    t, y = stamps_s[valid], ibi_ms[valid]
    if len(t) < 4:
        return out
    breaks = np.where(np.diff(t) > max_gap_s)[0] + 1
    for seg_t, seg_y in zip(np.split(t, breaks), np.split(y, breaks)):
        if len(seg_t) < 4:
            continue
        inside = (grid_s >= seg_t[0]) & (grid_s <= seg_t[-1])
        out[inside] = CubicSpline(seg_t, seg_y)(grid_s[inside])
    return out


def rmssd_sliding(peak_time_eeg_s, ibi_ms, grid_s, window_s, min_diffs, max_start_gap_s=np.inf):
    """Sliding-window RMSSD on ``grid_s`` (value at ``g`` uses beats in ``[g, g + window_s)``).

    Only successive differences of two valid IBI (three consecutive peaks, all
    inside the window) enter. Windows with fewer than ``min_diffs`` differences are NaN,
    and so are windows whose first beat is more than ``max_start_gap_s`` after the window
    start (the window would begin inside a dropout and only look at data after it).
    """
    out = np.full(len(grid_s), np.nan)
    if len(peak_time_eeg_s) < 3:
        return out
    valid_ibi = ~np.isnan(ibi_ms)
    diff = ibi_ms[2:] - ibi_ms[1:-1]
    valid = valid_ibi[2:] & valid_ibi[1:-1]
    squared = np.where(valid, diff, 0.0) ** 2
    start_key = peak_time_eeg_s[:-2]           # first peak of each difference
    end_key = peak_time_eeg_s[2:]              # last peak of each difference
    cum_count = np.concatenate([[0], np.cumsum(valid)])
    cum_sq = np.concatenate([[0.0], np.cumsum(squared)])
    lo = np.searchsorted(start_key, grid_s, side='left')
    hi = np.maximum(np.searchsorted(end_key, grid_s + window_s, side='left'), lo)
    count = cum_count[hi] - cum_count[lo]
    first_beat_gap = np.full(len(grid_s), np.inf)
    has_beat = lo < len(start_key)
    first_beat_gap[has_beat] = start_key[lo[has_beat]] - grid_s[has_beat]
    ok = (count >= min_diffs) & (first_beat_gap <= max_start_gap_s)
    out[ok] = np.sqrt((cum_sq[hi] - cum_sq[lo])[ok] / count[ok])
    return out


def compute_hrv_on_grid(table, fit, member, grid_s, params=None):
    """ECG, IBI and RMSSD of one BI-Vital device on the EEG grid.

    Parameters
    ----------
    table : pandas.DataFrame
        Raw BI-Vital table of the device.
    fit : clocksync.SyncFit
        Clock fit of the device (device time -> EEG time).
    member : str
        ``'ch'`` or ``'cg'`` (selects the IBI range).
    grid_s : numpy.ndarray
        Target EEG time grid (seconds).
    params : HrvParams, optional

    Returns
    -------
    dict
        ``ECG``, ``IBI`` (ms), ``RMSSD`` (ms) arrays on ``grid_s`` and ``info``, a
        dict with ``n_peaks``, ``n_ibi_valid``, ``n_ibi_rejected``, ``mean_hr_bpm``
        and ``valid_ecg_fraction`` (of the grid).
    """
    params = params or HrvParams()
    time_s, raw_ecg, mask = uniform_ecg_with_mask(table, params.ecg_fs_hz, params.ecg_max_gap_s)
    ecg_filtered = filter_ecg(raw_ecg, mask, params)
    ecg_grid = interpolate_to_grid(
        grid_s, device_time_to_eeg_time(time_s, fit), ecg_filtered, params.grid_max_gap_s
    )

    peak_dev_s, run_id = detect_r_peaks(time_s, ecg_filtered, params)
    peak_eeg_s = device_time_to_eeg_time(peak_dev_s, fit)
    ibi = ibi_from_peaks(
        peak_eeg_s, run_id, params.ibi_range_ms[member], params.ibi_median_tol, params.ibi_median_beats
    )
    ibi_grid = interpolate_ibi(peak_eeg_s, ibi, grid_s, params.max_ibi_gap_s)
    rmssd_grid = rmssd_sliding(
        peak_eeg_s, ibi, grid_s, params.rmssd_window_s, params.rmssd_min_diffs, params.max_ibi_gap_s
    )

    n_valid = int(np.sum(~np.isnan(ibi)))
    n_candidates = max(len(ibi) - 1, 0)
    info = {
        'n_peaks': int(len(peak_eeg_s)),
        'n_ibi_valid': n_valid,
        'n_ibi_rejected': int(np.sum(np.isnan(ibi[1:]) & (np.diff(run_id) == 0))) if n_candidates else 0,
        'mean_hr_bpm': float(60000.0 / np.nanmean(ibi)) if n_valid else float('nan'),
        'valid_ecg_fraction': float(np.mean(~np.isnan(ecg_grid))),
    }
    return {'ECG': ecg_grid, 'IBI': ibi_grid, 'RMSSD': rmssd_grid, 'info': info}
