"""Clock synchronisation of BI-Vital devices to the EEG.

Model: ``t_dev = a + (1 + b) * t_eeg``, where ``a`` is the device time of EEG
t=0 and ``b`` the relative clock drift. The offset is tracked by windowed
cross-correlation of the cardiac artefact in EEG against the device ECG.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import signal, stats

from .dsp import curve_peak, normalized_xcorr, refine_peak_parabolic, zscore


@dataclass
class TrackingParams:
    """Parameters of the windowed offset tracking.

    Attributes
    ----------
    fs_hz : float
        Sampling rate of the EEG cardiac proxy and the ECG.
    window_s : float
        Length of each cross-correlation window.
    step_s : float
        Hop between consecutive windows.
    margin_s : float
        Search range (+/-) around the position predicted by the coarse lag.
    """

    fs_hz: int
    window_s: float
    step_s: float
    margin_s: float


@dataclass
class SyncFit:
    """Result of the clock fit ``t_dev = intercept_s + (1 + slope) * t_eeg``.

    Attributes
    ----------
    slope : float
        Relative clock drift ``b`` (dimensionless).
    intercept_s : float
        Device time ``a`` (s) corresponding to EEG t=0.
    used_mask : numpy.ndarray
        Boolean mask of tracked windows kept in the final fit.
    tracks : pandas.DataFrame
        Per-window ``eeg_t``, ``offset`` (device time) and ``r``.
    """

    slope: float
    intercept_s: float
    used_mask: np.ndarray
    tracks: pd.DataFrame


def estimate_coarse_lag_s(eeg_start, device, fs_hz, margin_s):
    """Estimate the position of EEG t=0 in the device ECG array from start times.

    ``lag = (eeg_start - device_start) - first_sample_time``: the wall-clock
    difference of the two recording starts, expressed relative to the first
    valid ECG sample (the time base used by `track_offset`).

    Parameters
    ----------
    eeg_start : pandas.Timestamp
        Wall-clock start of the EEG.
    device : cardiac.DeviceRecording
        BI-Vital recording.
    fs_hz : float
        ECG sampling rate.
    margin_s : float
        Search margin that must fit before the predicted position.

    Returns
    -------
    float
        Coarse lag in seconds.

    Raises
    ------
    ValueError
        If the search window would start before the first ECG sample.
    """
    lag_s = (eeg_start - device.start_time).total_seconds() - device.first_sample_time_s
    if (lag_s - margin_s) * fs_hz < 0:
        raise ValueError(
            f'{device.name}: coarse lag {lag_s:.1f} s puts the search window before the ECG start'
        )
    return lag_s


def correlation_around_lag(eeg_proxy, ecg, eeg_start_idx, centre_lag_samples, window_len, margin_len, clip=False):
    """Cross-correlate one EEG window with the ECG around a predicted position.

    The standardised EEG window starting at ``eeg_start_idx`` is slid over the
    standardised ECG segment that extends ``margin_len`` samples on both sides
    of the position ``eeg_start_idx + centre_lag_samples``.

    Parameters
    ----------
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy.
    ecg : numpy.ndarray
        Filtered ECG at the same sampling rate.
    eeg_start_idx : int
        First sample of the EEG window.
    centre_lag_samples : int
        Predicted ECG-array position of EEG t=0 (samples).
    window_len, margin_len : int
        Window length and one-sided search margin in samples.
    clip : bool, optional
        Cut the ECG segment at the ends of the ECG instead of raising (very wide margins);
        the correlation is then shorter than ``2 * margin_len + 1``. Raises only if less
        than one window length of ECG remains.

    Returns
    -------
    correlation : numpy.ndarray
        Correlation coefficients of length ``2 * margin_len + 1`` (shorter when clipped); index ``k``
        corresponds to a lag of ``k - margin_len`` samples relative to the
        predicted position (positive: ECG later than predicted).
    segment_start_idx : int
        First sample of the ECG segment (ECG-array index).

    Raises
    ------
    ValueError
        If the ECG segment lies outside the ECG array.
    """
    template = zscore(eeg_proxy[eeg_start_idx:eeg_start_idx + window_len])
    segment_start_idx = centre_lag_samples + eeg_start_idx - margin_len
    segment_stop_idx = segment_start_idx + window_len + 2 * margin_len
    if clip:
        segment_start_idx = max(segment_start_idx, 0)
        segment_stop_idx = min(segment_stop_idx, len(ecg))
        if segment_stop_idx - segment_start_idx < window_len:
            raise ValueError('less than one window of ECG left inside the search range')
    if segment_start_idx < 0 or segment_stop_idx > len(ecg):
        raise ValueError(
            f'ECG search segment [{segment_start_idx}, {segment_stop_idx}) for the window at EEG '
            f'sample {eeg_start_idx} lies outside the ECG (length {len(ecg)}); '
            f'check coarse_lag_s and the search margin'
        )
    segment = zscore(ecg[segment_start_idx:segment_stop_idx])
    correlation = signal.correlate(segment, template, mode='valid') / window_len
    return correlation, segment_start_idx


def track_offset(eeg_proxy, ecg, coarse_lag_s, params, skip_out_of_range=False):
    """Track the EEG-to-ECG offset with windowed cross-correlation.

    For each EEG window the window is standardised and slid over the matching
    ECG segment, which extends by ``params.margin_s`` on both sides of the
    position predicted by ``coarse_lag_s``. The maximum of ``|xcorr|`` (the
    artefact polarity is unknown) is refined by a parabola and converted to
    the offset of EEG t=0 relative to the first ECG sample.

    Parameters
    ----------
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy at ``params.fs_hz``.
    ecg : numpy.ndarray
        Filtered ECG at ``params.fs_hz``.
    coarse_lag_s : float
        Coarse EEG-to-ECG lag in seconds (ECG-array time).
    params : TrackingParams
        Window length, hop, search margin and sampling rate.
    skip_out_of_range : bool, optional
        Cut the search segment at the ends of the ECG instead of raising (needed for very wide
        search margins); windows with less than one window of ECG left are skipped; raises only
        if no window remains.

    Returns
    -------
    pandas.DataFrame
        One row per window with columns ``eeg_t`` (window centre, EEG s),
        ``offset`` (s, ECG-array time), ``r`` (peak ``|correlation|``) and
        ``r_signed`` (signed correlation at the peak, i.e. the polarity).
    """
    fs = params.fs_hz
    window_len = int(params.window_s * fs)
    step_len = int(params.step_s * fs)
    margin_len = int(params.margin_s * fs)
    coarse_lag_samples = int(round(coarse_lag_s * fs))

    rows = []
    for eeg_start_idx in range(0, len(eeg_proxy) - window_len, step_len):
        try:
            correlation, segment_start_idx = correlation_around_lag(
                eeg_proxy, ecg, eeg_start_idx, coarse_lag_samples, window_len, margin_len,
                clip=skip_out_of_range,
            )
        except ValueError:
            if skip_out_of_range:
                continue
            raise
        abs_correlation = np.abs(correlation)
        integer_peak_idx = int(np.argmax(abs_correlation))
        signed_peak = correlation[integer_peak_idx]
        peak_idx = refine_peak_parabolic(abs_correlation, integer_peak_idx)

        window_centre_s = eeg_start_idx / fs + params.window_s / 2
        offset_s = (segment_start_idx + peak_idx) / fs - eeg_start_idx / fs
        rows.append((window_centre_s, offset_s, abs_correlation.max(), signed_peak))
    if not rows:
        raise ValueError('no tracking window fits inside the ECG for this lag and search margin')
    return pd.DataFrame(rows, columns=['eeg_t', 'offset', 'r', 'r_signed'])


def line_through_offsets(tracks):
    """Theil-Sen line through the tracked offsets (drift-aware reference).

    Returns
    -------
    value_at_centre_s : float
        Line value at the mean window time (the offset that is comparable between channels).
    residuals : numpy.ndarray
        ``offset - line`` for every window.
    """
    t, y = tracks['eeg_t'].to_numpy(), tracks['offset'].to_numpy()
    if len(t) < 3:
        centre = float(np.median(y))
        return centre, y - centre
    slope, intercept = stats.theilslopes(y, t)[:2]
    residuals = y - (intercept + slope * t)
    return float(intercept + slope * t.mean()), residuals


def refine_coarse_lag_s(eeg_proxy, ecg, coarse_lag_s, params, search_margin_s,
                        agreement_tol_s, min_agreement_fraction, detrend=False, skip_out_of_range=False):
    """Refine the coarse lag with one wide-margin tracking pass.

    The start-time based lag can be off by several seconds (the EEG computer
    and the BI-Vital clocks are not synchronised). Offsets are tracked with a
    wide search margin and their median becomes the refined lag, which is then
    accurate enough for the narrow-margin tracking of `fit_device_sync`.

    Parameters
    ----------
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy at ``params.fs_hz``.
    ecg : numpy.ndarray
        Filtered ECG at ``params.fs_hz``.
    coarse_lag_s : float
        Lag from the recording start times (ECG-array time).
    params : TrackingParams
        Window length and hop; its margin is replaced by ``search_margin_s``.
    search_margin_s : float
        Wide search range (+/-) around ``coarse_lag_s``.
    agreement_tol_s : float
        Windows within this distance of the median count as consistent.
    min_agreement_fraction : float
        Minimum fraction of consistent windows for the refinement to be accepted.
    detrend : bool, optional
        Measure the agreement against a robust straight line through the offsets instead of
        their median, so that a clock drift does not count as disagreement.
    skip_out_of_range : bool, optional
        Skip tracking windows whose search segment lies outside the ECG.

    Returns
    -------
    refined_lag_s : float
        Median tracked offset (ECG-array time).
    tracks : pandas.DataFrame
        Wide-margin tracking result (columns ``eeg_t``, ``offset``, ``r``).

    Raises
    ------
    ValueError
        If the wide search window does not fit in the ECG, or too few windows
        agree on the offset.
    """
    if (coarse_lag_s - search_margin_s) * params.fs_hz < 0:
        raise ValueError(
            f'coarse lag {coarse_lag_s:.1f} s is smaller than the refinement margin {search_margin_s} s'
        )
    tracks = track_offset(
        eeg_proxy, ecg, coarse_lag_s, replace(params, margin_s=search_margin_s), skip_out_of_range
    )
    if detrend:
        refined_lag_s, residuals = line_through_offsets(tracks)
        agreeing = np.abs(residuals) < agreement_tol_s
    else:
        refined_lag_s = float(np.median(tracks['offset']))
        agreeing = np.abs(tracks['offset'] - refined_lag_s) < agreement_tol_s
    if agreeing.mean() < min_agreement_fraction:
        raise ValueError(
            f'only {int(agreeing.sum())} of {len(tracks)} windows agree within {agreement_tol_s} s; '
            f'coarse-lag refinement failed (ECG/EEG cardiac signal too weak or margin too small)'
        )
    return refined_lag_s, tracks


def rank_channels_by_cardiac_match(channel_matrix, channel_names, ecg, coarse_lag_s, params,
                                   search_margin_s, agreement_tol_s, detrend=False, skip_out_of_range=False):
    """Rank EEG channels by how well they match the ECG in a wide-margin pass.

    Every channel is tracked against the ECG like the final proxy would be
    (windowed cross-correlation, margin ``search_margin_s``). A channel's
    score is its median peak ``|r|``; its offset is the median tracked offset,
    its agreement the fraction of windows within ``agreement_tol_s`` of that
    median, and its polarity the sign of the median signed correlation.

    Parameters
    ----------
    channel_matrix : numpy.ndarray
        Standardised channels, shape ``(n_channels, n_samples)`` at ``params.fs_hz``.
    channel_names : list of str
        Names matching the rows of ``channel_matrix``.
    ecg : numpy.ndarray
        Filtered ECG at ``params.fs_hz``.
    coarse_lag_s : float
        Lag from the recording start times (ECG-array time).
    params : TrackingParams
        Window length and hop; its margin is replaced by ``search_margin_s``.
    search_margin_s : float
        Wide search range (+/-) around ``coarse_lag_s``.
    agreement_tol_s : float
        Tolerance for the within-channel offset agreement.
    detrend : bool, optional
        Judge the agreement against a robust line through the channel's offsets (clock drift
        is then not disagreement) and report the line value at the recording centre as
        ``median_offset_s``.
    skip_out_of_range : bool, optional
        Skip tracking windows whose search segment lies outside the ECG.

    Returns
    -------
    pandas.DataFrame
        One row per channel with columns ``channel``, ``median_r``,
        ``median_offset_s``, ``agreement`` and ``polarity``, sorted by
        ``median_r`` (descending).
    """
    wide_params = replace(params, margin_s=search_margin_s)
    rows = []
    for name, channel in zip(channel_names, channel_matrix):
        tracks = track_offset(channel, ecg, coarse_lag_s, wide_params, skip_out_of_range)
        if detrend:
            median_offset_s, residuals = line_through_offsets(tracks)
            agreement = float((np.abs(residuals) < agreement_tol_s).mean())
        else:
            median_offset_s = float(tracks['offset'].median())
            agreement = float((np.abs(tracks['offset'] - median_offset_s) < agreement_tol_s).mean())
        polarity = float(np.sign(tracks['r_signed'].median()) or 1.0)
        rows.append(dict(
            channel=name,
            median_r=float(tracks['r'].median()),
            median_offset_s=median_offset_s,
            agreement=agreement,
            polarity=polarity,
        ))
    return pd.DataFrame(rows).sort_values('median_r', ascending=False).reset_index(drop=True)


def select_proxy_channels(ranking, n_select, agreement_tol_s, min_agreement):
    """Pick the best channels whose offsets agree with each other.

    Channels whose own windows disagree (``agreement < min_agreement``) are
    discarded. The consensus offset is the median offset of the ``n_select``
    best remaining channels; channels farther than ``agreement_tol_s`` from it
    are dropped, and the ``n_select`` best of the rest are returned.

    Parameters
    ----------
    ranking : pandas.DataFrame
        Output of `rank_channels_by_cardiac_match`.
    n_select : int
        Number of channels to keep.
    agreement_tol_s : float
        Maximum offset distance from the consensus.
    min_agreement : float
        Minimum within-channel window agreement.

    Returns
    -------
    selected : pandas.DataFrame
        Rows of ``ranking`` for the chosen channels (best first; may be
        fewer than ``n_select``).
    consensus_offset_s : float
        Median offset of the best candidates (ECG-array time).

    Raises
    ------
    ValueError
        If no channel has consistent offsets or none agrees with the consensus.
    """
    eligible = ranking[ranking['agreement'] >= min_agreement]
    if eligible.empty:
        raise ValueError('no EEG channel has consistent tracked offsets; cardiac artefact not detectable')
    consensus_offset_s = float(eligible.head(n_select)['median_offset_s'].median())
    agreeing = eligible[np.abs(eligible['median_offset_s'] - consensus_offset_s) < agreement_tol_s]
    if agreeing.empty:
        raise ValueError('the best EEG channels disagree on the offset; cardiac artefact not detectable')
    return agreeing.head(n_select), consensus_offset_s


def robust_line(x, y, threshold):
    """Fit a straight line robust to outliers.

    A Theil-Sen line is fitted first; points within ``threshold`` of it are
    then refitted by least squares.

    Parameters
    ----------
    x, y : numpy.ndarray
        Data points.
    threshold : float
        Maximum absolute deviation from the Theil-Sen line (units of ``y``).

    Returns
    -------
    coefficients : numpy.ndarray
        ``[slope, intercept]`` from the least-squares refit.
    inlier_mask : numpy.ndarray
        Boolean mask of the points used in the refit.

    Raises
    ------
    ValueError
        If fewer than two points are within ``threshold`` of the Theil-Sen line.
    """
    theil_slope, theil_intercept = stats.theilslopes(y, x)[:2]
    inlier_mask = np.abs(y - (theil_intercept + theil_slope * x)) < threshold
    if inlier_mask.sum() < 2:
        raise ValueError(
            f'robust_line: only {int(inlier_mask.sum())} of {len(x)} points lie within {threshold} of the '
            f'Theil-Sen line; the tracked offsets are inconsistent (typically the coarse lag is off by '
            f'more than the search margin)'
        )
    coefficients = np.polyfit(x[inlier_mask], y[inlier_mask], 1)
    return coefficients, inlier_mask


def fit_device_sync(device, eeg_proxy, coarse_lag_s, params, fit_threshold_s):
    """Fit the clock model of one device to the EEG.

    Parameters
    ----------
    device : cardiac.DeviceRecording
        BI-Vital recording.
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy of the device's wearer.
    coarse_lag_s : float
        Coarse EEG-to-ECG lag in seconds.
    params : TrackingParams
        Offset tracking parameters.
    fit_threshold_s : float
        Outlier threshold of `robust_line` (s).

    Returns
    -------
    SyncFit
        Offset, drift, inlier mask and the tracked offsets (device time).
    """
    tracks = track_offset(eeg_proxy, device.ecg, coarse_lag_s, params)
    tracks['offset'] += device.first_sample_time_s          # ECG-array time -> device 'Time [s]'
    (slope, intercept_s), used_mask = robust_line(
        tracks['eeg_t'].to_numpy(), tracks['offset'].to_numpy(), fit_threshold_s
    )
    return SyncFit(slope=slope, intercept_s=intercept_s, used_mask=used_mask, tracks=tracks)


def device_time_to_eeg_time(device_time_s, fit):
    """Convert device time to EEG time by inverting the clock model.

    Parameters
    ----------
    device_time_s : numpy.ndarray
        Device time in seconds.
    fit : SyncFit
        Clock fit of that device.

    Returns
    -------
    numpy.ndarray
        ``(t_dev - a) / (1 + b)``; negative before the EEG started.
    """
    return (device_time_s - fit.intercept_s) / (1 + fit.slope)


def eeg_sample_to_device_time(sample_index, fit, eeg_fs_hz):
    """Convert an EEG sample index to device time using the clock model.

    Parameters
    ----------
    sample_index : int
        0-based EEG sample index.
    fit : SyncFit
        Clock fit of the device.
    eeg_fs_hz : float
        EEG sampling rate.

    Returns
    -------
    float
        Device time in seconds.
    """
    return fit.intercept_s + (1 + fit.slope) * sample_index / eeg_fs_hz


def summarize_fit(device_name, fit):
    """Collect headline numbers of a clock fit.

    Parameters
    ----------
    device_name : str
        Device name.
    fit : SyncFit
        Clock fit.

    Returns
    -------
    dict
        Offset at EEG t=0, drift in ms/min, window counts and the residual
        standard deviation (ms) of the used windows.
    """
    tracks = fit.tracks[fit.used_mask]
    predicted = np.polyval([fit.slope, fit.intercept_s], tracks['eeg_t'])
    residual_sd_ms = 1e3 * np.std(tracks['offset'] - predicted)
    return dict(
        device=device_name,
        offset_at_eeg_t0_s=fit.intercept_s,
        drift_ms_per_min=fit.slope * 6e4,
        n_win=len(fit.tracks),
        n_used=int(fit.used_mask.sum()),
        resid_sd_ms=residual_sd_ms,
    )


@dataclass
class WindowCorrelations:
    """Cross-correlation curves of the tracking windows around the fitted alignment.

    Attributes
    ----------
    lags_ms : numpy.ndarray
        Lag axis (ms) relative to the position predicted by the fitted line;
        positive means the ECG is later than predicted.
    curves : numpy.ndarray
        Correlation curves, shape ``(n_windows, len(lags_ms))``.
    centres_s : numpy.ndarray
        Window centres (EEG s).
    selected_lag_ms : numpy.ndarray
        Lag of the peak selected by the tracking, per window.
    selected_value : numpy.ndarray
        Correlation of the curve at the selected peak, per window.
    used_mask : numpy.ndarray
        Windows kept in the final fit.
    """

    lags_ms: np.ndarray
    curves: np.ndarray
    centres_s: np.ndarray
    selected_lag_ms: np.ndarray
    selected_value: np.ndarray
    used_mask: np.ndarray


@dataclass
class AlignmentXcorr:
    """Cross-correlation of the aligned ECG with the EEG proxy over the whole EEG period.

    Attributes
    ----------
    lags_ms : numpy.ndarray
        Lag axis (ms); positive means the ECG is delayed relative to the proxy.
    correlation : numpy.ndarray
        Correlation coefficient per lag.
    peak_lag_ms : float
        Sub-sample lag of the peak (0 = perfectly aligned).
    peak_correlation : float
        Signed correlation at the peak.
    """

    lags_ms: np.ndarray
    correlation: np.ndarray
    peak_lag_ms: float
    peak_correlation: float


def window_correlation_curves(eeg_proxy, ecg, fit, first_sample_time_s, params, margin_s):
    """Recompute each window's cross-correlation around the fitted alignment.

    For every tracking window the correlation curve is centred on the
    position predicted by the fitted line, so the lag axis reads directly as
    the misalignment in ms. The peak selected by the tracking is reported on
    the same axis (its distance from 0 is the residual of that window).

    Parameters
    ----------
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy at ``params.fs_hz``.
    ecg : numpy.ndarray
        Filtered ECG at ``params.fs_hz``.
    fit : SyncFit
        Clock fit whose tracking windows are inspected.
    first_sample_time_s : float
        Device time of the first ECG sample (converts the fit to ECG-array time).
    params : TrackingParams
        Window length, hop and sampling rate used in the tracking.
    margin_s : float
        One-sided lag range of the displayed curves.

    Returns
    -------
    WindowCorrelations
        Curves, lag axis and the selected peaks.
    """
    fs = params.fs_hz
    window_len = int(params.window_s * fs)
    margin_len = int(margin_s * fs)
    curves, selected_lags, selected_values = [], [], []
    for centre_s, offset_device_s in zip(fit.tracks['eeg_t'], fit.tracks['offset']):
        eeg_start_idx = int(round((centre_s - params.window_s / 2) * fs))
        predicted_offset_s = fit.intercept_s + fit.slope * centre_s - first_sample_time_s
        centre_lag_samples = int(round(predicted_offset_s * fs))
        correlation, _ = correlation_around_lag(
            eeg_proxy, ecg, eeg_start_idx, centre_lag_samples, window_len, margin_len
        )
        selected_lag_ms = 1e3 * ((offset_device_s - first_sample_time_s) - centre_lag_samples / fs)
        lags_ms = 1e3 * np.arange(-margin_len, margin_len + 1) / fs
        curves.append(correlation)
        selected_lags.append(selected_lag_ms)
        selected_values.append(np.interp(selected_lag_ms, lags_ms, correlation))
    return WindowCorrelations(
        lags_ms=1e3 * np.arange(-margin_len, margin_len + 1) / fs,
        curves=np.array(curves),
        centres_s=fit.tracks['eeg_t'].to_numpy(),
        selected_lag_ms=np.array(selected_lags),
        selected_value=np.array(selected_values),
        used_mask=fit.used_mask,
    )


def map_ecg_to_eeg_grid(device, intercept_s, slope, n_samples, fs_hz):
    """Interpolate a device ECG onto the EEG sample grid for a given clock model.

    Parameters
    ----------
    device : cardiac.DeviceRecording
        BI-Vital recording.
    intercept_s : float
        Device time ``a`` (s) of EEG t=0.
    slope : float
        Relative clock drift ``b`` in ``t_dev = a + (1 + b) * t_eeg``.
    n_samples : int
        Number of EEG samples (at ``fs_hz``).
    fs_hz : float
        EEG analysis sampling rate.

    Returns
    -------
    numpy.ndarray
        ECG at EEG times ``0, 1/fs_hz, ...``.
    """
    eeg_time_s = np.arange(n_samples) / fs_hz
    return np.interp(intercept_s + (1 + slope) * eeg_time_s, device.time_s, device.ecg)


def resample_ecg_to_eeg_time(device, fit, n_samples, fs_hz):
    """Interpolate a device ECG onto the EEG sample grid using the clock fit.

    Parameters
    ----------
    device : cardiac.DeviceRecording
        BI-Vital recording.
    fit : SyncFit
        Clock fit of the device.
    n_samples : int
        Number of EEG samples (at ``fs_hz``).
    fs_hz : float
        EEG analysis sampling rate.

    Returns
    -------
    numpy.ndarray
        ECG at EEG times ``0, 1/fs_hz, ...``.
    """
    return map_ecg_to_eeg_grid(device, fit.intercept_s, fit.slope, n_samples, fs_hz)


def alignment_xcorr(eeg_proxy, ecg_on_eeg, fs_hz, max_lag_s):
    """Cross-correlate the aligned ECG with the EEG proxy over the whole EEG period.

    After a correct alignment the peak sits at 0 ms.

    Parameters
    ----------
    eeg_proxy : numpy.ndarray
        EEG cardiac proxy.
    ecg_on_eeg : numpy.ndarray
        ECG on the EEG grid (`resample_ecg_to_eeg_time`), same length.
    fs_hz : float
        Sampling rate.
    max_lag_s : float
        One-sided lag range.

    Returns
    -------
    AlignmentXcorr
        Curve and its sub-sample peak.
    """
    lags_ms, correlation = normalized_xcorr(ecg_on_eeg, eeg_proxy, fs_hz, max_lag_s)
    peak_lag_ms, peak_correlation = curve_peak(lags_ms, correlation, fs_hz)
    return AlignmentXcorr(lags_ms, correlation, peak_lag_ms, peak_correlation)
