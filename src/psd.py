"""Power spectral density computation for EEG channel data."""

import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.stats import zscore


def compute_psd_multitaper(data, sfreq, fmin, fmax, bandwidth):
    """Compute multitaper power spectral density for multichannel EEG data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        EEG signal.
    sfreq : float
        Sampling frequency in Hz.
    fmin : float
        Lowest frequency of interest in Hz.
    fmax : float
        Highest frequency of interest in Hz.
    bandwidth : float
        Multitaper frequency smoothing bandwidth in Hz.

    Returns
    -------
    freqs : ndarray, shape (n_freqs,)
        Frequency bins in Hz.
    psd : ndarray, shape (n_channels, n_freqs)
        Power spectral density per channel.
    """
    psd, freqs = psd_array_multitaper(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth, verbose=False,
    )
    return freqs, psd


def average_psd_across_conditions(psd_dict):
    """Average PSD arrays across conditions (e.g. movies).

    Parameters
    ----------
    psd_dict : dict
        Mapping of ``{condition_name: psd_array}``, where each ``psd_array`` has
        the same shape (e.g. n_channels x n_freqs).

    Returns
    -------
    ndarray
        Arithmetic mean PSD across conditions, same shape as each input array.
    """
    if not psd_dict:
        raise ValueError('psd_dict is empty; no conditions to average PSD over.')
    return np.mean(np.stack(list(psd_dict.values()), axis=0), axis=0)


def plot_continuous_psd_band(raw_avg, sfreq, fast_cf, fast_bw, title):
    """Plot a continuous ROI signal's PSD with the individualized band shaded.

    Parameters
    ----------
    raw_avg : np.ndarray, shape (n_times,)
        ROI-averaged raw signal, whole continuous chunk.
    sfreq : float
        Sampling frequency in Hz.
    fast_cf : float
        Individualized fast-rhythm center frequency in Hz.
    fast_bw : float
        Half-width of the individualized passband in Hz (``fast_bw / 2``
        already applied by the caller, matching the filter's own convention).
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs, psd = compute_psd_multitaper(raw_avg[np.newaxis, :], sfreq, fmin=1.0, fmax=20.0, bandwidth=1.0)
    figure, axis = plt.subplots(figsize=(3, 3), dpi=100)
    axis.plot(freqs, psd[0])
    axis.axvspan(fast_cf - fast_bw, fast_cf + fast_bw, color="orange", alpha=0.3, label="individual fast band")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_continuous_overlay(node_continuous, names, films_windows, title):
    """Plot every node's continuous downsampled design signal with film windows shaded.

    One row per node (in `names` order), so filter/anti-alias edge effects
    can be checked for every signal feeding the design matrix, regardless of
    how many nodes each role carries.

    Parameters
    ----------
    node_continuous : dict
        ``{node_name: {...}}``, each entry with ``signal_full`` (the
        continuous downsampled signal), ``sfreq``, and ``t0`` (see
        `scripts/stage02_envelopes.py`'s per-node continuous build).
    names : list of str
        Node names, in display order (see `src.design.node_names`).
    films_windows : list of tuple
        ``(film_name, start_s, end_s)`` for every film, to shade as the
        retained (post-segmentation) regions.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axes = plt.subplots(nrows=len(names), sharex=True, figsize=(10, 2.2 * len(names)), dpi=100)
    axes = np.atleast_1d(axes)
    for axis, name in zip(axes, names):
        nc = node_continuous[name]
        time = nc["t0"] + np.arange(nc["signal_full"].size) / nc["sfreq"]
        axis.plot(time, nc["signal_full"])
        for film_name, start_s, end_s in films_windows:
            axis.axvspan(start_s, end_s, color="green", alpha=0.2)
            axis.text(start_s, axis.get_ylim()[1], film_name, fontsize=8, va="top")
        axis.set_ylabel(name)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_design_variable_psd(node_segments, names, fs, title, plot_zscore, psd_bandwidth):
    """Plot the multitaper PSD of each downsampled design variable, to check for aliasing.

    Each variable is z-scored first (plotting only) so an EEG envelope and a
    raw IBI -- which differ by orders of magnitude in physical units -- can be
    compared on one axis; this is what makes it possible to confirm they
    occupy a comparable frequency band. Multitaper (`compute_psd_multitaper`)
    is used instead of a plain periodogram, which is too noisy on a ~60 s
    segment to read.

    Parameters
    ----------
    node_segments : dict
        ``{node_name: array}``, already segmented to one film window.
    names : list of str
        Node names, in plotting order (see `src.design.node_names`).
    fs : float
        Sampling frequency in Hz (Nyquist is ``fs / 2``).
    title : str
    plot_zscore : bool
        Whether to z-score each variable before computing its PSD.
    psd_bandwidth : float
        Multitaper frequency smoothing bandwidth in Hz.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots()
    for name in names:
        signal = node_segments[name]
        if plot_zscore:
            signal = zscore(signal)
        freqs, psd = compute_psd_multitaper(signal[np.newaxis, :], fs, fmin=0.0, fmax=fs / 2, bandwidth=psd_bandwidth)
        axis.plot(freqs, psd[0], label=name)
    axis.axvline(fs / 2, color="black", linestyle="--", label="Nyquist")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (z-scored input)" if plot_zscore else "PSD")
    axis.set_title(title)
    axis.legend(fontsize=7)
    figure.tight_layout()
    return figure
