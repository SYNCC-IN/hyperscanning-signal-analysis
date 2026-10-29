"""Preparation of cardiac signals: EEG channels for the ECG-in-EEG proxy and the BI-Vital ECG."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from .dsp import design_bandpass


@dataclass
class DeviceRecording:
    """Filtered ECG of one BI-Vital device.

    Attributes
    ----------
    name : str
        Device name.
    start_time : pandas.Timestamp
        Wall-clock start time from the CSV header.
    first_sample_time_s : float
        Device time ('Time [s]') of the first valid ECG sample.
    time_s : numpy.ndarray
        Device time of every ECG sample (uniform grid at the ECG rate).
    ecg : numpy.ndarray
        Band-passed, median-centred ECG.
    """

    name: str
    start_time: pd.Timestamp
    first_sample_time_s: float
    time_s: np.ndarray
    ecg: np.ndarray


def preprocess_eeg_channels(eeg, channel_names, selected_channels, raw_fs_hz, target_fs_hz, band_hz, filter_order):
    """Band-pass, resample and standardise selected EEG channels.

    Each selected channel is band-passed (zero-phase), resampled from
    ``raw_fs_hz`` to ``target_fs_hz`` and scaled to unit standard deviation.

    Parameters
    ----------
    eeg : numpy.ndarray
        EEG data of shape ``(n_channels, n_samples)`` at ``raw_fs_hz``.
    channel_names : list of str
        Channel names matching the rows of ``eeg``.
    selected_channels : list of str
        Names of the channels to process.
    raw_fs_hz, target_fs_hz : int
        Original and output sampling rates; ``raw_fs_hz`` must be a multiple
        of ``target_fs_hz``.
    band_hz : tuple of float
        Band-pass cut-off frequencies.
    filter_order : int
        Butterworth order.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(selected_channels), n_samples_at_target_fs)``.
    """
    rows = [channel_names.index(name) for name in selected_channels]
    sos = design_bandpass(raw_fs_hz, band_hz, filter_order)
    filtered = signal.sosfiltfilt(sos, eeg[rows], axis=1)
    resampled = signal.resample_poly(filtered, 1, raw_fs_hz // target_fs_hz, axis=1)
    return resampled / resampled.std(axis=1, keepdims=True)


def signed_mean_proxy(channel_matrix, polarities):
    """Average channels after flipping them to a common polarity.

    The cardiac artefact has opposite signs on different electrodes, so a
    plain mean can cancel it. Each channel is multiplied by its polarity
    (+1 or -1) before averaging.

    Parameters
    ----------
    channel_matrix : numpy.ndarray
        Array of shape ``(n_channels, n_samples)``.
    polarities : array_like of float
        One sign per channel.

    Returns
    -------
    numpy.ndarray
        Cardiac proxy of length ``n_samples``.
    """
    return np.mean(channel_matrix * np.asarray(polarities)[:, None], axis=0)


def prepare_device_ecg(name, header, table, fs_hz, band_hz, filter_order):
    """Clean and band-pass the ECG of a BI-Vital device.

    The ECG is resampled onto a uniform grid starting at the first valid
    sample (samples that are missing or lost are filled by linear
    interpolation), so that array index and device time are related exactly
    by ``time = first_sample_time_s + index / fs_hz``. Simply dropping the
    rows without ECG values would shorten the array and shift the time axis
    by 5 ms per dropped sample. The median is then removed and the signal is
    band-passed (zero-phase).

    Parameters
    ----------
    name : str
        Device name stored in the result.
    header : dict
        Header from `load.read_bivital` (needs the key ``'start'``).
    table : pandas.DataFrame
        Sample table from `load.read_bivital` (needs ``'t'`` and ``'ecg'``).
    fs_hz : float
        ECG sampling rate.
    band_hz : tuple of float
        Band-pass cut-off frequencies.
    filter_order : int
        Butterworth order.

    Returns
    -------
    DeviceRecording
        Filtered ECG with its device time axis and header start time.
    """
    valid = table.dropna(subset=['ecg'])
    valid_time_s = valid['t'].to_numpy()
    n_samples = int(np.floor((valid_time_s[-1] - valid_time_s[0]) * fs_hz)) + 1
    time_s = valid_time_s[0] + np.arange(n_samples) / fs_hz               # uniform grid: index <-> device time
    raw_ecg = np.interp(time_s, valid_time_s, valid['ecg'].to_numpy())    # fills dropped / lost samples
    sos = design_bandpass(fs_hz, band_hz, filter_order)
    filtered = signal.sosfiltfilt(sos, raw_ecg - np.median(raw_ecg))
    return DeviceRecording(
        name=name,
        start_time=header['start'],
        first_sample_time_s=valid_time_s[0],
        time_s=time_s,
        ecg=filtered,
    )
