"""Build the table of BI-Vital signals aligned to the EEG amplifier clock."""
import numpy as np
import pandas as pd

from .clocksync import device_time_to_eeg_time


def interpolate_to_grid(grid_s, source_time_s, values, max_gap_s):
    """Linearly interpolate irregularly present samples onto a time grid.

    Grid points outside the sampled range, or inside a gap between two valid
    samples longer than ``max_gap_s``, are set to NaN instead of being bridged.

    Parameters
    ----------
    grid_s : numpy.ndarray
        Target (EEG) time grid in seconds, increasing.
    source_time_s : numpy.ndarray
        Times of the source samples (same time base as ``grid_s``); NaN values
        in ``values`` are dropped together with their time.
    values : numpy.ndarray
        Source values, same length as ``source_time_s``.
    max_gap_s : float
        Longest gap between neighbouring valid samples that is interpolated.

    Returns
    -------
    numpy.ndarray
        Interpolated values on ``grid_s`` (NaN where not covered; all NaN if ``values`` has no valid sample).
    """
    valid = ~np.isnan(values)
    valid_time_s = source_time_s[valid]
    valid_values = values[valid]
    if not valid.any():                      # signal absent in this recording (e.g. no accelerometer)
        return np.full(len(grid_s), np.nan)
    interpolated = np.interp(grid_s, valid_time_s, valid_values)

    right = np.searchsorted(valid_time_s, grid_s, side='left')
    inside = (right > 0) & (right < len(valid_time_s))
    gap_s = np.full(len(grid_s), np.inf)
    gap_s[inside] = valid_time_s[right[inside]] - valid_time_s[right[inside] - 1]
    on_sample = np.isin(grid_s, valid_time_s)
    interpolated[(gap_s > max_gap_s) & ~on_sample] = np.nan
    interpolated[(grid_s < valid_time_s[0]) | (grid_s > valid_time_s[-1])] = np.nan
    return interpolated


def build_synced_table(grid_s, tables, fits, column_suffix, max_gap_s):
    """Align ECG and accelerometer of each BI-Vital device to the EEG time grid.

    Parameters
    ----------
    grid_s : numpy.ndarray
        EEG time grid in seconds (``eeg_t`` column).
    tables : dict of str to pandas.DataFrame
        Raw BI-Vital tables per device (columns ``t``, ``ecg``, ``ax``, ``ay``, ``az``).
    fits : dict of str to clocksync.SyncFit
        Clock fit per device.
    column_suffix : dict of str to str
        Device name -> suffix of its columns (e.g. ``'child' -> 'ch'``).
    max_gap_s : dict of str to float
        Longest interpolated gap for ``'ecg'`` and ``'acc'``.

    Returns
    -------
    pandas.DataFrame
        ``eeg_t`` plus ``ecg_aligned_<s>`` and ``acc_{x,y,z}_<s>`` per device.
    """
    columns = {'eeg_t': grid_s}
    for device_name, suffix in column_suffix.items():
        table = tables[device_name]
        eeg_time_s = device_time_to_eeg_time(table['t'].to_numpy(), fits[device_name])
        columns[f'ecg_aligned_{suffix}'] = interpolate_to_grid(
            grid_s, eeg_time_s, table['ecg'].to_numpy(), max_gap_s['ecg']
        )
        for axis in ('x', 'y', 'z'):
            columns[f'acc_{axis}_{suffix}'] = interpolate_to_grid(
                grid_s, eeg_time_s, table[f'a{axis}'].to_numpy(), max_gap_s['acc']
            )
    return pd.DataFrame(columns)
