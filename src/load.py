"""Loaders for BrainVision EEG recordings and BI-Vital CSV files (Heidelberg dyads).

All EEG loaders take the *path stem* of a BrainVision recording, i.e. the path
without extension (``.../H032`` for ``H032.vhdr``, ``H032.eeg``, ``H032.vmrk``).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Configuration ----
EEG_DTYPE = '<i2'                       # BrainVision multiplexed INT16, little endian
DEFAULT_N_CHANNELS = 64
DEFAULT_RESOLUTION_UV = 0.1             # microvolts per LSB
BIVITAL_N_HEADER_ROWS = 2               # rows skipped before the numeric table
BIVITAL_COLUMNS = ['t', 'hr', 'rr', 'rpk', 'ecg', 'q', 'ax', 'ay', 'az']   # columns of the returned table
# BI-Vital firmware versions write different column sets (e.g. 9 columns with ``Heartrate``/``R-Peak``/
# ``ECG-Quality``, or 12 columns with angular velocity and magnetic field), so columns are identified
# by the start of their header name (lower case; ``ecg-quality`` must be tested before ``ecg``).
BIVITAL_HEADER_PREFIXES = [
    ('time', 't'),
    ('heartrate', 'hr'),
    ('rr-interval', 'rr'),
    ('r-peak', 'rpk'),
    ('ecg-quality', 'q'),
    ('ecg', 'ecg'),
    ('acceleration_x', 'ax'),
    ('acceleration_y', 'ay'),
    ('acceleration_z', 'az'),
]

VHDR_CHANNEL_PATTERN = re.compile(r'^Ch\d+=([^,]+),', re.M)
VMRK_STIMULUS_PATTERN = re.compile(r'Mk\d+=Stimulus,([^,]+),(\d+)')
VMRK_NEW_SEGMENT_PATTERN = re.compile(
    r'^Mk\d+=New Segment,[^,]*,[^,]*,[^,]*,[^,]*,(\d{14})(\d{6})', re.M
)
BIVITAL_START_PATTERN = re.compile(r'Start Time (\S+ \S+)')


def _with_extension(path_stem, extension):
    """Return ``path_stem`` + ``extension`` as a `Path` (safe for stems containing dots)."""
    return Path(f'{path_stem}{extension}')


def read_vhdr_channels(path_stem):
    """Read channel names from a BrainVision header (``.vhdr``).

    Parameters
    ----------
    path_stem : str or Path
        Recording path without extension.

    Returns
    -------
    list of str
        Channel names in file order.
    """
    header_path = _with_extension(path_stem, '.vhdr')
    print(f'Reading VHDR channels from {header_path}')
    header_text = header_path.read_text(encoding='utf-8')
    return [match.group(1) for match in VHDR_CHANNEL_PATTERN.finditer(header_text)]


def read_eeg(path_stem, n_channels=DEFAULT_N_CHANNELS, resolution_uv=DEFAULT_RESOLUTION_UV):
    """Read multiplexed INT16 BrainVision data (``.eeg``).

    The file stores samples channel-interleaved, so the flat array is reshaped
    to ``(n_samples, n_channels)`` and transposed.

    Parameters
    ----------
    path_stem : str or Path
        Recording path without extension.
    n_channels : int, optional
        Number of channels in the recording.
    resolution_uv : float, optional
        Amplitude of one integer step in microvolts.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_channels, n_samples)`` in microvolts.
    """
    eeg_path = _with_extension(path_stem, '.eeg')
    print(f'Reading EEG from {eeg_path}')
    raw = np.fromfile(eeg_path, dtype=EEG_DTYPE).reshape(-1, n_channels).T
    return raw * resolution_uv


def read_markers(path_stem):
    """Read stimulus markers from a BrainVision marker file (``.vmrk``).

    Parameters
    ----------
    path_stem : str or Path
        Recording path without extension.

    Returns
    -------
    list of tuple (str, int)
        ``(description, sample_index)`` per stimulus marker, with the
        description stripped of spaces and the sample converted from
        1-based to 0-based indexing.
    """
    marker_path = _with_extension(path_stem, '.vmrk')
    print(f'Reading markers from {marker_path}')
    markers = []
    with open(marker_path, encoding='utf-8') as handle:
        for line in handle:
            match = VMRK_STIMULUS_PATTERN.match(line)
            if match:
                description = match.group(1).replace(' ', '')
                markers.append((description, int(match.group(2)) - 1))
    return markers


def read_eeg_start(path_stem):
    """Read the wall-clock start of the EEG recording from the marker file.

    Parses the ``New Segment`` marker, whose last field has the form
    ``YYYYMMDDhhmmssuuuuuu`` (date, time and microseconds), e.g.
    ``Mk1=New Segment,,1,1,0,20260807162504225756``.

    Parameters
    ----------
    path_stem : str or Path
        Recording path without extension.

    Returns
    -------
    pandas.Timestamp
        Timezone-naive start time of the recording.

    Raises
    ------
    ValueError
        If no ``New Segment`` marker is found.
    """
    marker_path = _with_extension(path_stem, '.vmrk')
    match = VMRK_NEW_SEGMENT_PATTERN.search(marker_path.read_text(encoding='utf-8'))
    if match is None:
        raise ValueError(f'No "New Segment" marker found in {marker_path}')
    seconds_part, microseconds_part = match.groups()
    start = pd.to_datetime(seconds_part, format='%Y%m%d%H%M%S')
    return start + pd.Timedelta(microseconds=int(microseconds_part))


def read_bivital(csv_path):
    """Read a BI-Vital CSV export (any firmware version).

    The first line must contain ``Start Time <date> <time>``; the third line holds the column
    names. Columns are recognised by name (see ``BIVITAL_HEADER_PREFIXES``), so files with different
    column sets and orders are read alike; columns missing in a file (e.g. ``Heartrate``,
    ``R-Peak`` and ``ECG-Quality`` in older files) are returned as NaN, and unused columns
    (angular velocity, magnetic field) are dropped.

    Parameters
    ----------
    csv_path : str or Path
        Path to the BI-Vital CSV file.

    Returns
    -------
    header : dict
        ``{'start': pandas.Timestamp, 'hdr': str}`` with the device start
        time and the raw header line.
    table : pandas.DataFrame
        Samples with columns ``t`` (device time in s), ``hr``, ``rr``,
        ``rpk``, ``ecg``, ``q``, ``ax``, ``ay``, ``az``.

    Raises
    ------
    ValueError
        If the file has no ``Start Time`` or lacks the time or ECG column.
    """
    with open(csv_path) as handle:
        header_line = handle.readline()
    start_match = BIVITAL_START_PATTERN.search(header_line)
    if start_match is None:
        raise ValueError(f'No "Start Time" in the first line of {csv_path}')
    raw = pd.read_csv(csv_path, skiprows=BIVITAL_N_HEADER_ROWS, skipinitialspace=True)
    renamed = {}
    for column in raw.columns:
        name = str(column).strip().lower()
        for prefix, target in BIVITAL_HEADER_PREFIXES:
            if name.startswith(prefix):
                renamed.setdefault(target, column)
                break
    for required in ('t', 'ecg'):
        if required not in renamed:
            raise ValueError(f'Column for "{required}" not found in {csv_path}; columns: {list(raw.columns)}')
    table = pd.DataFrame({
        target: pd.to_numeric(raw[renamed[target]], errors='coerce') if target in renamed else np.nan
        for target in BIVITAL_COLUMNS
    })
    header = {'start': pd.Timestamp(start_match.group(1)), 'hdr': header_line.strip()}
    return header, table
