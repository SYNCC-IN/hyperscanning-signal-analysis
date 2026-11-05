"""Import of a Heidelberg dyad (BrainVision EEG + two BI-Vital ECG recorders) into `MultimodalData`.

`create_multimodal_data_heidelberg` is the Heidelberg counterpart of
`dataloader.create_multimodal_data`. The returned `MultimodalData` has the same
layout as the Warsaw one (one DataFrame, one common sampling rate, ``events`` dict
and ``events`` column), so `export.export_chunk_to_xarray` works on it unchanged.

Stages
------
1. read EEG (``.eeg``/``.vhdr``), markers (``.vmrk``) and wall-clock start;
2. synchronise BI-Vital recorders to the EEG clock (`bivital_sync.synchronize_bivital`);
   ``<dyad>_synced_signals.csv`` is written as a by-product when ``sync_output_dir`` is given;
3. filter EEG at the native rate (as in Warsaw: 50 Hz notch + Butterworth band-pass, zero phase);
4. resample EEG to the Warsaw rate (1000 Hz -> 128 Hz, exact 16/125 polyphase);
5. ECG, IBI and RMSSD on the same 128 Hz grid (`hrv_sync.compute_hrv_on_grid`);
6. events from the stimulus markers.

Time axis: seconds from the start of the EEG recording (``time`` column).

Triggers (BrainVision ``Stimulus`` markers; the ``S<n>0`` marker ends the event)::

    S1 / S10 Incredibles   S2 / S20 Brave   S3 / S30 Peppa
    S4 / S40 Talk_1        S5 / S50 Talk_2
"""
import re
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .bivital_sync import SyncParams, synchronize_bivital
from .data_structures import MultimodalData
from .dataloader import _apply_filters, _design_eeg_filters
from .hrv_sync import HrvParams, compute_hrv_on_grid
from .load import read_eeg, read_eeg_start, read_markers, read_vhdr_channels

# ---- Constants ----
MEMBER_SUFFIX = {'ch': '_C', 'cg': '_P'}                 # EEG channel suffix per member (child / parent=caregiver)
DEVICE_MEMBER = {'child': 'ch', 'adult': 'cg'}
DEVICE_OF_MEMBER = {'ch': 'child', 'cg': 'adult'}        # BI-Vital device worn by each member
CHANNEL_RENAME = {'FP1': 'Fp1', 'FP2': 'Fp2'}            # inconsistent upper-case labels in the recording
NON_EEG_CHANNELS = ('VEOG',)                             # exported, but not EEG (excluded from CAR, interpolation, common set)
# 19 scalp channels shared with the Warsaw montage (modern 10-20 labels, mastoids excluded).
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz',
    'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]
# Event name -> (start marker, end marker).
EVENT_MARKERS = {
    'Incredibles': ('S1', 'S10'),
    'Brave': ('S2', 'S20'),
    'Peppa': ('S3', 'S30'),
    'Talk_1': ('S4', 'S40'),
    'Talk_2': ('S5', 'S50'),
}
DEFAULT_EVENT_DURATION_S = {'Incredibles': 60.0, 'Brave': 60.0, 'Peppa': 60.0, 'Talk_1': 180.0, 'Talk_2': 180.0}
# Raw file names, e.g. ``synccin_LAKA36_46.vhdr`` (EEG) and ``LAKA36_46-child.csv`` / ``JEAN90_21_adult.csv``
# (BI-Vital; separator ``-`` or ``_``; a recording split into parts is ``LAEL07_25-adult-01.csv``, ``...-02.csv``):
# the leading part is ignored, the dyad number is the digit group after the last underscore of the head.
EEG_FILE_PATTERN = re.compile(r'^(?P<head>.+)_(?P<number>\d+)\.vhdr$', re.I)
BIVITAL_FILE_PATTERN = re.compile(
    r'^(?P<head>.+)_(?P<number>\d+)[-_](?P<device>child|adult)(?:[-_](?P<part>\d+))?\.csv$', re.I
)
EEG_REFERENCE_NOTE = 'Recorded against TP9 (reference electrode, not part of the recording)'


def parse_dyad_id(dyad_id):
    """``'H032'`` or ``'H_032'`` -> ``'H_032'`` (site letter + three digits)."""
    match = re.fullmatch(r'([A-Za-z])_?(\d+)', dyad_id)
    if match is None:
        raise ValueError(f'Cannot parse dyad id {dyad_id!r}; expected e.g. "H032".')
    return f'{match.group(1).upper()}_{int(match.group(2)):03d}'


def base_channel_name(name, suffix):
    """Strip the member suffix and unify the spelling (``'FP2_C'`` -> ``'Fp2'``)."""
    base = name[:-len(suffix)]
    return CHANNEL_RENAME.get(base, base)


def events_from_markers(markers, raw_fs_hz):
    """Build the ``events`` dict (name -> start/duration in seconds) from stimulus markers.

    Parameters
    ----------
    markers : list of tuple (str, int)
        ``(description, sample_index)`` from `load.read_markers`.
    raw_fs_hz : float
        Sampling rate the marker positions refer to.

    Returns
    -------
    dict
        ``{'Incredibles': {'name', 'start', 'duration'}, ...}`` in seconds from EEG start;
        events whose start marker is missing are skipped, a missing end marker falls
        back to the nominal duration (with a warning).
    """
    first_sample = {}
    for description, sample in markers:
        first_sample.setdefault(description, sample)
    events = {}
    for name, (start_marker, end_marker) in EVENT_MARKERS.items():
        if start_marker not in first_sample:
            warnings.warn(f'Start marker {start_marker} of {name} not found; event skipped.')
            continue
        start_s = first_sample[start_marker] / raw_fs_hz
        if end_marker in first_sample:
            duration_s = first_sample[end_marker] / raw_fs_hz - start_s
        else:
            duration_s = DEFAULT_EVENT_DURATION_S[name]
            warnings.warn(f'End marker {end_marker} of {name} not found; using {duration_s:.0f} s.')
        events[name] = {'name': name, 'start': float(start_s), 'duration': float(duration_s)}
    return events


def discover_heidelberg_dyads(eeg_dir, ecg_dir):
    """Find the dyads that have a complete set of raw files.

    EEG: ``<anything>_<N>.vhdr`` with ``.eeg`` and ``.vmrk`` of the same name next to it
    (e.g. ``synccin_LAKA36_46.vhdr``); BI-Vital: ``<anything>_<N>-child.csv`` and
    ``<anything>_<N>-adult.csv`` (e.g. ``LAKA36_46-child.csv``). Files are matched by the dyad
    number ``N`` only (the separator before ``child``/``adult`` may be ``-`` or ``_``). A dyad with
    more than one candidate file of a kind is reported as ambiguous, not guessed; a BI-Vital
    recording split into parts (``..._adult-01.csv``, ``..._adult-02.csv``) is reported as split
    (multi-part recordings are not imported yet).

    Parameters
    ----------
    eeg_dir, ecg_dir : str or Path
        Folders with the raw BrainVision and BI-Vital files.

    Returns
    -------
    complete : dict of str to dict
        ``{'H046': {'number': 46, 'eeg_stem': Path, 'bivital': {'child': Path, 'adult': Path}}, ...}``
        sorted by dyad number; the key is the dyad id ``H<NNN>`` (exported as ``H_046``).
    problems : dict of int to str
        Dyad number -> reason why it is not complete (missing or ambiguous files).
    """
    eeg_dir, ecg_dir = Path(eeg_dir), Path(ecg_dir)
    eeg_candidates, ecg_candidates = {}, {}
    for path in sorted(eeg_dir.iterdir()):
        match = EEG_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        stem = path.with_suffix('')
        missing = [ext for ext in ('.eeg', '.vmrk') if not Path(f'{stem}{ext}').exists()]
        eeg_candidates.setdefault(int(match['number']), []).append((stem, missing))
    for path in sorted(ecg_dir.iterdir()):
        match = BIVITAL_FILE_PATTERN.match(path.name)
        if match is not None:
            ecg_candidates.setdefault(int(match['number']), {}).setdefault(match['device'].lower(), []).append(
                (match['part'], path)
            )

    complete, problems = {}, {}
    for number in sorted(set(eeg_candidates) | set(ecg_candidates)):
        reasons = []
        eeg_stem = None
        eeg_found = eeg_candidates.get(number, [])
        if not eeg_found:
            reasons.append('no EEG .vhdr')
        elif len(eeg_found) > 1:
            reasons.append('several EEG recordings: ' + ', '.join(stem.name for stem, _ in eeg_found))
        elif eeg_found[0][1]:
            reasons.append(f'EEG {eeg_found[0][0].name} lacks ' + ', '.join(eeg_found[0][1]))
        else:
            eeg_stem = eeg_found[0][0]
        bivital = {}
        for device in ('child', 'adult'):
            found = ecg_candidates.get(number, {}).get(device, [])
            if not found:
                reasons.append(f'no {device} BI-Vital CSV')
            elif any(part is not None for part, _ in found):
                parts = ', '.join(sorted(part or '-' for part, _ in found))
                reasons.append(f'{device} BI-Vital recording split into parts ({parts}); multi-part import not supported yet')
            elif len(found) > 1:
                reasons.append(f'several {device} CSVs: ' + ', '.join(p.name for _, p in found))
            else:
                bivital[device] = found[0][1]
        if reasons:
            problems[number] = '; '.join(reasons)
        else:
            complete[f'H{number:03d}'] = {'number': number, 'eeg_stem': eeg_stem, 'bivital': bivital}
    return complete, problems


def create_multimodal_data_heidelberg(
    data_dir,
    dyad_id,
    bivital_dir=None,
    eeg_path_stem=None,
    bivital_paths=None,
    sync_output_dir=None,
    lowcut=1.0,
    highcut=64.0,
    filter_type='iir',
    target_fs_hz=128,
    sync_params=None,
    hrv_params=None,
    verbose=True,
):
    """Load one Heidelberg dyad into a `MultimodalData` (EEG, ECG, IBI, RMSSD at ``target_fs_hz``).

    Parameters
    ----------
    data_dir : str or Path or None
        Folder with ``<dyad_id>.eeg/.vhdr/.vmrk`` (and the BI-Vital CSVs unless ``bivital_dir`` is set).
        May be None when ``eeg_path_stem`` and ``bivital_paths`` are given.
    dyad_id : str
        e.g. ``'H032'``; the resulting ``MultimodalData.id`` is ``'H_032'``.
    bivital_dir : str or Path, optional
        Folder with ``<dyad_id>-adult.csv`` and ``<dyad_id>-child.csv`` (default ``data_dir``).
        Missing files are skipped with a warning (EEG-only import).
    eeg_path_stem : str or Path, optional
        Explicit path of the EEG recording without extension (files not named ``<dyad_id>.*``,
        e.g. from `discover_heidelberg_dyads`).
    bivital_paths : dict of str to path, optional
        Explicit ``{'child': ..., 'adult': ...}`` CSV paths; overrides ``bivital_dir``.
    sync_output_dir : str or Path, optional
        When given, the synchronisation by-products (``<dyad_id>_synced_signals.csv``, fit
        summary, channel rankings, offset tracks) are written there.
    lowcut, highcut : float
        EEG band-pass (Hz); Warsaw production values 1.0 and 64.
    filter_type : {'iir', 'fir'}
    target_fs_hz : int
        Output rate (Warsaw: 1024 / 8 = 128 Hz).
    sync_params : bivital_sync.SyncParams, optional
    hrv_params : hrv_sync.HrvParams, optional
    verbose : bool

    Returns
    -------
    MultimodalData
        ``data`` columns: ``time``, ``time_idx``, ``events``, ``EEG_<ch|cg>_<label>`` (incl. ``VEOG``),
        ``ECG_<ch|cg>`` (BI-Vital ADC counts), ``IBI_<ch|cg>`` and ``RMSSD_<ch|cg>`` (ms).
        Extra attributes: ``site``, ``common_channels``, ``sync_info``, ``hrv_info``.
    """
    log = print if verbose else (lambda *args, **kwargs: None)
    sync_params = sync_params or SyncParams()
    hrv_params = hrv_params or HrvParams()
    if eeg_path_stem is not None:
        path_stem = Path(eeg_path_stem)
    elif data_dir is not None:
        path_stem = Path(data_dir) / dyad_id
    else:
        raise ValueError('Give data_dir or eeg_path_stem.')
    raw_fs_hz = sync_params.eeg_raw_fs_hz

    # ---- 1. EEG, markers, start time ----
    channel_names = read_vhdr_channels(path_stem)
    eeg = read_eeg(path_stem, n_channels=len(channel_names))
    markers = read_markers(path_stem)
    eeg_start = read_eeg_start(path_stem)
    events = events_from_markers(markers, raw_fs_hz)
    log('EEG start:', eeg_start, '| events:', {k: (round(v['start'], 2), round(v['duration'], 2)) for k, v in events.items()})

    # ---- 2. BI-Vital synchronisation (by-product: <dyad>_synced_signals.csv) ----
    if bivital_paths is None:
        bivital_root = Path(bivital_dir) if bivital_dir is not None else path_stem.parent
        bivital_paths = {device: bivital_root / f'{dyad_id}-{device}.csv' for device in DEVICE_OF_MEMBER.values()}
    found_paths = {}
    for device in DEVICE_OF_MEMBER.values():
        csv_path = Path(bivital_paths[device]) if device in bivital_paths else None
        if csv_path is not None and csv_path.exists():
            found_paths[device] = csv_path
        else:
            warnings.warn(f'BI-Vital file {csv_path} not found; no ECG/IBI/RMSSD for the {device} device.')
    bivital_paths = found_paths
    sync = None
    if bivital_paths:
        sync = synchronize_bivital(
            eeg, channel_names, eeg_start, bivital_paths, params=sync_params,
            dyad_id=dyad_id, output_dir=sync_output_dir, verbose=verbose,
        )

    # ---- 3. EEG filtering at the native rate ----
    eeg_rows = {}
    names = {'ch': [], 'cg': []}
    for member, suffix in MEMBER_SUFFIX.items():
        for row, name in enumerate(channel_names):
            if name.endswith(suffix):
                names[member].append(name)
                eeg_rows[name] = row
    md = MultimodalData()
    md.fs = raw_fs_hz
    md.id = parse_dyad_id(dyad_id)
    md.eeg_channel_names_ch = list(names['ch'])        # full recording names; needed by the shared filter code
    md.eeg_channel_names_cg = list(names['cg'])
    md.eeg_channel_mapping = dict(eeg_rows)
    filters = _design_eeg_filters(md, lowcut, highcut, filter_type=filter_type)
    _apply_filters(md, filters, eeg)

    # ---- 4. Resample to the Warsaw rate ----
    ratio = Fraction(target_fs_hz, raw_fs_hz).limit_denominator(1000)
    if ratio.numerator / ratio.denominator * raw_fs_hz != target_fs_hz:
        raise ValueError(f'Cannot resample {raw_fs_hz} Hz to {target_fs_hz} Hz with a rational factor.')
    log(f'Resampling EEG {raw_fs_hz} Hz -> {target_fs_hz} Hz (up={ratio.numerator}, down={ratio.denominator})')
    eeg_rs = signal.resample_poly(eeg, ratio.numerator, ratio.denominator, axis=1)
    n_samples = eeg_rs.shape[1]
    grid_s = np.arange(n_samples) / target_fs_hz

    columns = {'time': grid_s, 'time_idx': np.arange(n_samples)}
    for member, suffix in MEMBER_SUFFIX.items():
        for name in names[member]:
            columns[f'EEG_{member}_{base_channel_name(name, suffix)}'] = eeg_rs[eeg_rows[name]]
    del eeg_rs

    # ---- 5. ECG, IBI, RMSSD on the same grid ----
    modalities = ['EEG']
    hrv_info = {}
    if sync is not None:
        for member, device in DEVICE_OF_MEMBER.items():
            if device in sync.failures:
                # synchronisation failed: keep the file layout, ECG/IBI/RMSSD are all NaN
                for prefix in ('ECG', 'IBI', 'RMSSD'):
                    columns[f'{prefix}_{member}'] = np.full(n_samples, np.nan)
                hrv_info[member] = {'sync_status': 'failed', 'reason': sync.failures[device]}
                log(f'{member} ({device}): synchronisation failed -> ECG/IBI/RMSSD set to NaN')
                continue
            if device not in sync.fits:
                continue
            hrv = compute_hrv_on_grid(sync.bivital_tables[device], sync.fits[device], member, grid_s, hrv_params)
            columns[f'ECG_{member}'] = hrv['ECG']
            columns[f'IBI_{member}'] = hrv['IBI']
            columns[f'RMSSD_{member}'] = hrv['RMSSD']
            hrv_info[member] = hrv['info']
            log(f'{member} ({device}): {hrv["info"]}')
        if any(f'ECG_{m}' in columns for m in MEMBER_SUFFIX):
            modalities += ['ECG', 'IBI', 'RMSSD']

    # ---- 6. Events column ----
    event_column = np.full(n_samples, None, dtype=object)
    for event in events.values():
        event_column[(grid_s >= event['start']) & (grid_s <= event['start'] + event['duration'])] = event['name']
    columns['events'] = event_column

    # ---- Assemble ----
    md.data = pd.DataFrame(columns)
    md.fs = float(target_fs_hz)
    md.events = events
    md.modalities = modalities
    md.eeg_channel_names_ch = [base_channel_name(n, MEMBER_SUFFIX['ch']) for n in names['ch']]
    md.eeg_channel_names_cg = [base_channel_name(n, MEMBER_SUFFIX['cg']) for n in names['cg']]
    md.eeg_channel_mapping = {}
    md.references = f'{EEG_REFERENCE_NOTE}; original reference retained at import'
    md.site = 'H'
    md.common_channels = list(COMMON_CHANNELS)
    md.hrv_info = hrv_info
    md.sync_info = {} if sync is None else sync.summary.to_dict(orient='records')
    md.sync_failures = {} if sync is None else {DEVICE_MEMBER[d]: r for d, r in sync.failures.items()}
    md.notes = (
        f'Heidelberg dyad {md.id}; EEG {raw_fs_hz} Hz -> {target_fs_hz} Hz; ECG/IBI/RMSSD from BI-Vital '
        f'(ECG in ADC counts) synchronised to the EEG clock; time = seconds from EEG start.'
    )
    return md
