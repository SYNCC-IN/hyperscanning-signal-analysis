"""Synchronisation of BI-Vital ECG recorders to a BrainVision EEG (Heidelberg dyads).

Model: ``t_dev = a + (1 + b) * t_eeg`` (seconds from recording start; device time is
the CSV column ``Time [s]``), where ``a`` is the device time of EEG t=0 and ``b`` the
relative clock drift. The model is estimated for every device from the cardiac
artefact that the wearer's EEG channels pick up (see `clocksync`).

`synchronize_bivital` is the single entry point. It is used as a stage of
`heidelberg_loader.create_multimodal_data_heidelberg`; the table of ECG and
accelerometer on the EEG time grid (``<dyad>_synced_signals.csv``) and the other
diagnostic tables are written as a by-product when ``output_dir`` is given.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .cardiac import prepare_device_ecg, preprocess_eeg_channels, signed_mean_proxy
from .clocksync import (
    TrackingParams,
    device_time_to_eeg_time,
    estimate_coarse_lag_s,
    fit_device_sync,
    rank_channels_by_cardiac_match,
    refine_coarse_lag_s,
    select_proxy_channels,
    summarize_fit,
)
from .load import read_bivital
from .synced_export import build_synced_table


@dataclass
class SyncParams:
    """Configuration of the EEG <-> BI-Vital synchronisation.

    Attributes mirror the constants of the former ``scripts/heidelberg_sync.py``.
    """

    # ---- sampling and filtering ----
    eeg_raw_fs_hz: int = 1000               # sampling rate of the BrainVision recording
    analysis_fs_hz: int = 200               # common rate of EEG cardiac proxy and ECG
    cardiac_band_hz: tuple = (5, 25)        # band-pass for cardiac artefact / ECG
    filter_order: int = 3                   # Butterworth order (forward-backward)
    # ---- offset tracking ----
    track_window_s: float = 120             # length of each cross-correlation window
    track_step_s: float = 60                # hop between windows
    search_margin_s: float = 4              # +/- search range around the coarse lag
    robust_fit_threshold_s: float = 0.01    # Theil-Sen outlier threshold
    on_failure: str = 'skip'                # 'skip': device without a fit gets no ECG (EEG still exported); 'raise': stop
    wide_search_margin_s: float = 30        # wide search range (channel ranking, coarse-lag refinement)
    fallback_search_margin_s: float = 300   # search range of the 3rd strategy (wall-clocks differing by minutes)
    full_search_margin_s: float = 14400     # search range of the last strategy (effectively the whole ECG)
    coarse_agreement_tol_s: float = 0.1     # windows within this of the median offset are consistent
    coarse_min_agreement: float = 0.5       # minimum fraction of consistent windows
    # ---- devices and cardiac proxy ----
    wearer_channel_suffix: dict = field(default_factory=lambda: {'adult': '_P', 'child': '_C'})
    n_proxy_channels: int = 3               # best channels averaged into the cardiac proxy
    channel_agreement_tol_s: float = 0.05
    channel_min_agreement: float = 0.5
    # ---- export of the synchronised signals (<dyad>_synced_signals.csv) ----
    export_fs_hz: int = 200                 # sampling rate of the exported eeg_t grid
    export_include_pre_eeg: bool = True     # grid also covers BI-Vital data before EEG t=0
    export_column_suffix: dict = field(default_factory=lambda: {'child': 'ch', 'adult': 'cg'})
    export_max_gap_s: dict = field(default_factory=lambda: {'ecg': 0.1, 'acc': 0.5})
    export_float_format: str = '%.6g'


@dataclass
class SyncResult:
    """Everything `synchronize_bivital` produces.

    Attributes
    ----------
    dyad_id : str
    eeg_start : pandas.Timestamp
        Wall-clock start of the EEG recording.
    tracking_params : clocksync.TrackingParams
    recordings : dict of str to cardiac.DeviceRecording
        Band-passed (sync band) ECG per device; used only for the synchronisation.
    bivital_tables : dict of str to pandas.DataFrame
        Raw BI-Vital tables (columns ``t, hr, rr, rpk, ecg, q, ax, ay, az``).
    fits : dict of str to clocksync.SyncFit
    proxies : dict of str to numpy.ndarray
        EEG cardiac proxy per device at ``analysis_fs_hz``.
    rankings : dict of str to pandas.DataFrame
        Channel ranking per device.
    summary : pandas.DataFrame
        One row per device (`clocksync.summarize_fit`).
    synced_table : pandas.DataFrame
        ``eeg_t``, ``ecg_aligned_*`` and ``acc_*`` on the EEG grid at ``export_fs_hz``.
    """

    dyad_id: str
    eeg_start: pd.Timestamp
    tracking_params: TrackingParams
    recordings: dict
    bivital_tables: dict
    fits: dict
    proxies: dict
    rankings: dict
    summary: pd.DataFrame
    synced_table: pd.DataFrame
    failures: dict = field(default_factory=dict)


STRATEGIES = (
    # name, drift-aware agreement, search range, re-centre the refinement on the channel consensus
    ('start-time lag', False, 'wide_search_margin_s', False),
    ('start-time lag, drift-aware', True, 'wide_search_margin_s', False),
    ('wide search, drift-aware', True, 'fallback_search_margin_s', True),
    ('full-range search, drift-aware', True, 'full_search_margin_s', True),
)


def _fit_device_with_fallbacks(device_name, recording, channel_matrix, wearer_channels, eeg_start,
                               tracking_params, params, ranking_path, log):
    """Find the clock fit of one device, trying progressively more tolerant strategies.

    1. ``start-time lag``: the original procedure (offsets within the wide margin around the lag implied
       by the recording start times; channels must agree with their own median offset);
    2. ``start-time lag, drift-aware``: same, but agreement is judged against a straight line through
       the offsets, so a large relative clock drift is not mistaken for a missing cardiac artefact;
    3. ``wide search, drift-aware``: search range of ``fallback_search_margin_s`` around the start-time lag
       (wall-clocks of the EEG computer and the BI-Vital that differ by minutes);
    4. ``full-range search, drift-aware``: the search range covers the whole ECG
       (``full_search_margin_s``; e.g. a BI-Vital clock set to the wrong time zone). Slowest.

    Dyads that work with strategy 1 give exactly the same result as before. The channel ranking of the
    last tried strategy is written to ``ranking_path`` (also when all strategies fail).

    Returns
    -------
    fit, eeg_proxy, ranking, strategy_name, coarse_lag_s, start_time_lag_s

    Raises
    ------
    ValueError
        If every strategy fails; the message lists the reason and the best channels of each.
    """
    failures = []
    raw_lag_s = (eeg_start - recording.start_time).total_seconds() - recording.first_sample_time_s
    for name, detrend, margin_key, recenter in STRATEGIES:
        margin_s = getattr(params, margin_key)
        ranking = None
        try:
            if recenter:
                start_time_lag_s = raw_lag_s                  # window positions outside the ECG are skipped
            else:
                start_time_lag_s = estimate_coarse_lag_s(
                    eeg_start, recording, params.analysis_fs_hz, params.wide_search_margin_s
                )
            ranking = rank_channels_by_cardiac_match(
                channel_matrix, wearer_channels, recording.ecg, start_time_lag_s, tracking_params,
                margin_s, params.channel_agreement_tol_s, detrend=detrend, skip_out_of_range=recenter,
            )
            if ranking_path is not None:
                ranking.assign(strategy=name).to_csv(ranking_path, index=False)
            selected, consensus_offset_s = select_proxy_channels(
                ranking, params.n_proxy_channels, params.channel_agreement_tol_s, params.channel_min_agreement
            )
            description = ', '.join(
                f'{row.channel}({int(row.polarity):+d}, r={row.median_r:.2f})' for row in selected.itertuples()
            )
            log(f'{device_name}: proxy channels {description}; consensus offset {consensus_offset_s:.3f} s [{name}]')
            selected_rows = [wearer_channels.index(channel) for channel in selected['channel']]
            eeg_proxy = signed_mean_proxy(channel_matrix[selected_rows], selected['polarity'].to_numpy())
            centre_lag_s = consensus_offset_s if recenter else start_time_lag_s
            coarse_lag_s, refine_tracks = refine_coarse_lag_s(
                eeg_proxy, recording.ecg, centre_lag_s, tracking_params,
                params.wide_search_margin_s, params.coarse_agreement_tol_s, params.coarse_min_agreement,
                detrend=detrend, skip_out_of_range=recenter,
            )
            log(
                f'{device_name}: refined lag0 = {coarse_lag_s:.3f} s (shift {coarse_lag_s - start_time_lag_s:+.3f} s '
                f'from the start-time lag, {len(refine_tracks)} windows, median r = {refine_tracks["r"].median():.2f})'
            )
            fit = fit_device_sync(recording, eeg_proxy, coarse_lag_s, tracking_params, params.robust_fit_threshold_s)
            if name != STRATEGIES[0][0]:
                log(f'{device_name}: WARNING synchronised only with the fallback strategy "{name}" '
                    f'(drift {fit.slope * 6e4:.2f} ms/min); check the QC of this dyad')
            return fit, eeg_proxy, ranking, name, coarse_lag_s, start_time_lag_s
        except ValueError as error:
            top = ''
            if ranking is not None:
                top = '; best channels: ' + ', '.join(
                    f'{row.channel} (r={row.median_r:.2f}, agreement={row.agreement:.2f})'
                    for row in ranking.head(3).itertuples()
                )
            failures.append(f'[{name}] {error}{top}')
            log(f'{device_name}: strategy "{name}" failed: {error}')
    raise ValueError(
        f'{device_name}: no synchronisation strategy succeeded (start-time lag {raw_lag_s:.1f} s, ECG '
        f'{len(recording.ecg) / params.analysis_fs_hz:.0f} s). ' + ' | '.join(failures)
    )


def synchronize_bivital(eeg, channel_names, eeg_start, bivital_paths, params=None,
                        dyad_id='', output_dir=None, verbose=True):
    """Fit the clock model of every BI-Vital device and align its signals to the EEG.

    Parameters
    ----------
    eeg : numpy.ndarray
        Raw EEG ``(n_channels, n_samples)`` at ``params.eeg_raw_fs_hz`` (unfiltered).
    channel_names : list of str
        Names of the EEG rows; the wearer's channels end with the suffix given in
        ``params.wearer_channel_suffix`` (e.g. ``'TP10_P'``).
    eeg_start : pandas.Timestamp
        Wall-clock start of the EEG (`load.read_eeg_start`).
    bivital_paths : dict of str to path
        Device name (``'adult'``, ``'child'``) -> BI-Vital CSV.
    params : SyncParams, optional
    dyad_id : str
        Used in the names of the by-product files.
    output_dir : path, optional
        When given, ``<dyad>_synced_signals.csv``, ``<dyad>_sync_summary.csv``,
        ``<dyad>_<device>_channel_ranking.csv`` and ``<dyad>_<device>_offset_tracks.csv``
        are written there.
    verbose : bool

    Returns
    -------
    SyncResult
    """
    params = params or SyncParams()
    log = print if verbose else (lambda *args, **kwargs: None)
    tracking_params = TrackingParams(
        fs_hz=params.analysis_fs_hz,
        window_s=params.track_window_s,
        step_s=params.track_step_s,
        margin_s=params.search_margin_s,
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    recordings, tables, fits, proxies, rankings, strategies = {}, {}, {}, {}, {}, {}
    failures = {}
    for device_name, channel_suffix in params.wearer_channel_suffix.items():
        if device_name not in bivital_paths:
            continue
        header, table = read_bivital(bivital_paths[device_name])
        recording = prepare_device_ecg(
            device_name, header, table, params.analysis_fs_hz, params.cardiac_band_hz, params.filter_order
        )
        wearer_channels = [name for name in channel_names if name.endswith(channel_suffix)]
        channel_matrix = preprocess_eeg_channels(
            eeg, channel_names, wearer_channels, params.eeg_raw_fs_hz, params.analysis_fs_hz,
            params.cardiac_band_hz, params.filter_order,
        )
        log(f'{device_name}: BI-Vital start {recording.start_time}, ECG {len(recording.ecg) / params.analysis_fs_hz:.0f} s')
        ranking_path = None if output_dir is None else output_dir / f'{dyad_id}_{device_name}_channel_ranking.csv'
        try:
            fit, eeg_proxy, ranking, strategy, coarse_lag_s, start_time_lag_s = _fit_device_with_fallbacks(
                device_name, recording, channel_matrix, wearer_channels, eeg_start,
                tracking_params, params, ranking_path, log,
            )
        except ValueError as error:
            if params.on_failure != 'skip':
                raise
            failures[device_name] = str(error)
            log(f'{device_name}: WARNING synchronisation failed; no ECG/IBI/RMSSD for this device ({error})')
            continue
        strategies[device_name] = strategy
        fitted_offset_s = fit.intercept_s - recording.first_sample_time_s
        log(
            f'{device_name}: fitted offset at EEG t=0 = {fitted_offset_s:.3f} s vs refined lag0 = '
            f'{coarse_lag_s:.3f} s (difference {1e3 * (coarse_lag_s - fitted_offset_s):.0f} ms)'
        )
        if output_dir is not None:
            fit.tracks.to_csv(output_dir / f'{dyad_id}_{device_name}_offset_tracks.csv', index=False)

        recordings[device_name] = recording
        tables[device_name] = table
        fits[device_name] = fit
        proxies[device_name] = eeg_proxy
        rankings[device_name] = ranking

    if not fits and not failures:
        raise ValueError('No BI-Vital device found in bivital_paths.')

    summary = pd.DataFrame([summarize_fit(name, fit) for name, fit in fits.items()])
    if fits:
        summary['search_strategy'] = [strategies[name] for name in fits]
        log(summary.round(4).to_string())

    # ---- ECG and accelerometer of all devices on the EEG time grid ----
    n_eeg_samples = eeg.shape[1]
    export_start_s = 0.0
    if params.export_include_pre_eeg and fits:
        export_start_s = min(
            device_time_to_eeg_time(tables[name]['t'].iloc[0], fits[name]) for name in fits
        )
        export_start_s = np.floor(export_start_s * params.export_fs_hz) / params.export_fs_hz
    export_grid_s = np.arange(
        int(round(export_start_s * params.export_fs_hz)),
        int(n_eeg_samples * params.export_fs_hz / params.eeg_raw_fs_hz),
    ) / params.export_fs_hz
    suffix = {name: s for name, s in params.export_column_suffix.items() if name in fits}
    synced_table = (
        build_synced_table(export_grid_s, tables, fits, suffix, params.export_max_gap_s)
        if fits else pd.DataFrame({'eeg_t': export_grid_s})
    )

    if output_dir is not None:
        summary.to_csv(output_dir / f'{dyad_id}_sync_summary.csv', index=False)
        synced_path = output_dir / f'{dyad_id}_synced_signals.csv'
        synced_table.to_csv(synced_path, index=False, float_format=params.export_float_format)
        log(f'saved {synced_path.name}: {len(synced_table)} rows')
        if failures:
            pd.DataFrame(
                [{'device': name, 'reason': reason} for name, reason in failures.items()]
            ).to_csv(output_dir / f'{dyad_id}_sync_failures.csv', index=False)

    return SyncResult(
        dyad_id=dyad_id, eeg_start=eeg_start, tracking_params=tracking_params,
        recordings=recordings, bivital_tables=tables, fits=fits, proxies=proxies,
        rankings=rankings, summary=summary, synced_table=synced_table, failures=failures,
    )
