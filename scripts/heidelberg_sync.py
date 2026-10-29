"""QC driver for the BI-Vital <-> Heidelberg EEG synchronisation.

The synchronisation itself lives in `src.bivital_sync.synchronize_bivital` and runs as a stage
of `src.heidelberg_loader.create_multimodal_data_heidelberg`. This script runs only that stage on one
dyad and produces the diagnostic output:

1. fit the clock model ``t_dev = a + (1 + b) * t_eeg`` of both BI-Vital devices (offset and drift)
   and save ``<DYAD>_synced_signals.csv`` (ECG and accelerometer on the EEG time grid) and the
   fit tables;
2. map both ECGs onto the EEG time axis (including negative times) and save them;
3. locate the period in which both devices were worn by the same person and plot the overlaid,
   aligned ECGs;
4. alignment quality-control figures per device.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))                     # `src` is a package

import matplotlib

MATPLOTLIB_BACKEND = 'MacOSX'           # must be selected before pyplot is imported
matplotlib.use(MATPLOTLIB_BACKEND)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.bivital_sync import SyncParams, synchronize_bivital
from src.clocksync import (
    alignment_xcorr,
    device_time_to_eeg_time,
    eeg_sample_to_device_time,
    map_ecg_to_eeg_grid,
    resample_ecg_to_eeg_time,
    window_correlation_curves,
)
from src.ecg_compare import SharedRunParams, align_to_common_grid, find_shared_wearer_run
from src.load import read_eeg, read_eeg_start, read_markers, read_vhdr_channels
from src.plotting import (
    OverlayParams,
    QcParams,
    plot_aligned_traces,
    plot_ecg_overlay,
    plot_offset_tracks,
    plot_xcorr_qc,
)

# =============================================================================
# Configuration
# =============================================================================
# ---- Paths and recording ----
DATA_DIR = Path('/Users/admin/Documents/Hoza/PROJEKTY/SYNCC_IN_LOCAL_HOME/tmp')  # H032.vhdr/.eeg/.vmrk, H032-adult.csv, ...
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
DYAD_ID = 'H032'

# ---- Sampling and filtering ----
EEG_RAW_FS_HZ = 1000                    # sampling rate of the BrainVision recording
ANALYSIS_FS_HZ = 200                    # common rate of EEG cardiac proxy and ECG
CARDIAC_BAND_HZ = (5, 25)               # band-pass for cardiac artefact / ECG
FILTER_ORDER = 3                        # Butterworth order (applied forward-backward)

# ---- Offset tracking ----
TRACK_WINDOW_S = 120                    # length of each cross-correlation window
TRACK_STEP_S = 60                       # hop between windows
SEARCH_MARGIN_S = 4                     # +/- search range around the coarse lag
ROBUST_FIT_THRESHOLD_S = 0.01           # points farther than this from the Theil-Sen line are rejected
WIDE_SEARCH_MARGIN_S = 30               # wide +/- search range for channel ranking and coarse-lag refinement
COARSE_AGREEMENT_TOL_S = 0.1            # windows within this of the median offset count as consistent
COARSE_MIN_AGREEMENT = 0.5              # minimum fraction of consistent windows to accept the refinement
N_MARKERS_TO_PRINT = 4

# ---- Devices and cardiac proxy ----
# EEG channels of each device's wearer end with this suffix (e.g. 'TP10_P' = parent, 'TP10_C' = child).
WEARER_CHANNEL_SUFFIX = {'adult': '_P', 'child': '_C'}
N_PROXY_CHANNELS = 3                    # best channels averaged (sign-aligned) into the cardiac proxy
CHANNEL_AGREEMENT_TOL_S = 0.05          # offsets of windows / of channels must agree within this
CHANNEL_MIN_AGREEMENT = 0.5             # minimum fraction of a channel's windows agreeing with its own median
REFERENCE_DEVICE = 'adult'              # first trace in the overlay; defines the lag sign
COMPARISON_DEVICE = 'child'

# ---- Overlay of the two BI-Vital ECGs ----
ROLLING_WINDOW_S = 30                   # window of the rolling correlation
ROLLING_STEP_S = 15
MIN_CORRELATION = 0.5                   # threshold on peak |r| defining the "same person" run
ZOOM_WINDOW_S = 6                       # length of each zoomed trace
ZOOM_FRACTIONS = (0.1, 0.5, 0.9)        # zoom positions inside the run (0 = start, 1 = end)
MAX_RESIDUAL_LAG_S = 0.2                # lag range searched in every window (lag-tolerant correlation)

# ---- Alignment quality-control figures (saved in outputs/QC_figs_<DYAD_ID>/) ----
QC_WINDOW_LAG_RANGE_S = 0.5             # one-sided lag range of the per-window correlation curves
QC_GLOBAL_LAG_RANGE_S = 0.5             # one-sided lag range of the whole-period cross-correlation
QC_ZOOM_LAG_MS = 50                     # one-sided lag range of the zoomed whole-period panel
QC_UNCORRECTED_LAG_RANGE_S = 15         # one-sided lag range for the uncorrected (start-time) alignment
QC_UNCORRECTED_ZOOM_MS = 150            # one-sided range of the peak-centred fitted-vs-uncorrected comparison
QC_TRACE_WINDOW_S = 8                   # length of each aligned time-series zoom
QC_TRACE_FRACTIONS = (0.1, 0.5, 0.9)    # zoom positions inside the EEG period (0 = start, 1 = end)

# ---- Export of the synchronised signals (outputs/<DYAD_ID>_synced_signals.csv) ----
EXPORT_FS_HZ = ANALYSIS_FS_HZ# (200Hz);  EEG_RAW_FS_HZ            # sampling rate of the exported eeg_t grid
EXPORT_INCLUDE_PRE_EEG = True          # True: grid also covers BI-Vital data before EEG t=0 (negative eeg_t)
EXPORT_COLUMN_SUFFIX = {'child': 'ch', 'adult': 'cg'}
EXPORT_MAX_GAP_S = {'ecg': 0.1, 'acc': 0.5}   # longer gaps stay NaN instead of being bridged
EXPORT_FLOAT_FORMAT = '%.6g'

# ---- Figures ----
FIGURE_DPI = 120

# =============================================================================
# Pipeline
# =============================================================================
OUTPUT_DIR.mkdir(exist_ok=True)
path_stem = DATA_DIR / DYAD_ID

sync_params = SyncParams(
    eeg_raw_fs_hz=EEG_RAW_FS_HZ,
    analysis_fs_hz=ANALYSIS_FS_HZ,
    cardiac_band_hz=CARDIAC_BAND_HZ,
    filter_order=FILTER_ORDER,
    track_window_s=TRACK_WINDOW_S,
    track_step_s=TRACK_STEP_S,
    search_margin_s=SEARCH_MARGIN_S,
    robust_fit_threshold_s=ROBUST_FIT_THRESHOLD_S,
    wide_search_margin_s=WIDE_SEARCH_MARGIN_S,
    coarse_agreement_tol_s=COARSE_AGREEMENT_TOL_S,
    coarse_min_agreement=COARSE_MIN_AGREEMENT,
    wearer_channel_suffix=WEARER_CHANNEL_SUFFIX,
    n_proxy_channels=N_PROXY_CHANNELS,
    channel_agreement_tol_s=CHANNEL_AGREEMENT_TOL_S,
    channel_min_agreement=CHANNEL_MIN_AGREEMENT,
    export_fs_hz=EXPORT_FS_HZ,
    export_include_pre_eeg=EXPORT_INCLUDE_PRE_EEG,
    export_column_suffix=EXPORT_COLUMN_SUFFIX,
    export_max_gap_s=EXPORT_MAX_GAP_S,
    export_float_format=EXPORT_FLOAT_FORMAT,
)
shared_run_params = SharedRunParams(
    fs_hz=ANALYSIS_FS_HZ,
    rolling_window_s=ROLLING_WINDOW_S,
    rolling_step_s=ROLLING_STEP_S,
    min_correlation=MIN_CORRELATION,
    max_lag_s=MAX_RESIDUAL_LAG_S,
)
qc_params = QcParams(
    zoom_lag_ms=QC_ZOOM_LAG_MS,
    uncorrected_zoom_ms=QC_UNCORRECTED_ZOOM_MS,
    zoom_window_s=QC_TRACE_WINDOW_S,
    zoom_fractions=QC_TRACE_FRACTIONS,
    dpi=FIGURE_DPI,
)
overlay_params = OverlayParams(
    reference_name=REFERENCE_DEVICE,
    comparison_name=COMPARISON_DEVICE,
    rolling_window_s=ROLLING_WINDOW_S,
    min_correlation=MIN_CORRELATION,
    max_lag_s=MAX_RESIDUAL_LAG_S,
    zoom_window_s=ZOOM_WINDOW_S,
    zoom_fractions=ZOOM_FRACTIONS,
    dpi=FIGURE_DPI,
)

# ---- 1. Load EEG ----
channel_names = read_vhdr_channels(path_stem)
eeg = read_eeg(path_stem, n_channels=len(channel_names))
markers = read_markers(path_stem)
eeg_start = read_eeg_start(path_stem)
print('EEG start:', eeg_start)

# ---- 2. Synchronise the BI-Vital devices (also writes <DYAD>_synced_signals.csv and the fit tables) ----
sync = synchronize_bivital(
    eeg, channel_names, eeg_start,
    {device: DATA_DIR / f'{DYAD_ID}-{device}.csv' for device in WEARER_CHANNEL_SUFFIX},
    params=sync_params, dyad_id=DYAD_ID, output_dir=OUTPUT_DIR,
)
recordings, fits, proxies = sync.recordings, sync.fits, sync.proxies
for device_name, fit in fits.items():
    marker_times_s = {
        description: round(eeg_sample_to_device_time(sample, fit, EEG_RAW_FS_HZ), 3)
        for description, sample in markers[:N_MARKERS_TO_PRINT]
    }
    print(device_name, 'marker times in device time (s):', marker_times_s)
plot_offset_tracks(fits, OUTPUT_DIR / f'{DYAD_ID}_sync_drift.png', FIGURE_DPI)

# ---- 3. Map both ECGs onto the EEG time axis ----
eeg_time_by_device = {
    name: device_time_to_eeg_time(recordings[name].time_s, fits[name]) for name in recordings
}
ecg_by_device = {name: recordings[name].ecg for name in recordings}
grid_s, aligned_ecg = align_to_common_grid(eeg_time_by_device, ecg_by_device, ANALYSIS_FS_HZ)
for device_name, ecg in aligned_ecg.items():
    pd.DataFrame({'eeg_t': grid_s, 'ecg_aligned': ecg}).to_csv(
        OUTPUT_DIR / f'{DYAD_ID}_{device_name}_ecg_aligned.csv', index=False
    )

# ---- 4. Compare the two ECGs where they share a wearer ----
shared_run = find_shared_wearer_run(
    grid_s, aligned_ecg[REFERENCE_DEVICE], aligned_ecg[COMPARISON_DEVICE], shared_run_params
)
print(
    f'run with |r|>{MIN_CORRELATION}: {shared_run.start_s:.1f}..{shared_run.stop_s:.1f} s EEG time '
    f'({shared_run.stop_s - shared_run.start_s:.0f} s); peak r={shared_run.correlation:.3f}, '
    f'r(0)={shared_run.correlation_zero_lag:.3f}, '
    f'residual lag {REFERENCE_DEVICE}-{COMPARISON_DEVICE} = {shared_run.lag_ms:.1f} ms'
)
plot_ecg_overlay(
    grid_s, aligned_ecg[REFERENCE_DEVICE], shared_run, overlay_params,
    OUTPUT_DIR / f'{DYAD_ID}_ecg_adult_vs_child_aligned.png',
)

# ---- 5. Quality control of the EEG-ECG alignment (figures per device) ----
tracking_params = sync.tracking_params
qc_dir = OUTPUT_DIR / f'QC_figs_{DYAD_ID}'
qc_dir.mkdir(exist_ok=True)
for device_name, recording in recordings.items():
    fit = fits[device_name]
    eeg_proxy = proxies[device_name]
    window_correlations = window_correlation_curves(
        eeg_proxy, recording.ecg, fit, recording.first_sample_time_s,
        tracking_params, QC_WINDOW_LAG_RANGE_S,
    )
    ecg_on_eeg = resample_ecg_to_eeg_time(recording, fit, len(eeg_proxy), ANALYSIS_FS_HZ)
    whole_period_xcorr = alignment_xcorr(eeg_proxy, ecg_on_eeg, ANALYSIS_FS_HZ, QC_GLOBAL_LAG_RANGE_S)
    # Uncorrected alignment: EEG t=0 placed at the wall-clock start time difference, no clock drift.
    start_time_intercept_s = (eeg_start - recording.start_time).total_seconds()
    ecg_uncorrected = map_ecg_to_eeg_grid(recording, start_time_intercept_s, 0.0, len(eeg_proxy), ANALYSIS_FS_HZ)
    uncorrected_xcorr = alignment_xcorr(eeg_proxy, ecg_uncorrected, ANALYSIS_FS_HZ, QC_UNCORRECTED_LAG_RANGE_S)
    print(
        f'{device_name}: uncorrected alignment (start-time lag, no drift): peak at '
        f'{uncorrected_xcorr.peak_lag_ms:+.0f} ms, r = {uncorrected_xcorr.peak_correlation:.3f} '
        f'(fitted: r = {whole_period_xcorr.peak_correlation:.3f})'
    )
    print(
        f'{device_name}: aligned ECG vs EEG proxy over the EEG period: peak at '
        f'{whole_period_xcorr.peak_lag_ms:+.1f} ms, r = {whole_period_xcorr.peak_correlation:.3f}; '
        f'per-window residual lags {window_correlations.selected_lag_ms[window_correlations.used_mask].min():+.1f}'
        f'..{window_correlations.selected_lag_ms[window_correlations.used_mask].max():+.1f} ms'
    )
    plot_xcorr_qc(
        device_name, window_correlations, whole_period_xcorr, uncorrected_xcorr, qc_params,
        qc_dir / f'QC_{DYAD_ID}_{device_name}_xcorr.png',
    )
    eeg_time_s = np.arange(len(eeg_proxy)) / ANALYSIS_FS_HZ
    plot_aligned_traces(
        device_name, eeg_time_s, eeg_proxy, ecg_on_eeg, whole_period_xcorr, qc_params,
        qc_dir / f'QC_{DYAD_ID}_{device_name}_aligned_traces.png',
    )

plt.show()
