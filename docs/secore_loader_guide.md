# Loading SECORE Experiment Data (H10 IBI & RMSSD)

This document describes how to load and process Polar H10 inter-beat interval (IBI)
data from the SECORE experiment using the `src.secore_loader` module.

## Table of contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Expected data layout](#expected-data-layout)
- [Quick start](#quick-start)
- [Output xarray structure](#output-xarray-structure)
  - [Dimensions and coordinates](#dimensions-and-coordinates)
  - [Channels](#channels)
  - [Attributes](#attributes)
  - [Event codes](#event-codes)
- [Processing pipeline](#processing-pipeline)
  - [1. Load raw H10 IBI](#1-load-raw-h10-ibi)
  - [2. Ectopic-beat correction and interpolation](#2-ectopic-beat-correction-and-interpolation)
  - [3. Cross-correlation alignment](#3-cross-correlation-alignment)
  - [4. Event annotation](#4-event-annotation)
  - [5. xarray construction](#5-xarray-construction)
- [API reference](#api-reference)
  - [build_h10_ibi_rmssd_xarray_auto](#build_h10_ibi_rmssd_xarray_auto)
  - [build_h10_ibi_rmssd_xarray](#build_h10_ibi_rmssd_xarray)
  - [load_h10_ibi](#load_h10_ibi)
  - [fix_and_interpolate_ibi](#fix_and_interpolate_ibi)
  - [compute_signal_lag](#compute_signal_lag)
- [Plotting example](#plotting-example)
- [Demo notebook](#demo-notebook)

---

## Overview

The SECORE experiment records cardiac signals from both child (CH) and caregiver (CG)
using Polar H10 chest-strap sensors. Each sensor produces a CSV file with raw IBI
values. The `secore_loader` module:

1. Loads raw H10 IBI CSVs for both participants.
2. Corrects ectopic beats using the Kubios method (via neurokit2).
3. Interpolates IBI to a uniform time grid using cubic splines.
4. Computes sliding-window RMSSD (root mean square of successive differences).
5. Aligns H10 signals to the EEG-derived IBI using cross-correlation.
6. Annotates experimental event windows (puzzle, cleaning, wrong present, surprise).
7. Packages everything into a single `xarray.DataArray`.

## Prerequisites

Required Python packages (all listed in `requirements.txt`):

- `numpy`, `pandas`, `scipy`
- `neurokit2` — ectopic beat correction
- `xarray` — output data structure
- `matplotlib` — optional alignment QC plots

The module also depends on `src.dataloader` for loading EEG-derived IBI used during
the cross-correlation alignment step.

## Expected data layout

The loader expects H10 IBI files and a timing file inside the standard dyad folder:

```
data/
  W_<NNN>/
    eeg/
      W_<NNN>_<DD_MM_YYYY>_<HH_MM>_<DEVICE_ID>_IBI.csv   # one per H10 device
      W_<NNN>_1_25fps.txt                                   # experiment timing file
```

Each IBI CSV has three columns (no header):

| Column | Content |
|--------|---------|
| 0 | Stage code (integer) |
| 1 | Computer timestamp (seconds) |
| 2 | IBI value (milliseconds) |

The timing file (`*_1_25fps.txt`) contains tab-separated rows with labels T1–T4
marking the start/end of experimental phases. Lines 5–8 (0-indexed: 4–7) are parsed.

## Quick start

The simplest way to load SECORE data is a single function call:

```python
from src.secore_loader import build_h10_ibi_rmssd_xarray_auto

h10_xarray = build_h10_ibi_rmssd_xarray_auto(
    dyad_nr='030',
    plot=True,                      # show alignment QC plots
    preferred_dev_ch='A83E1E24',    # child H10 device ID
    preferred_dev_cg='A839C92B',    # caregiver H10 device ID
)

print(h10_xarray)
print(f'shape: {h10_xarray.shape}')
```

The `_auto` variant auto-detects the recording date/time and device IDs from the
filenames in the `eeg/` folder. If `preferred_dev_ch` / `preferred_dev_cg` are
provided and match found devices, they are used; otherwise the function falls back to
the available devices.

## Output xarray structure

### Dimensions and coordinates

| Dimension | Description |
|-----------|-------------|
| `time` | Uniform time axis in seconds (zero-aligned to the start of stage 2) |
| `channel` | Channel labels (5 channels) |

### Channels

| Channel | Unit | Description |
|---------|------|-------------|
| `IBI_CH` | ms | Child inter-beat interval (corrected, interpolated) |
| `IBI_CG` | ms | Caregiver inter-beat interval (corrected, interpolated) |
| `RMSSD_CH` | ms | Child sliding-window RMSSD |
| `RMSSD_CG` | ms | Caregiver sliding-window RMSSD |
| `events` | integer code | Event annotation channel |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sampling_frequency_Hz` | int | Uniform sampling rate (default: 8 Hz) |
| `dyad_id` | str | Dyad identifier (e.g. `W_030`) |
| `window_size_RMSSD_s` | int | RMSSD sliding window size in seconds |
| `device_CH` | str | Child H10 device ID |
| `device_CG` | str | Caregiver H10 device ID |
| `recording_date` | str | Recording date (DD_MM_YYYY) |
| `recording_time` | str | Recording time (HH_MM) |
| `units_IBI` | str | `"ms"` |
| `units_RMSSD` | str | `"ms"` |
| `units_events` | str | `"integer_code"` |
| `event_code_map_json` | str | JSON-encoded mapping of event names → integer codes |
| `event_windows_s_json` | str | JSON-encoded dict with start/end times per event window |
| `description` | str | Human-readable description |

### Event codes

| Code | Event |
|------|-------|
| 0 | baseline (no event) |
| 1 | puzzle |
| 2 | cleaning |
| 3 | wrong present |
| 4 | surprise |

Event window boundaries (in seconds) are derived from the timing file using the
following rules relative to phase markers T2–T4:

| Event | Start | End |
|-------|-------|-----|
| puzzle | T2 + 1 min | T2 + 2.5 min |
| cleaning | T3 − 1.5 min | T3 |
| wrong present | T3 | T3 + 1.5 min |
| surprise | T4 | T4 + 1.5 min |

---

## Processing pipeline

### 1. Load raw H10 IBI

`load_h10_ibi(path)` reads the three-column CSV and returns `(stage, timestamp_s, ibi_ms)`.

### 2. Ectopic-beat correction and interpolation

`fix_and_interpolate_ibi(...)`:

1. Converts cumulative IBI times to sample indices at a virtual sample rate (default: 1024 Hz).
2. Applies the **Kubios** ectopic-beat correction method via `neurokit2.signal_fixpeaks()`.
3. Computes NN intervals from corrected peak positions.
4. Fits a **cubic spline** to NN intervals and resamples to a uniform grid at `fs_out` Hz (default: 8 Hz).
5. Computes **sliding-window RMSSD** over each time point using a configurable window
   (default: 30 s). At least 3 peaks must fall within the window for a valid RMSSD value;
   remaining gaps are linearly interpolated.
6. Interpolates the stage signal to the same uniform grid.

### 3. Cross-correlation alignment

The H10 and EEG clocks are not synchronized. To align them:

1. EEG-derived IBI is loaded via `dataloader.create_multimodal_data()` and decimated.
2. `compute_signal_lag(...)` computes the normalized cross-correlation between each
   H10 IBI signal and its corresponding EEG-derived IBI.
3. The lag that maximizes cross-correlation is used to shift the signals so that
   CH and CG are temporally aligned.

When `plot=True`, the alignment QC panel shows the overlap region with lag values in
samples and seconds in the subplot titles.

### 4. Event annotation

The timing file is parsed to extract phase boundaries (T1–T4). Time is re-zeroed to
the start of stage 2 (the experimental phase). Event windows are mapped to integer
codes and stored both as a signal channel and as JSON metadata in the xarray attrs.

### 5. xarray construction

All five channels are stacked into a 2D `xarray.DataArray` with dimensions
`(time, channel)` and comprehensive metadata attributes.

---

## API reference

### `build_h10_ibi_rmssd_xarray_auto`

```python
build_h10_ibi_rmssd_xarray_auto(
    dyad_nr,                        # e.g. '030'
    data_base_path="../data",
    fs_ibi=8,                       # output sampling rate (Hz)
    window_size_rmssd_s=30,         # RMSSD window (seconds)
    decimate_factor_loader=8,       # EEG loader decimation
    decimate_factor_align=16,       # additional alignment decimation
    selected_time=(0, 220),         # EEG time window for alignment (seconds)
    lowcut=1.0,                     # EEG bandpass low cutoff (Hz)
    highcut=40.0,                   # EEG bandpass high cutoff (Hz)
    eeg_filter_type="iir",          # 'iir' or 'fir'
    plot=False,                     # show QC plots
    preferred_dev_ch=None,          # preferred child device ID
    preferred_dev_cg=None,          # preferred caregiver device ID
) -> xr.DataArray
```

Auto-detects the latest recording date/time and device IDs from filenames. Only
`dyad_nr` is required; all other parameters have sensible defaults.

### `build_h10_ibi_rmssd_xarray`

```python
build_h10_ibi_rmssd_xarray(
    dyad_nr, date, time_of_recording, dev_ch, dev_cg,
    data_base_path="../data",
    fs_ibi=8,
    window_size_rmssd_s=30,
    decimate_factor_loader=8,
    decimate_factor_align=16,
    selected_time=(0, 220),
    lowcut=1.0,
    highcut=40.0,
    eeg_filter_type="iir",
    plot=False,
) -> xr.DataArray
```

Full-control version requiring explicit date, time, and device IDs.

### `load_h10_ibi`

```python
load_h10_ibi(path: str) -> tuple[ndarray, ndarray, ndarray]
```

Returns `(stage, computer_timestamps_s, ibi_ms)` from an H10 IBI CSV file.

### `fix_and_interpolate_ibi`

```python
fix_and_interpolate_ibi(
    ibi_ms, ibi_cum_s, stage,
    fs_out=8, samp_rate=1024, window_size=30,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]
```

Returns `(t_interp, ibi_interp, stage_interp, nn_ms, t_nn, rmssd_interp)`.

### `compute_signal_lag`

```python
compute_signal_lag(signal1, signal2, plot=False, label1="", label2="") -> int
```

Returns the integer-sample lag that maximizes the cross-correlation between
`signal1` and `signal2`.

---

## Plotting example

After building the xarray, you can plot IBI and RMSSD with event windows:

```python
import json
import matplotlib.pyplot as plt

t = h10_xarray.coords['time'].values
ibi_ch = h10_xarray.sel(channel='IBI_CH').values
ibi_cg = h10_xarray.sel(channel='IBI_CG').values
rmssd_ch = h10_xarray.sel(channel='RMSSD_CH').values
rmssd_cg = h10_xarray.sel(channel='RMSSD_CG').values

event_code_map = json.loads(h10_xarray.attrs['event_code_map_json'])
event_windows_s = json.loads(h10_xarray.attrs['event_windows_s_json'])

event_colors = {
    'puzzle': '#fde725',
    'cleaning': '#5ec962',
    'wrong present': '#21918c',
    'surprise': '#3b528b',
}

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 8))

axes[0].plot(t, ibi_ch, label='IBI CH', lw=1.2)
axes[0].plot(t, ibi_cg, label='IBI CG', lw=1.2, alpha=0.9)
axes[0].set_ylabel('IBI [ms]')
axes[0].set_title('IBI and RMSSD with event windows')

axes[1].plot(t, rmssd_ch, label='RMSSD CH', lw=1.2)
axes[1].plot(t, rmssd_cg, label='RMSSD CG', lw=1.2, alpha=0.9)
axes[1].set_ylabel('RMSSD [ms]')
axes[1].set_xlabel('Time [s]')

for event_name, window in event_windows_s.items():
    color = event_colors.get(event_name, '#bbbbbb')
    for ax in axes:
        ax.axvspan(float(window['start_s']), float(window['end_s']),
                   color=color, alpha=0.18)

plt.tight_layout()
plt.show()
```

---

## Demo notebook

See [scripts/secore_import_demo.ipynb](../scripts/secore_import_demo.ipynb) for a
runnable example that:

1. Configures dyad number and device IDs.
2. Calls `build_h10_ibi_rmssd_xarray_auto` to build the xarray.
3. Plots IBI and RMSSD with color-coded event windows.
