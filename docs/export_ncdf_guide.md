# Export to NCDF and Import from NCDF

This document describes how to export processed multimodal data to NetCDF (`.nc`) files and how to load it back into xarray.

## Table of contents

- [Overview](#overview)
- [Output folder structure](#output-folder-structure)
- [Naming conventions used in export](#naming-conventions-used-in-export)
  - [Dyad members](#dyad-members)
  - [Modalities](#modalities)
  - [Site and dyad IDs](#site-and-dyad-ids)
  - [Experimental session names](#experimental-session-names)
- [Structure of xarray data stored in exported NCDF](#structure-of-xarray-data-stored-in-exported-ncdf)
  - [Data variable](#data-variable)
  - [Dimensions and coordinates](#dimensions-and-coordinates)
  - [Attributes on `signals`](#attributes-on-signals)
  - [Structured metadata payload (`metadata_json`)](#structured-metadata-payload-metadata_json)
- [Metadata serialization format](#metadata-serialization-format)
  - [NetCDF serialization constraints](#netcdf-serialization-constraints)
- [Export/load data](#exportload-data)
  - [Export a full dyad to NCDF](#export-a-full-dyad-to-ncdf)
  - [Export one selection to xarray](#export-one-selection-to-xarray)
  - [Load one NCDF file back to xarray](#load-one-ncdf-file-back-to-xarray)
  - [Minimal round-trip example](#minimal-round-trip-example)
- [EEG quality checking](#eeg-quality-checking)
  - [Functions](#functions)
  - [Single-file interactive demo](#single-file-interactive-demo)
  - [Batch processing notebook](#batch-processing-notebook)
- [MATLAB R2019b compatibility (channel names)](#matlab-r2019b-compatibility-channel-names)

---

## Overview

The export/import workflow is implemented in [src/export.py](../src/export.py):

- `write_dyad_to_uniwaw_imported(...)` exports a whole dyad into a folder tree with one `.nc` file per modality/member/event.
- `export_to_xarray(...)` exports one selected modality/member/event to a single `xarray.DataArray`.
- `load_xarray_from_netcdf(...)` loads a saved `.nc` file back into `xarray.DataArray`.
- `get_export_metadata(...)` reads the structured metadata payload from `metadata_json`.

## Output folder structure

A typical export path looks like this:

- `data/UNIWAW_imported/<MODALITY>/<DYAD_ID>/<member_folder>/<file>.nc`

Example:

- `data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc`

Where:

- `member_folder` is `child` for `ch` and `caregiver` for `cg`.
- `<file>` follows `<DYAD_ID>_<MODALITY>_<member_code>_<EVENT>.nc`.

## Naming conventions used in export

This project uses the following conventions in NCDF export paths and filenames.

### Dyad members

- Full member names are: `child` and `caregiver`.
- Member codes used in filenames are:
    - `ch` = `child`
    - `cg` = `caregiver`

### Modalities

- Modality names are uppercase in paths and filenames, e.g.:
    - `EEG`
    - `ET`
    - `FNIRS`
    - `IBI`

### Site and dyad IDs

- Site codes:
    - `K` = Kopenhagen
    - `W` = Warsaw
    - `M` = Milan
    - `H` = Heidelberg
- Dyad numeric code uses three zero-padded digits, e.g. `003`, `030`, `125`.
- Practical dyad ID format used in files: `<SITE>_<NNN>` (example: `W_030`).

### Experimental session names

- Session/event names used in exported filenames:
    - `Secore`
    - `Talk1`
    - `Talk2`
    - `Peppa`
    - `Incredibles`
    - `Brave`

Example filename built from these conventions:

- `W_030_EEG_ch_Peppa.nc`

## Structure of xarray data stored in exported NCDF

Each exported file stores one `xarray.DataArray` named `signals`.

### Data variable

- Variable name: `signals`
- Shape: `[n_time, n_channel]`
- Meaning: signal amplitudes/samples for one selected modality, dyad member, and event.

### Dimensions and coordinates

- Dimensions:
    - `time`
    - `channel`
- Coordinates:
    - `time`: relative time in seconds (event start is shifted to `0.0`)
    - `channel`: channel labels (e.g., `Fp1`, `Fp2`, `Cz`)

### Attributes on `signals`

Common scalar/string attributes written during export:

- `dyad_id`
- `who` (`ch` or `cg`)
- `sampling_freq`
- `event_name`
- `event_start` (relative start in exported window; currently `0.0`)
- `event_duration`
- `time_margin_s`
- `channel_names_csv` (MATLAB-friendly comma-separated channel names)
- `channel_names_json` (MATLAB-friendly JSON array of channel names)
- `metadata_json` (serialized structured metadata)

### Structured metadata payload (`metadata_json`)

- JSON object containing, depending on modality:
    - `notes`
    - `child_info`
    - for EEG additionally: `eeg.filtration` and `eeg.references`

Use `get_export_metadata(...)` to decode and access this payload safely.

## Metadata serialization format

Exported DataArrays include:

- compact scalar attrs (for quick filtering), e.g. `dyad_id`, `event_name`, `who`, `sampling_freq`, `event_start`, `event_duration`
- structured metadata serialized to `metadata_json`

Use helper API to access structured metadata safely:

```python
from src.export import get_export_metadata

metadata = get_export_metadata(data_xr)
print(metadata.keys())
```
### NetCDF serialization constraints

NetCDF attributes do not support all Python object types directly.

Current export behavior sanitizes attrs before writing:

- `None` -> empty string
- `dict` and nested structures -> JSON string
- non-serializable objects -> string representation

This avoids runtime errors during `to_netcdf(...)`.

---
## Export/load data
### Export a full dyad to NCDF

```python
from src.export import write_dyad_to_uniwaw_imported

write_dyad_to_uniwaw_imported(
    dyad_id_list=["W_030"],
    input_data_path="data",
    export_path="data/UNIWAW_imported",
    load_eeg=True,
    load_et=True,
    load_meta=True,
    eeg_filter_type="iir",
    time_margin=10,
    verbose=True,
)
```

#### Notes

- Use `verbose=True` to see progress logs.
- The function exports all events for all available modalities/members in the dyad.

### Export one selection to xarray

```python
from src import dataloader
from src.export import export_to_xarray

mmd = dataloader.create_multimodal_data(
    data_base_path="data",
    dyad_id="W_030",
    load_eeg=True,
    load_et=False,
    load_meta=False,
    decimate_factor=8,
)

data_xr = export_to_xarray(
    multimodal_data=mmd,
    selected_event="Peppa",
    selected_channels=["Fp1", "Fp2", "F3"],
    selected_modality="EEG",
    member="ch",
    time_margin=10,
    verbose=False,
)
```

### Load one NCDF file back to xarray

```python
from pathlib import Path
from src.export import load_xarray_from_netcdf

dyad_id = "W_030"
selected_modality = "EEG"
selected_member = "ch"
selected_event = "Peppa"

member_folder = {"ch": "child", "cg": "caregiver"}[selected_member]

nc_path = Path("data/UNIWAW_imported") / selected_modality / dyad_id / member_folder / (
    f"{dyad_id}_{selected_modality}_{selected_member}_{selected_event}.nc"
)

data_xr = load_xarray_from_netcdf(str(nc_path))
print(data_xr)
```

---

### Minimal round-trip example

```python
from src.export import load_xarray_from_netcdf, get_export_metadata

path = "data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc"
da = load_xarray_from_netcdf(path)
meta = get_export_metadata(da)

print(type(da).__name__)          # DataArray
print("child_info" in meta)       # True for newly exported files
```

---

## EEG quality checking

Three functions in [src/export.py](../src/export.py) implement an AutoReject-based quality pipeline for exported EEG NCDF files.

### Functions

#### `load_eeg_ncdf_as_mne_raw(ncdf_path, montage, scale_to_volts)`

Loads an EEG NCDF file and returns an `mne.io.RawArray` object.

- Reads the `signals` DataArray and transposes it to `[channel, time]`.
- Infers sampling frequency from `sampling_freq` attr; falls back to median time-delta when the attr is missing.
- Applies `scale_to_volts` (default `1e-6`, i.e. µV → V).
- Attaches `montage` (default `"standard_1020"`); unknown channels are silently ignored.

#### `plot_eeg_with_rejected_segments(raw, rejected_windows, ..., time_offset, event_duration, time_margin_s)`

Renders stacked EEG traces with rejection and margin overlays.

- Time axis is shifted by `time_offset` so that **0 s = event start**.
- Light-gray shading marks pre-event and post-event margin regions.
- Dashed vertical lines are drawn at t = 0 and t = `event_duration`.
- Red semi-transparent bands mark rejected windows (passed in as a DataFrame with `start_s`/`end_s` columns).

#### `run_eeg_autoreject_quality_report(ncdf_path, epoch_duration_s, n_interpolate, cv, random_state, n_jobs, montage, scale_to_volts, verbose)`

Full pipeline: NCDF → MNE → AutoReject → tabular summaries + visualization.

| Parameter | Default | Description |
|---|---|---|
| `ncdf_path` | — | Path to EEG NCDF file |
| `epoch_duration_s` | `2.0` | Fixed epoch length in seconds |
| `n_interpolate` | `(1, 2, 4)` | AutoReject grid for max interpolated channels |
| `cv` | `5` | Cross-validation folds for AutoReject |
| `random_state` | `42` | Reproducibility seed |
| `n_jobs` | `-1` | Parallel jobs (`-1` = all CPUs) |
| `montage` | `"standard_1020"` | MNE montage name |
| `scale_to_volts` | `1e-6` | µV → V conversion factor |
| `verbose` | `True` | Print MNE / AutoReject progress |

Returns a `dict` with keys:

| Key | Content |
|---|---|
| `raw` | `mne.io.RawArray` |
| `epochs` | `mne.Epochs` (fixed-length) |
| `autoreject` | Fitted `AutoReject` object |
| `reject_log` | `RejectLog` from `ar.transform(return_log=True)` |
| `epoch_summary` | `pd.DataFrame` — one row per epoch: `epoch_idx`, `start_s`, `end_s`, `interpolated_channels`, `rejected`, `in_margin` |
| `channel_summary` | `pd.DataFrame` — one row per channel: `channel`, `interpolated_epochs`, `bad_labels`, `interpolated_pct`, `bad_labels_pct`, sorted by `bad_labels` descending |
| `global_summary` | `dict` — `ncdf_path`, `n_channels`, `n_epochs`, `epoch_duration_s`, `rejected_epochs`, `rejected_epochs_pct`, `total_interpolations` |
| `figure` | `matplotlib.Figure` — stacked EEG plot with rejection and margin overlays |
| `axis` | `matplotlib.Axes` |

**Note on `in_margin`:** An epoch is flagged `in_margin=True` when it lies entirely before t = 0 or entirely after t = `event_duration`. Rejected windows shown in the plot and saved to the CSV report **exclude** margin epochs.

**Dependency:** requires `mne` and `autoreject` (both listed in `requirements.txt`).

### Single-file interactive demo

[scripts/eeg_quality_report_demo.ipynb](../scripts/eeg_quality_report_demo.ipynb) walks through the full pipeline for one EEG NCDF file:

1. Auto-discovers the first EEG file in the export folder.
2. Runs `run_eeg_autoreject_quality_report`.
3. Displays `global_summary`, `channel_summary`, and `epoch_summary` tables.
4. Prints a human-readable channel interpretation per channel, e.g.:  
   `- Fp1 was problematic in 65% of epochs: 0% fixable + 65% still bad`
5. Saves a TOML-style CSV report and a PNG plot (same folder as the NCDF file).

The TOML-style CSV format:

```
section,key,value
global,ncdf_file,"W_000_EEG_cg_Brave.nc"
global,rejected_epochs,2
rejected_windows,epoch_2,{start_s=4.000, end_s=6.000, interpolated_channels=10}
global,rejected_epochs_pct,5.0
global,epoch_duration_s,2.0
global,total_interpolations,47
top_channels,ch_1,{name="Fp1", problematic_pct=65, fixable_pct=0, still_bad_pct=65}
```

Output artefacts follow the NCDF basename:

- `<stem>_quality_report.csv`
- `<stem>_quality_plot.png`

### Batch processing notebook

[scripts/eeg_quality_report_batch.ipynb](../scripts/eeg_quality_report_batch.ipynb) processes all EEG NCDF files found under the export folder:

- **Cell 3** — crawls the tree with `rglob("*.nc")`, filtering on `"_EEG_"` in the filename (538 files for the UNIWAW dataset).
- **Cell 4** — smoke-test toggle:
  ```python
  smoke_test = True   # set False for full batch
  subset_size = 12
  ```
- **Cell 5** — helper functions: `_toml_scalar`, `_extract_dyad_name`, `_get_top_bad_channels`, `_build_report_rows`, `_save_report_artifacts`.
- **Cell 6** — per-file loop: shows dyad heading, runs AutoReject (`verbose=False`), displays the TOML table and plot, saves CSV + PNG artefacts, collects results.
- **Cell 7** — final summary table saved to `EEG_quality_summary_report.csv` in the export folder root.

Summary columns: `dyad`, `ncdf_file`, `status`, `rejected_epochs`, `top_bad_channels` (channels with `bad_labels_pct > 10%`). An `error` column is appended automatically when any file fails.

---

## MATLAB R2019b compatibility (channel names)

To make channel names easy to read in MATLAB R2019b, export now stores channel labels
as variable attributes on `signals`:

- `channel_names_csv` (comma-separated text)
- `channel_names_json` (JSON array text)

Example in MATLAB:

```matlab
ncFile = 'data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc';

% easiest option
csvNames = ncreadatt(ncFile, 'signals', 'channel_names_csv');
channels = strsplit(csvNames, ',');

% alternative: JSON payload
jsonNames = ncreadatt(ncFile, 'signals', 'channel_names_json');
channelsFromJson = jsondecode(jsonNames);
```


