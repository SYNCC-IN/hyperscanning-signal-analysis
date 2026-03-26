# Export to NCDF and Import from NCDF

This document describes how to export processed multimodal data to NetCDF (`.nc`) files and how to load it back into xarray.

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
    - `Incredible`
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
    - `event_order` (chronological order by event start time for available target events:
      `Peppa`, `Incredible`, `Brave`)
    - for EEG additionally: `eeg.filtration` and `eeg.references`

Use `get_export_metadata(...)` to decode and access this payload safely.

#### Example: reading `event_order`

```python
from src.export import load_xarray_from_netcdf, get_export_metadata

da = load_xarray_from_netcdf("data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc")
meta = get_export_metadata(da)

print(meta["event_order"])
# e.g. ['Peppa', 'Brave', 'Incredible']
```

## Export a full dyad to NCDF

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

### Notes

- Use `verbose=True` to see progress logs.
- The function exports all events for all available modalities/members in the dyad.

## Export one selection to xarray

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

## Load one NCDF file back to xarray

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

## Metadata serialization format

Exported DataArrays include:

- compact scalar attrs (for quick filtering), e.g. `dyad_id`, `event_name`, `who`, `sampling_freq`, `start_time`, `end_time`
- structured metadata serialized to `metadata_json`

Use helper API to access structured metadata safely:

```python
from src.export import get_export_metadata

metadata = get_export_metadata(data_xr)
print(metadata.keys())
```

### Important

In raw NetCDF attrs, `metadata_json` is a JSON string, so direct indexing like this is incorrect:

```python
# ❌ do not do this
# data_xr.attrs["metadata_json"]["eeg"]
```

Use `get_export_metadata(...)` (or `json.loads(...)`) first, then access nested fields.
If you load data with `load_xarray_from_netcdf(..., decode_json_attrs=True)` (default),
attrs may already be decoded to Python objects.

## NetCDF serialization constraints

NetCDF attributes do not support all Python object types directly.

Current export behavior sanitizes attrs before writing:

- `None` -> empty string
- `dict` and nested structures -> JSON string
- non-serializable objects -> string representation

This avoids runtime errors during `to_netcdf(...)`.

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

## Minimal round-trip example

```python
from src.export import load_xarray_from_netcdf, get_export_metadata

path = "data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc"
da = load_xarray_from_netcdf(path)
meta = get_export_metadata(da)

print(type(da).__name__)          # DataArray
print("child_info" in meta)       # True for newly exported files
```
