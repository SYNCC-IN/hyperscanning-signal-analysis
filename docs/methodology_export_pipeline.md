# Methodology: Raw Data Import and NetCDF Export Pipeline

Technical description of how raw SVAROG (EEG/ECG) and Pupil-Labs (eye-tracking)
recordings are read, processed, and exported to the per-task NetCDF (`.nc`) files
that every downstream pipeline in this repository consumes.

Code: [`src/dataloader.py`](../src/dataloader.py), [`src/data_structures.py`](../src/data_structures.py),
[`src/export.py`](../src/export.py), [`src/netcdf_io.py`](../src/netcdf_io.py),
[`src/mne_bridge.py`](../src/mne_bridge.py).
Batch driver: [`scripts/export_dyade_to_ncdf_by_task_batch.py`](../scripts/export_dyade_to_ncdf_by_task_batch.py).
Related reference docs: [export_ncdf_guide.md](export_ncdf_guide.md) (I/O contract and
naming conventions), [data_structure_spec.md](data_structure_spec.md) (`MultimodalData`
field reference), [architecture_diagram.md](architecture_diagram.md) (diagrams).

---

## 1. Purpose and scope

This is Stage 1+2 of the overall data flow (see [CLAUDE.md](../CLAUDE.md)): it turns a
dyad's raw recordings into clean, self-describing per-task NetCDF files that carry
their own provenance (filter settings, referencing, channel interpolation) in their
attributes, so every later stage can be re-run without re-deriving these choices.

The pipeline has two conceptually separate steps that happen to be composed inside one
function for the current production path:

1. **Raw import** (`src.dataloader.create_multimodal_data`) — read the two raw
   modalities into one internal, time-aligned container (`MultimodalData`).
2. **Per-task export** (`src.export.export_passive_and_talk_data`) — call step 1
   internally, then slice and write two continuous NetCDF chunks per modality/member.

---

## 2. Stage 1: raw import (`create_multimodal_data`)

### 2.1 Directory contract and channel layout

Input layout per dyad:

```
<data_base_path>/<dyad_id>/
  EEG/ or eeg/
    <dyad_id>.obci   # binary multiplexed signal, float32, SVAROG format
    <dyad_id>.xml    # channel count, sampling frequency, channel labels
  ET/ or et/
    child/{000,001,002}/       # movies, talk1, talk2
    caregiver/{000,001,002}/
```

EEG channel layout is fixed by the recording montage: 21 channels per person
(`Fp1, Fp2, F7, F3, Fz, F4, F8, M1, T3, C3, Cz, C4, T4, M2, T5, P3, Pz, P4, T6, O1,
O2`, suffixed `_cg` for the caregiver), plus two ECG channels (`EKG1`/`EKG2`, `_cg`)
and a photodiode channel, all multiplexed in one `.obci`/`.xml` pair. Old-style
temporal-lobe labels (`T3/T4/T5/T6`) are the SVAROG/10-20 convention used throughout
raw import and internal storage; they are only renamed to the modern
10-20 equivalents (`T7/T8/P7/P8`) at export time, for MNE montage compatibility (§3.3).

### 2.2 EEG processing chain

Applied per person (child/caregiver independently), in this order:

1. **Optional re-referencing** (`mounts_eeg=True`, off by default in production):
   linked-ears montage — every channel has `0.5·(M1 + M2)` subtracted (or the
   caregiver's `M1_cg`/`M2_cg` average), while `M1`/`M2` themselves keep their
   original (pre-reference) values so the operation can be inverted later. Default
   production export leaves the original recording reference untouched
   (`references = "No EEG montage applied; original reference retained"`); CAR
   re-referencing happens later, at export time (§3.3), not here.
2. **Filter design** (`_design_eeg_filters`): a 50 Hz notch (`Q=30`, IIR notch) plus a
   band-pass built from `lowcut`/`highcut`, in one of two families:
   - `fir`: `firwin` low-pass (201 taps) and high-pass (3049 taps), applied with
     `lfilter` and manually delay-corrected (`np.roll` by half the combined filter
     order, zero-padding the tail) since FIR filtering here is causal, not
     zero-phase.
   - `iir`: 2nd-order Butterworth low-pass/high-pass, applied with `filtfilt`
     (zero-phase). **This is the production choice**
     (`export_dyade_to_ncdf_by_task_batch.py`: `eeg_filter_type='iir'`).
   Every filter's type, cutoff, order, and coefficients are recorded on
   `multimodal_data.eeg_filtration` and later serialized into the exported file's
   `metadata_json.eeg.filtration`.
3. **DC removal + filtering** (`_apply_filters`): per channel, subtract the channel
   mean, then apply notch → low-pass → high-pass in sequence.
4. **Storage**: filtered signals are written into `MultimodalData.data`, one column
   per channel (`EEG_ch_<label>` / `EEG_cg_<label>`).
5. **Time origin reset**: the DataFrame's `time`/`time_idx` columns are shifted so
   the first movie event (`Peppa`/`Incredibles`/`Brave`, whichever starts first)
   begins at `t = 0` — the reference every later stage's event windows are expressed
   relative to.

**Production filter settings** (`export_dyade_to_ncdf_by_task_batch.py`):
`lowcut=1.0 Hz`, `highcut=64 Hz` (deliberately wide — narrower band-pass would remove
spectral content ICLabel's classifier relies on downstream), `eeg_filter_type='iir'`.

### 2.3 ECG → IBI extraction

From the same raw multiplexed array, per person:

1. Bipolar ECG derivation: `EKG1 − EKG2` (and the `_cg` pair).
2. High-pass at 0.5 Hz (5th-order Butterworth, `sosfiltfilt`) + 50 Hz notch
   (`filtfilt`).
3. Interbeat intervals (IBI) and a sliding-window RMSSD feature are derived from the
   filtered ECG internally (`MultimodalData._set_ibi`, `_set_RMSSD_from_ECG`,
   `window_size=30 s` by default) and stored as their own `ECG_*`/`IBI_*`/`RMSSD_*`
   columns.

### 2.4 Event detection (diode-based)

Experimental events (which movie is playing, which conversation block) are not
logged separately — they are recovered from a photodiode channel that flashes at
scene onsets, using a purely signal-based decision rule
(`_scan_for_events`, threshold = 0.75 × max diode amplitude):

- The diode is binarized at 75% of its own maximum, then rising/falling edges are
  extracted.
- A sliding 4-second window counts how many short flash pulses preceded a long
  ON-period. A long ON-period (> 55 s, i.e. a movie) preceded by exactly 1, 2, or 3
  flashes is labelled `Brave`, `Peppa`, or `Incredibles` respectively; the event's
  recorded start is the *first preceding flash*, not the movie onset itself, so the
  reported span covers flash burst + movie.
- A short ON-period (< 2 s) followed by a long silence (> 175 s ≈ 2 min 55 s) is a
  conversation trigger, labelled `Talk_1` or `Talk_2`; its start is one sample after
  the flash ends, and its duration is the length of the following silence.

This is intentionally paradigm-specific (three movies then two conversations) and is
not a generic event-detector.

### 2.5 Eye-tracking loading (optional)

`load_et_data` reads per-task Pupil-Labs CSVs (gaze position + 3D pupil diameter +
annotations + blinks) for child and caregiver, builds one common time vector spanning
movies/talk1/talk2, and processes each stream: median filtering + zero-phase low-pass
for gaze position (`median_filter_size=64`, `low_pass_et_order=351`,
`et_pos_cutoff=128 Hz`), confidence-gated median filtering + low-pass for pupil
diameter (`et_pupil_cutoff=4 Hz`, `pupil_model_confidence=0.9`), and blink-confidence
tagging. ET is not part of the current production export (`load_et=False` in the
batch script) but the loader is exercised by tests and the small local demo dataset.

### 2.6 Decimation and consistency check

- If `decimate_factor > 1`, `_decimate_signals` anti-alias-filters (FIR low-pass at
  `0.8·(fs/2q)`) and downsamples every modality together, then updates `fs`.
  Production export uses `decimate_factor=8` (from the native EEG sampling rate down
  to the export rate used by all downstream pipelines).
- `check_consistency_of_multimodal_data` (run by default,
  `consistency_start_error=0.35 s`) cross-validates: (a) `modalities` matches the
  columns actually present, (b) the `events` column agrees with the `events` dict
  structure, (c) when ET is loaded, EEG- and ET-derived event start times agree within
  the allowed tolerance. Failures are reported, not silently ignored; `consistency_strict=True`
  would turn a failure into a hard error (off by default).

---

## 3. Stage 2: per-task NetCDF export (`export_passive_and_talk_data`)

This is the **preferred, current export path** (see [CLAUDE.md](../CLAUDE.md)). For
each dyad it calls `create_multimodal_data` (§2) with the production settings below,
then exports **two continuous chunks per modality × member**, rather than one file per
individual event:

- `passive_movies` — one chunk spanning `Peppa`, `Incredibles`, `Brave` back to back
  (whichever of these three are present).
- `talk` — one chunk spanning every event whose name contains `"talk"`.

### 3.1 Production configuration

From `export_dyade_to_ncdf_by_task_batch.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `lowcut` / `highcut` | 1.0 / 64 Hz | wide band, so ICLabel (Stage 3) has the spectral content it needs |
| `eeg_filter_type` | `'iir'` | zero-phase Butterworth (§2.2) |
| `decimate_factor` | 8 | shared downsample factor for every modality |
| `time_margin` | 20 s | padding kept before/after the chunk for filter-edge safety downstream |
| `export_mounted` | `'CAR'` (default) | common-average reference at export time, not at raw-import time |
| `mounts_eeg_multimodal` | `False` | raw-import linked-ears montage left off; only CAR (below) is applied |
| `EEG_bad_channels` | per-dyad list, e.g. `['Fz_ch','Cz_ch','O2_ch']` | channels flagged bad by visual QC, interpolated before CAR (§3.3) |

`selected_channels['EEG']` is the fixed 21-channel list from §2.1 (including `M1`/`M2`,
dropped later — §3.3).

### 3.2 Chunk assembly (`export_chunk_to_xarray`)

For each `(modality, member, chunk)` combination:

1. **Time window** (`_compute_time_window`): events are sorted by start time; the
   chunk spans from the first selected event's start to the last one's end, padded by
   `±time_margin` (clipped to the recording's actual extent).
2. **Extraction** (`_extract_and_strip`): `MultimodalData.get_signals(...)` pulls the
   raw `(time, channels, data)` slice; the modality/member column prefix (e.g.
   `EEG_ch_`) is stripped from channel names, and time is reset to 0 at the chunk
   start.
3. **EEG-specific correction** (`_apply_eeg_montage`, EEG only, §3.3).
4. **DataArray assembly** (`_build_dataarray`): wraps `(time, channel)` data into one
   `xarray.DataArray` named `signals` with a rich attrs block (§3.4).

### 3.3 EEG channel repair and referencing (via MNE)

Applied only to the EEG modality, using a temporary `mne.io.RawArray`:

1. **Channel renaming for montage compatibility**: old-style 10-20 labels not
   recognised by MNE's `standard_1020` montage are renamed —
   `T3→T7, T4→T8, T5→P7, T6→P8` — purely for spatial interpolation/CAR; the exported
   file's channel coordinate uses the *new* names from this point on. `M1`/`M2` are
   typed `misc` (not `eeg`), so they never enter the CAR average.
2. **Bad-channel interpolation**: channels named in the per-dyad `EEG_bad_channels`
   config (matched by `_<member>` suffix) are marked bad and spherical-spline
   interpolated (`mne.io.Raw.interpolate_bads`) *before* re-referencing, so a single
   noisy channel cannot bias the common average. The interpolation is recorded in
   `metadata_json.interpolation` (a free-text note, e.g. `"Interpolated: ['Fz']"`,
   parsed downstream by `src.assemble.parse_interpolated_channels`).
3. **Common Average Reference (CAR)**: `raw.set_eeg_reference('average')`, then the
   mastoids (`M1`/`M2`) are dropped from the exported channel set (they were excluded
   from the average and carry no further information once CAR is applied). This is
   the production default (`export_mounted='CAR'`); passing `None` instead exports
   the original (per-raw-import) reference untouched.

### 3.4 Output structure

One file per `(modality, member, chunk)`, at
`<export_path>/<MODALITY>/<dyad_id>/<child|caregiver>/<dyad_id>_<MODALITY>_<ch|cg>_<passive_movies|talk>.nc`,
containing one `xarray.DataArray` named `signals`:

- Dims: `(time, channel)`. `time` is seconds, chunk-relative (0 = first movie/talk
  event's start minus the margin is *not* zero — 0 is the event start itself; the
  margin extends into negative/beyond-duration time).
- Scalar attrs: `dyad_id`, `who` (`ch`/`cg`), `modality`, `units` (`μV` for
  EEG/ECG, `px` for ET, `ms` for IBI/RMSSD), `sampling_freq`, `task_name`,
  `task_start` (`0.0`), `task_duration`, `time_margin_s`, `channel_names_csv`,
  `channel_names_json` (MATLAB-friendly redundant encodings), `task_event_names_csv/json`.
- `task_events_structure`: the individual event boundaries making up the chunk
  (e.g. where `Peppa` ends and `Incredibles` begins inside `passive_movies`) —
  the single source of truth downstream stages use to locate per-movie windows
  inside the continuous chunk (`src.netcdf_io.task_regions`).
- `metadata_json`: a JSON-serialized payload with `notes`, `child_info` and, for
  EEG, `eeg.filtration` (notch/low-pass/high-pass settings from §2.2) and
  `eeg.references`/`interpolation` (from §3.3).
- All attrs are run through `sanitize_netcdf_attrs_inplace` before writing (NetCDF
  attrs cannot hold arbitrary Python objects): `None → ''`, dicts/lists → JSON
  strings, anything else non-primitive → `str(...)`.

Files are written `engine='netcdf4', format='NETCDF4_CLASSIC'`.

### 3.5 Quality gate

Immediately after export, `src.mne_bridge.check_exported_data_quality` runs an
AutoReject-based quality report on each member's `passive_movies` file (fixed 2 s
epochs, `n_interpolate=(1,2,4)`, 5-fold CV) and saves a figure + summary to
`<export_path>/EEG/Quality_reports/`. This is descriptive QC (rejected-epoch
percentage, per-channel bad-label rate) — it does not gate whether a file is written,
only whether a human reviewing the report decides to add the dyad to
`EEG_bad_channels` and re-export. See [export_ncdf_guide.md](export_ncdf_guide.md#eeg-quality-checking)
for the full quality-report contract.

---

## 4. Known limitations and deliberate non-goals

- **Older per-event export path** (`write_dyad_to_uniwaw_imported`, one `.nc` per
  individual event rather than per task) still exists and backs the small local
  `data/UNIWAW_imported/` demo dataset, but is not the production path; a third
  function, `export_to_xarray`, that used to expose the same per-event logic as a
  standalone call, is currently commented out in `src/export.py` (dead code, not
  imported anywhere) — only `export_chunk_to_xarray` (the by-task builder) and
  `write_dyad_to_uniwaw_imported` are live per-event/per-chunk paths today.
- **Event detection is paradigm-specific**: `_scan_for_events`'s flash-counting logic
  assumes exactly this experiment's structure (three movies, two conversations,
  specific flash-count encoding) and does not generalize to other diode-timing
  schemes.
- **`mne.Raw.interpolate_bads`** assumes the channel montage (post T3→T7 etc.
  renaming) is spatially valid for spherical-spline interpolation; channels not in
  `standard_1020` are silently ignored when the montage is applied
  (`on_missing='ignore'`).
- **No automatic bad-channel detection**: `EEG_bad_channels` lists are curated by hand
  per dyad from visual/quality-report inspection, not computed by an automated
  detector.
- Raw IBI is exported without any additional smoothing/detrending beyond what §2.3
  already does; the ectopic-beat correction and further HRV feature engineering used
  by the SECORE branch (`src.secore_loader`) is a *separate* pipeline over a
  different acquisition (Polar H10), out of scope here.
