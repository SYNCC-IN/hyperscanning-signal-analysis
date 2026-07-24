# ICAPreprocessor — documentation

A class for EEG signal decomposition using ICA, spectral parametrization of
components (FOOOF/specparam), group-level clustering, and signal reconstruction
from selected components.

Designed for hyperscanning analysis (mother–child dyads), where the goal is to
identify components corresponding to specific EEG rhythms (alpha, theta, etc.)
and compare them across participants.

---

## Table of contents

- [Dependencies](#dependencies)
- [Pipeline architecture](#pipeline-architecture)
- [Output file structure](#output-file-structure)
- [Class methods](#class-methods)
  - [Initialization and file discovery](#initialization-and-file-discovery)
  - [Phase 1 — ICA decomposition](#phase-1--ica-decomposition)
  - [Phase 2 — component visualization](#phase-2--component-visualization)
  - [Phase 3 — group-level analysis](#phase-3--group-level-analysis)
  - [Phase 4 — user assignments and reconstruction](#phase-4--user-assignments-and-reconstruction)
  - [Helper methods](#helper-methods)
- [Typical workflow](#typical-workflow)
- [File format: _components.nc](#file-format-_componentsnc)
- [Assignment CSV format](#assignment-csv-format)
- [Design notes](#design-notes)

---

## Dependencies

```
mne
xarray
netCDF4
numpy
pandas
matplotlib
scikit-learn
scipy
fooof          # or: specparam (the newer name of the same library)
```

Local:
```python
from src.export import load_eeg_ncdf_as_mne_raw, plot_loaded_eeg_signals
```

---

## Pipeline architecture

The pipeline consists of four sequential phases. Each phase reads from files
produced by the previous one.

```
Source EEG files (.nc)
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 1 — decompose_and_save()         │
│            compute_component_features() │
│                                         │
│  Writes to disk:                        │
│  *_raw.nc          original signal      │
│  *-ica.fif         ICA model (MNE)      │
│  *_sources.nc      source time series   │
│  *_components.nc   topographies + PSD   │
│                    + FOOOF parameters   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 2 — plot_component()             │
│            plot_component_grid()        │
│                                         │
│  Interactive component inspection       │
│  (topomap + spectrum with FOOOF fit)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 3 — collect_features()           │
│            cluster_components()         │
│            find_cross_group_equivalents │
│            export_assignment_template() │
│                                         │
│  Within-group clustering (ch, cg)       │
│  Cross-group comparison                 │
│  CSV template export for verification   │
└─────────────────────────────────────────┘
        │
        ▼
   [user fills in CSV]
        │
        ▼
┌─────────────────────────────────────────┐
│  Phase 4 — load_user_assignments()      │
│            reconstruct_with_assignments │
│                                         │
│  Signal reconstruction from selected    │
│  components per EEG rhythm              │
│  *_reconstructed_{rhythm_label}.nc      │
└─────────────────────────────────────────┘
```

---

## Output file structure

```
ica_output/
└── W_030/
    ├── W_030_EEG_ch_passive_movies_raw.nc
    ├── W_030_EEG_ch_passive_movies-ica.fif
    ├── W_030_EEG_ch_passive_movies_sources.nc
    ├── W_030_EEG_ch_passive_movies_components.nc
    └── W_030_EEG_ch_passive_movies_ica_topographies.png

ica_reconstructed/
└── W_030/
    ├── W_030_EEG_ch_passive_movies_reconstructed_alpha.nc
    └── W_030_EEG_ch_passive_movies_reconstructed_theta.nc
```

All `.nc` files store either `xr.DataArray` (signals) or `xr.Dataset`
(components) with full provenance (`dyad_id`, `who`, `site`) in their attributes.

---

## Class methods

---

### Initialization and file discovery

#### `__init__`

```python
ICAPreprocessor(export_folder: Path, target_events: list)
```

| Parameter | Type | Description |
|---|---|---|
| `export_folder` | `Path` | Root directory containing source `.nc` files |
| `target_events` | `list[str]` | Event names to process, e.g. `['passive_movies']` |

---

#### `find_eeg_files`

```python
find_eeg_files(smoke_test: bool = True, smoke_dyads_n: int = 2)
```

Recursively searches `export_folder` for `.nc` files containing `_EEG_` in
their name and ending with one of the `target_events`. Groups files by dyad
and populates `self.eeg_files`.

| Parameter | Default | Description |
|---|---|---|
| `smoke_test` | `True` | If `True`, processes only the first `smoke_dyads_n` dyads |
| `smoke_dyads_n` | `2` | Number of dyads in smoke test mode |

Prints a summary: number of dyads, files, and selected dyad list.

---

### Phase 1 — ICA decomposition

#### `decompose_and_save`

```python
decompose_and_save(
    output_folder: Path,
    ica_n_components: int = 15,
    ica_max_iter: int = 2000,
    save_plot: bool = True,
) -> None
```

For each file in `self.eeg_files`:

1. Saves the original signal (`*_raw.nc`).
2. Fits ICA on a 1 Hz high-pass filtered copy of the signal.
3. Saves the ICA model (`*-ica.fif`).
4. Computes source time series from the **original** (unfiltered) signal and saves them (`*_sources.nc`).
5. Optionally saves a PNG with component topographies.

| Parameter | Default | Description |
|---|---|---|
| `output_folder` | — | Output directory; subdirectories created per `dyad_id` |
| `ica_n_components` | `15` | Number of ICA components |
| `ica_max_iter` | `2000` | Maximum number of FastICA iterations |
| `save_plot` | `True` | Whether to save component topography PNG |

> **Important:** source time series are computed from the original (unfiltered)
> `raw`, not from `raw_for_ica`. This ensures the full frequency range is
> preserved in `*_sources.nc` and that results are consistent with the
> saved ICA model.

---

#### `compute_component_features`

```python
compute_component_features(
    decomposition_folder: Path,
    psd_fmin: float = 1.0,
    psd_fmax: float = 45.0,
    psd_n_fft: int | None = None,
    fooof_max_n_peaks: int = 4,
    fooof_peak_threshold: float = 2.0,
    fooof_r_squared_threshold: float = 0.90,
    fooof_aperiodic_mode: str = 'fixed',
) -> None
```

For each file, loads `*_sources.nc` and `*-ica.fif` and computes:

- **Topographies** via `ica.get_components()` — raw (`topomap_raw`) and L2-normalized with sign correction (`topomap`).
- **PSD** using Welch's method (`psd_power`).
- **FOOOF parameters** — aperiodic component (`offset`, `exponent`) and oscillatory peaks (`CF`, `PW`, `BW`).

Results are saved to `*_components.nc` (see [File format](#file-format-_componentsnc)).

| Parameter | Default | Description |
|---|---|---|
| `psd_fmin` / `psd_fmax` | `1.0` / `45.0` | Frequency range for PSD and FOOOF (Hz) |
| `psd_n_fft` | `None` | FFT window size; `None` = auto (~4 s of data) |
| `fooof_max_n_peaks` | `4` | Maximum number of FOOOF peaks per component |
| `fooof_peak_threshold` | `2.0` | Peak detection threshold (SD above residuals) |
| `fooof_r_squared_threshold` | `0.90` | Minimum R² — components below this get `fooof_valid=False` |
| `fooof_aperiodic_mode` | `'fixed'` | Aperiodic component mode: `'fixed'` or `'knee'` |

> **Requires:** `fooof` or `specparam`. Raises `ImportError` if neither is installed.

---

### Phase 2 — component visualization

#### `plot_component`

```python
plot_component(
    components_path: Path,
    comp_id: str | int,
    *,
    topomap_cmap: str = 'RdBu_r',
    show: bool = True,
    save_path: Path | None = None,
) -> plt.Figure
```

Draws a two-panel figure for a single component:

- **Left panel** — scalp topomap with a color bar.
- **Right panel** — power spectrum (semilogy: power units on a log-scale Y axis) with FOOOF fit overlay: gray PSD, dashed line (aperiodic component), solid line (full model), shaded EEG bands (delta/theta/alpha/beta/gamma), peak annotations (CF and PW).

| Parameter | Description |
|---|---|
| `components_path` | Path to `*_components.nc` |
| `comp_id` | Component name (`'ICA003'`) or index (`3`) |
| `show` | Whether to call `plt.show()` |
| `save_path` | Optional path to save as PNG |

Returns `plt.Figure`. The figure is **not closed** inside the method — the caller
manages its lifecycle.

---

#### `plot_component_grid`

```python
plot_component_grid(
    components_path: Path,
    *,
    n_cols: int = 5,
    topomap_cmap: str = 'RdBu_r',
    show: bool = True,
    save_path: Path | None = None,
) -> plt.Figure
```

Draws a grid of topomaps for all components in a single file. Below each
topomap: component name, peak CF values, `exponent`, and percentage of explained
variance. Components with `fooof_valid=True` have a green border; others have gray.

---

### Phase 3 — group-level analysis

#### `collect_features`

```python
collect_features(
    decomposition_folder: Path,
    member_filter: str | None = None,
    valid_only: bool = True,
) -> pd.DataFrame
```

Aggregates features from all `*_components.nc` files into a flat `pd.DataFrame`
with one row per component.

| Parameter | Description |
|---|---|
| `member_filter` | `'ch'` (children), `'cg'` (caregivers), `None` (all) |
| `valid_only` | If `True`, skips components with `fooof_valid=False` |

**DataFrame columns:**

| Column | Description |
|---|---|
| `file_stem`, `dyad_id`, `who`, `site` | Provenance |
| `component` | Component name (`ICA000`…) |
| `fooof_valid`, `fooof_r_squared` | FOOOF fit quality |
| `explained_var_ratio` | Approximate explained variance |
| `fooof_offset`, `fooof_exponent` | Aperiodic parameters |
| `fooof_cf_{k}`, `fooof_pw_{k}`, `fooof_bw_{k}` | Parameters of the k-th peak |
| `topo_0` … `topo_{n-1}` | Normalized topography (L2, sign-corrected) |

---

#### `cluster_components`

```python
cluster_components(
    features_df: pd.DataFrame,
    n_clusters: int = 5,
    features: str = 'both',
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]
```

Clusters components using K-means after feature standardization
(`StandardScaler`). NaN values in peak columns are replaced with zero before
clustering.

| Parameter | Description |
|---|---|
| `n_clusters` | Number of clusters |
| `features` | `'topomap'`, `'fooof'`, or `'both'` |

Returns `(df_out, templates)` where `df_out` is the input DataFrame with an
added `cluster_label` column, and `templates` is a dict containing mean
topographies and FOOOF parameters per cluster.

> Run clustering separately for children and caregivers — EEG rhythm frequency
> ranges may differ between age groups.

---

#### `find_cross_group_equivalents`

```python
find_cross_group_equivalents(
    templates_a: dict,
    templates_b: dict,
    features: str = 'topomap',
) -> pd.DataFrame
```

Computes cosine similarity between cluster templates from two groups.
Applies sign correction — `sim = max(cos(a,b), cos(a,-b))` — because ICA
topographies are sign-ambiguous.

Returns a DataFrame sorted in descending order by `similarity`, with columns:
`cluster_a`, `cluster_b`, `similarity`, `n_a`, `n_b`, `mean_cf_a`, `mean_cf_b`,
`features_used`.

---

#### `export_assignment_template`

```python
export_assignment_template(
    features_df: pd.DataFrame,
    output_csv: Path,
    suggested_label_col: str = 'cluster_label',
) -> None
```

Generates a CSV with key component parameters and empty columns
`rhythm_label`, `confidence`, `notes` for manual completion by the user.
Optionally includes a column with the suggested cluster label.

See [Assignment CSV format](#assignment-csv-format).

---

### Phase 4 — user assignments and reconstruction

#### `load_user_assignments`

```python
load_user_assignments(csv_path: Path) -> dict[str, dict[str, str]]
```

Reads the user-filled CSV and returns a nested dict:

```python
{
    'W_030_EEG_ch_passive_movies': {
        'ICA003': 'alpha',
        'ICA007': 'theta',
    },
    ...
}
```

Rows with an empty `rhythm_label` are skipped. Prints a summary: number of
assigned rows out of total rows.

---

#### `reconstruct_with_assignments`

```python
reconstruct_with_assignments(
    assignments: dict[str, dict[str, str]],
    decomposition_folder: Path,
    output_folder: Path,
    rhythm_labels: list[str] | None = None,
) -> None
```

For each file in `assignments`:

1. Loads the ICA model (`*-ica.fif`) and the original signal (`*_raw.nc`).
2. Groups components by `rhythm_label`.
3. Calls `ica.apply(raw, include=[comp_indices])` — keeps only the selected components, zeroes out the rest.
4. Saves the result as `*_reconstructed_{rhythm_label}.nc`.

| Parameter | Description |
|---|---|
| `rhythm_labels` | Label filter; `None` = all non-empty labels |

> Before `ica.apply`, `ica.exclude = []` is set explicitly so that components
> marked during earlier preprocessing are not silently removed.

---

### Helper methods

| Method | Description |
|---|---|
| `_extract_provenance(attrs, stem)` | Extracts `dyad_id`, `who`, `site` from DataArray attributes or from the filename as fallback |
| `_sanitize_attrs(attrs)` | Converts attribute values to NetCDF-compatible types (`dict/list → JSON`, `None → ''`) |
| `_nanmean_peaks(df_mask, prefix)` | Mean of non-NaN values across columns matching a given prefix (e.g. `fooof_cf_`) |
| `_template_vector(tmpl, features)` | Builds a feature vector from a cluster template for cosine comparison |
| `_signed_cosine_sim(a, b)` | Sign-invariant cosine similarity: `max(cos(a,b), cos(a,-b))` |

---

## Typical workflow

### Notebook 1 — decomposition and feature extraction

```python
from pathlib import Path
from ica_preprocessing import ICAPreprocessor

proc = ICAPreprocessor(
    export_folder=Path('data/UNIWAW_imported/EEG'),
    target_events=['passive_movies'],
)
proc.find_eeg_files(smoke_test=True, smoke_dyads_n=2)

ica_folder = Path('data/ica_output')

# Phase 1a — fit ICA, save models and sources
proc.decompose_and_save(ica_folder, ica_n_components=15)

# Phase 1b — compute PSD and FOOOF parameters
proc.compute_component_features(ica_folder, psd_fmax=45.0)
```

### Notebook 2 — component inspection

```python
comp_nc = ica_folder / 'W_030' / 'W_030_EEG_ch_passive_movies_components.nc'

# Overview of all components
proc.plot_component_grid(comp_nc, n_cols=5,
                         save_path=ica_folder / 'W_030' / 'grid.png')

# Detailed view of a single component
fig = proc.plot_component(comp_nc, comp_id=3)
fig = proc.plot_component(comp_nc, comp_id='ICA007')
```

### Notebook 3 — group-level clustering

```python
# Collect features separately for children and caregivers
df_ch = proc.collect_features(ica_folder, member_filter='ch')
df_cg = proc.collect_features(ica_folder, member_filter='cg')

# Within-group clustering
df_ch_cl, templates_ch = proc.cluster_components(df_ch, n_clusters=6, features='both')
df_cg_cl, templates_cg = proc.cluster_components(df_cg, n_clusters=6, features='both')

# Cross-group comparison (topography-based)
equiv = proc.find_cross_group_equivalents(templates_cg, templates_ch, features='topomap')
print(equiv.head(10))

# Export templates for manual labelling
proc.export_assignment_template(df_ch_cl, Path('assignments_ch.csv'))
proc.export_assignment_template(df_cg_cl, Path('assignments_cg.csv'))
```

### Notebook 4 — reconstruction after user review

```python
# After the user has filled in assignments_ch.csv
assignments = proc.load_user_assignments(Path('assignments_ch.csv'))

proc.reconstruct_with_assignments(
    assignments,
    decomposition_folder=Path('data/ica_output'),
    output_folder=Path('data/ica_reconstructed'),
    rhythm_labels=['alpha', 'theta'],   # None = all non-empty labels
)
```

---

## File format: _components.nc

An `xr.Dataset` — the central hub of the analysis. One file per recording.

### Variables

| Variable | Dimensions | Type | Description |
|---|---|---|---|
| `topomap_raw` | `[component, channel]` | float64 | Raw column of the ICA mixing matrix |
| `topomap` | `[component, channel]` | float64 | L2-normalized, max-abs positive |
| `psd_power` | `[component, frequency]` | float64 | Welch PSD, linear scale (a.u.²/Hz) |
| `fooof_aperiodic` | `[component, ap_param]` | float64 | `[offset, exponent]` |
| `fooof_peaks` | `[component, peak_idx, peak_param]` | float64 | `[CF, PW, BW]`; NaN where no peak |
| `fooof_r_squared` | `[component]` | float64 | FOOOF fit quality |
| `fooof_error` | `[component]` | float64 | FOOOF fit error |
| `fooof_valid` | `[component]` | int8 | `1` if `r² >= fooof_r_squared_threshold` |
| `explained_var_ratio` | `[component]` | float64 | Approximate explained variance ratio |

### Coordinates

| Coordinate | Example values |
|---|---|
| `component` | `['ICA000', ..., 'ICA014']` |
| `channel` | `['Fp1', 'Fp2', ..., 'Oz']` |
| `frequency` | `[1.0, 1.25, ..., 45.0]` Hz |
| `peak_idx` | `[0, 1, 2, 3]` |
| `peak_param` | `['CF', 'PW', 'BW']` |
| `ap_param` | `['offset', 'exponent']` |

### Access patterns

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('W_030_EEG_ch_passive_movies_components.nc')

# Topography of a single component
topo = ds['topomap_raw'].sel(component='ICA003').values

# All peak CFs (NaN where no peak was found)
cf = ds['fooof_peaks'].sel(peak_param='CF').values  # [n_comp, max_peaks]

# Only components with a good FOOOF fit
valid = ds['fooof_valid'].values.astype(bool)
ds_valid = ds.sel(component=ds.component[valid])

# Check whether a component has a peak in the alpha band (8–13 Hz)
cf_vals = ds['fooof_peaks'].sel(component='ICA005', peak_param='CF').values
has_alpha = np.any((cf_vals >= 8.0) & (cf_vals <= 13.0))

ds.close()
```

### Concatenating multiple files

```python
import pandas as pd

datasets = {stem: xr.open_dataset(path) for stem, path in files.items()}
combined = xr.concat(
    list(datasets.values()),
    dim=pd.Index(list(datasets.keys()), name='file'),
)
# combined['topomap'] has shape [n_files, n_components, n_channels]
```

> After concatenation, provenance attributes fall out of `attrs` (xarray does
> not merge attributes across files). For group-level analysis, use
> `collect_features()`, which preserves provenance as explicit DataFrame columns.

---

## Assignment CSV format

Generated by `export_assignment_template()`, filled in manually by the user,
and read back by `load_user_assignments()`.

```csv
file_stem,component,dyad_id,who,fooof_exponent,fooof_cf_0,fooof_pw_0,explained_var_ratio,fooof_r_squared,cluster_label,rhythm_label,confidence,notes
W_030_EEG_ch_passive_movies,ICA003,W_030,ch,1.8234,10.2100,0.4321,0.0823,0.9712,2,alpha,high,posterior parietal
W_030_EEG_ch_passive_movies,ICA007,W_030,ch,2.1045,6.3200,0.3102,0.0612,0.9534,4,theta,,frontal midline
W_030_EEG_cg_passive_movies,ICA005,W_030,cg,1.9821,9.8700,0.5234,0.1023,0.9801,2,alpha,high,
```

**Columns to be filled in by the user:**

| Column | Description |
|---|---|
| `rhythm_label` | EEG rhythm name: `alpha`, `theta`, `beta`, `delta`, `gamma`, or any custom label; **empty = skipped** |
| `confidence` | Optional: `high`, `med`, `low` |
| `notes` | Optional: free-text remarks about the component |

---

## Design notes

### `ica.get_components()` instead of `ica.mixing_matrix_`

`get_components()` inverts the PCA pre-whitening transformation and returns
topographies in the original channel space `[n_channels, n_components]`. This
is the only correct method for obtaining topographies suitable for visualization
and clustering. Using `mixing_matrix_` directly would yield topographies in the
PCA-whitened space, not the original EEG space.

### Source time series from the original `raw`

ICA is fitted on a 1 Hz high-pass filtered signal (to improve algorithmic
convergence), but source time series are computed by applying the unmixing
matrix to the **original** unfiltered signal. This ensures that `*_sources.nc`
retains the full frequency content and remains consistent with the saved model.

### Topography sign correction

ICA is sign-ambiguous: `+alpha` and `-alpha` represent the same rhythm. During
normalization (`topomap`) we enforce the convention that the electrode with the
largest absolute value is always positive. When computing similarity between
components from different recordings, we use `max(cos(a,b), cos(a,-b))`.

### Semilogy in `plot_component`

EEG spectra have a 1/f structure — on a linear Y axis, delta-band power
dominates visually and obscures theta/alpha peaks. A logarithmic Y axis
(`set_yscale('log')`) with a linear X axis (Hz) is the standard presentation
in neurophysiology: it preserves physical power units while making oscillatory
peaks above the aperiodic background clearly visible.

### `fooof_valid` stored as `int8` in NetCDF

NetCDF4 does not have a native boolean type and silently converts `bool` to
`int8` on write. Always apply an explicit cast on read:

```python
fooof_valid = ds['fooof_valid'].values.astype(bool)
```

### `ica.exclude = []` before reconstruction

An ICA object loaded from `.fif` may carry a non-empty `exclude` list from a
previous processing session. Before calling `ica.apply(..., include=[...])`,
we always reset this list so that no components are silently removed beyond
those explicitly excluded by the `include` argument.