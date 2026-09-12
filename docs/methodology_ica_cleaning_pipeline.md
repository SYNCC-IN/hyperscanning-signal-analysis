# Methodology: EEG ICA Cleaning Pipeline

Technical description of the Independent Component Analysis (ICA) pipeline that
removes ocular, muscular, cardiac, and channel-noise artifacts from the exported
`passive_movies` EEG chunks before any spectral or connectivity analysis.

Code: [`src/ica_preprocessing.py`](../src/ica_preprocessing.py) (`ICAPreprocessor`
class). Driver script: [`scripts/EEG_ICA_clean.py`](../scripts/EEG_ICA_clean.py).
Depends on [`src/mne_bridge.py`](../src/mne_bridge.py)'s `load_eeg_ncdf_as_mne_raw`
and the `mne-icalabel` package (`ICLabel` classifier).

---

## 1. Purpose and scope

Stage 3 of the overall data flow (see [CLAUDE.md](../CLAUDE.md)). Input: one
`passive_movies` NetCDF per dyad member, produced by the export pipeline
(see [methodology_export_pipeline.md](methodology_export_pipeline.md)), already
CAR-referenced with bad channels interpolated. Output: the same signal, ICA-cleaned,
written to
`UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/<dyad_id>/<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`,
which every downstream analysis pipeline treats as the canonical clean-EEG input.

The pipeline is explicitly **three separate stages with a mandatory manual QC
checkpoint between stages 2 and 3** — components are never auto-excluded and applied
in the same pass. This is a deliberate design choice: ICLabel's automatic
classification is used to *propose* an exclusion set, but a human reviews the
per-component figure and the editable CSV before any component is actually removed.

## 2. `ICAPreprocessor` — file discovery

`ICAPreprocessor(export_folder, target_events=['passive_movies'])`;
`find_eeg_files(smoke_test=True, smoke_dyads_n=2, dyad_ids=None)` scans
`export_folder` recursively for `*.nc` files whose name contains `_EEG_` and ends in
one of `target_events`, groups them by dyad id (parsed from the filename stem), and
selects either: an explicit `dyad_ids` list, the first `smoke_dyads_n` dyads
(`smoke_test=True`, the safe default for iterating on the method), or every dyad found
(`smoke_test=False`). All three subsequent stages iterate over `self.eeg_files`
populated here, so the same selection is reused across fit → classify → apply for one
run.

## 3. Stage 1 — Fit ICA (`fit_and_save_ica`)

For each selected file:

1. Load as `mne.io.RawArray` via `load_eeg_ncdf_as_mne_raw` (montage
   `standard_1020`, µV → V scaling).
2. Copy and band-pass **1–60 Hz** (`raw_hp`) — a fitting-only filter; the persisted
   cleaned signal in Stage 3 is derived from the *original*, more broadly filtered
   raw, not from `raw_hp`.
3. **Rank estimation**: `mne.compute_rank(raw_hp, rank='info')['eeg']`. This is a
   data-based SVD rank estimate that correctly accounts for the effective
   degrees-of-freedom lost to channel interpolation and CAR referencing at export time
   (both operations reduce the data's true rank below the raw channel count).
4. **ICA decomposition**: `mne.preprocessing.ICA` with:
   - `n_components = min(n_components, rank)` — the requested component count
     (default 15) is capped by the estimated rank, so ICA is never asked to extract
     more independent sources than the data can actually support.
   - `method='infomax'`, `fit_params=dict(extended=True)` — **Extended Infomax**,
     which (unlike standard Infomax) can separate both super- and sub-Gaussian
     sources — appropriate when both spiky artifacts (eye blinks, muscle) and
     smoother physiological sources (alpha rhythms) are expected in the same
     decomposition.
   - `random_state=42` (reproducible decomposition), `max_iter=2000`.
   - `ica.fit(raw_hp, picks='eeg', reject=dict(eeg=500e-6), reject_by_annotation=True)`
     — segments with any EEG channel exceeding ±500 µV are excluded from the fit
     (not from the eventual cleaned output), so a handful of extreme artifacts do
     not dominate the unmixing estimate.
5. The fitted `ICA` object is saved (`<label>-ica.fif`) under
   `<ica_folder>/ICA_COMPs/<dyad_id>/` for stage 2/3 to reload.

## 4. Stage 2 — Classify components (`classify_and_save_labels`)

For each file with a saved ICA model:

1. Reload the raw signal and the saved `ICA`; band-pass **1 Hz high-pass only**
   (`l_freq=1.0, h_freq=None`) — ICLabel's published classifier is trained on
   high-pass-only-filtered data, distinct from the 1–60 Hz fitting filter in Stage 1.
2. **ICLabel classification**: calls `mne_icalabel.iclabel.iclabel_label_components`
   directly (not the higher-level `label_components` wrapper) to obtain the full
   `(n_components, 7)` soft-max probability matrix over ICLabel's seven classes
   (`brain, muscle artifact, eye blink, heart beat, line noise, channel noise,
   other`), rather than only the wrapper's per-component arg-max. The predicted class
   per component is the arg-max of that row.
3. **Per-component amplitude check**: for each component `j`, reconstruct the raw
   signal using *only* that component
   (`ica.apply(raw.copy(), include=[j], n_pca_components=ica.n_components_)` —
   `n_pca_components` is pinned to the actual component count so the residual
   PCA dimensions beyond `n_components_` are not silently added back in, which would
   otherwise inflate every component's apparent amplitude identically) and record the
   maximum absolute back-projected amplitude (µV), restricted to time samples that
   fall inside an actual task event (movie/talk), from `task_events_structure`.
4. **Auto-exclusion decision** (`auto_exclude`), the logical OR of three independent
   criteria, each with a distinct rationale:
   - **c1 — confident artifact**: the predicted class is in
     `exclude_labels = {muscle artifact, eye blink, heart beat, line noise, channel
     noise}` (`brain`/`other` are never auto-excluded by this criterion) AND the
     predicted-class probability ≥ `iclabel_threshold` (default **0.70**).
   - **c2 — not confidently neural**: the combined "keep it in" probability mass is
     below `neural_threshold` (default **0.50**). This combined mass is computed
     adaptively: `p_neural = p_brain + p_other` if `p_brain ≥ p_artifacts` (i.e.
     "other" is treated as more likely genuinely neural than artifactual for this
     component), otherwise `p_neural = p_brain` alone (and `p_other` is folded into
     the artifact mass instead) — this avoids letting an ambiguous "other" label
     rescue a component ICLabel otherwise leans toward classifying as an artifact.
   - **c3 — amplitude veto**: the per-component projected amplitude from step 3
     exceeds `amplitude_threshold` (default **100 µV**), regardless of ICLabel's
     class probabilities. This is a hard safety net against components ICLabel
     misclassifies as benign but which are physiologically implausible in amplitude.
5. Results are written to `<label>_ica_labels.csv` — one row per component, columns
   `component, iclabel, iclabel_prob, prob_brain, prob_muscle, prob_eye, prob_heart,
   prob_line_noise, prob_channel_noise, prob_other, max_projected_amp, auto_exclude,
   exclude, notes`. **`exclude` starts equal to `auto_exclude` but is the column a
   human is expected to hand-edit** before Stage 3 runs; `notes` is a free-text field
   for the reviewer.
6. A QC figure (`<label>_ica_classification.png`) is saved: one row per component,
   scalp topography (left) next to its time course (right, amplitude-normalized and
   then rescaled to its true projected µV amplitude within the task window), coloured
   red (auto-excluded), green (`brain`/`other`, kept), or orange (uncertain but not
   auto-excluded), with all seven class probabilities printed in the panel title.

## 5. Manual QC checkpoint (between stages 2 and 3)

This is not a script step but a required human action: open
`<label>_ica_classification.png` and `<label>_ica_labels.csv` side by side, review
every `auto_exclude=True` component (and any `orange`/uncertain one) against its
topography and time course, and edit the CSV's `exclude` column by hand where the
automatic decision looks wrong (either direction — un-flagging a false positive or
flagging a component the heuristic missed). Stage 3 reads whatever is in `exclude` at
the time it runs, so this file is the actual record of the cleaning decision, not the
`auto_exclude` proposal.

## 6. Stage 3 — Apply and save (`apply_ica_and_save`)

For each file with a reviewed labels CSV:

1. Reload the original raw signal (not `raw_hp`) and the fitted `ICA`.
2. Set `ica.exclude` from the CSV's (possibly hand-edited) `exclude` column.
3. `ica.apply(raw_cleaned)` — MNE reconstructs the signal from every *non-excluded*
   component's contribution, i.e. subtracts the excluded components' projections
   back into channel space.
4. The cleaned signal is written to
   `EEG_ICA_CLEANED/<dyad_id>/<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`, an
   `xarray.DataArray` carrying forward the original file's attrs plus provenance:
   `ica_method='infomax_extended_picard'`, `ica_excluded` (component names),
   `ica_excluded_labels` (their ICLabel classes), and an appended
   `processing_history` note (`... -> ICA_cleaned`).
5. Optionally (`save_plots=True`, default) a full-signal preview figure is saved to
   `EEG_ICA_CLEANED/FIGS/<label>_cleaned_plot.png` (via
   `src.plot_utils.plot_xarray_signals`), titled with the excluded component list, for
   a final visual sanity check of the cleaned trace.

## 7. Output contract

- One cleaned NetCDF per dyad member: same `(time, channel)` shape and channel set as
  the input `passive_movies` file (no channels are dropped in this stage — component
  *rejection*, not channel rejection), same time axis and event structure.
- This is the sole input every later pipeline reads through
  `src.io_utils.get_participant_files` / `load_eeg_nc`.

## 8. Known limitations and deliberate design choices

- **No automatic-only mode is offered as the intended workflow**: the auto-exclude
  heuristic (c1/c2/c3) exists to make manual review tractable (pre-flagging likely
  artifacts), not to replace it. Running Stage 3 directly on unreviewed
  `auto_exclude` values is possible (the CSV starts with `exclude = auto_exclude`)
  but is not the documented, expected usage.
- **Fixed thresholds** (`iclabel_threshold=0.70`, `neural_threshold=0.50`,
  `amplitude_threshold=100 µV`) are not tuned per dataset or per age group; they are
  script defaults a caller can override per run but are not re-derived automatically
  from the data.
- **`n_components` is a soft cap**, not a fixed request — the effective number of
  components extracted is `min(n_components, rank)`, so two dyads with different
  numbers of interpolated channels can legitimately end up with different component
  counts even when `n_components` is requested identically.
- The two different band-pass filters used across stages (1–60 Hz for fitting vs.
  1 Hz-high-pass-only for ICLabel classification) are both intentional and must not
  be unified — using the classification filter for fitting (or vice versa) would
  deviate from ICLabel's trained assumptions or from good ICA-fitting practice,
  respectively.
- The pipeline assumes `mne-icalabel` and `autoreject` are installed
  (`pip install mne-icalabel`); these are optional/heavier dependencies not always
  required by other parts of the repository (see [CLAUDE.md](../CLAUDE.md)'s
  Environment section).
