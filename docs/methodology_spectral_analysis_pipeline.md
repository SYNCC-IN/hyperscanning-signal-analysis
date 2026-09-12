# Methodology: Exploratory Spectral Parameterization Pipeline

Technical description of the seven-step pipeline that derives **individualized**
EEG frequency bands and regions of interest (ROIs) per participant, replacing
fixed adult band definitions (e.g. a flat "alpha = 8–13 Hz") that do not hold for a
3–6-year-old cohort whose peak rhythms sit at lower, more variable frequencies.

Code: [`scripts/pipeline_Exploratory_spectral_analysis/`](../scripts/pipeline_Exploratory_spectral_analysis/)
(`01_compute_psd.py` … `07_movie_stability_check.py`), built on
[`src/psd.py`](../src/psd.py), [`src/specparam_utils.py`](../src/specparam_utils.py),
[`src/peaks.py`](../src/peaks.py), [`src/bands.py`](../src/bands.py),
[`src/roi.py`](../src/roi.py), [`src/viz.py`](../src/viz.py).
Input: cleaned EEG from [methodology_ica_cleaning_pipeline.md](methodology_ica_cleaning_pipeline.md),
read via `src.io_utils`.

Each step reads the previous step's output from
`Exploratory_spectral_analysis/<NN_step_name>/` and writes its own subfolder there;
every script is config-block-at-top, functions-in-`src/` (the pattern
[CLAUDE.md](../CLAUDE.md) asks new pipelines to follow).

---

## 1. Common building blocks

- **PSD**: multitaper spectral density (`mne.time_frequency.psd_array_multitaper`
  via `src.psd.compute_psd_multitaper`), computed once per participant × movie ×
  channel and averaged across the three movies per participant
  (`src.psd.average_psd_across_conditions`) — the pipeline analyses each
  participant's *movie-averaged* spectrum by default, with movie-level stability
  checked separately at the end (Step 7).
- **Peak fitting**: `specparam` (formerly FOOOF) via `src.specparam_utils`, which
  decomposes a PSD into an aperiodic (1/f) component plus a variable number of
  Gaussian peaks; only the peak parameters (`center_freq`, `power`, `bandwidth`)
  drive the downstream band/ROI logic in this pipeline.
- **19-channel 10-20 montage**: `Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
  P7, P3, Pz, P4, P8, O1, O2` throughout.
- **Theory-driven ROI groupings** (`src.roi.define_rois_theory`, reused/extended per
  step): `frontal-midline={Fz}`, `frontal-lateral={F3,F4}`,
  `sensorimotor/central={C3,C4}` (`central-midline={Cz}` kept separate),
  `parietal={P3,Pz,P4}`, `occipital={O1,O2}`, `lateral-temporal={T7,T8}`,
  `temporo-parietal={P7,P8}` (the TPJ proxy used downstream in the interbrain dDTF
  pipeline, see [methodology_multimodal_ddtf_pipeline.md](methodology_multimodal_ddtf_pipeline.md)).

---

## 2. Step 1 — Compute PSD (`01_compute_psd.py`)

For every participant × movie, load the cleaned EEG chunk, trim to the movie's event
window (`src.io_utils.trim_to_event_window`), and compute the multitaper PSD:
`FMIN=1.0 Hz, FMAX=30.0 Hz, BANDWIDTH=2.0 Hz` (smoothing half-bandwidth). PSDs are
then averaged across the three movies per participant
(`average_psd_across_conditions`).

`EXCLUDED_DYADS` (12 dyads: `W_008, W_012, W_025, W_027, W_035, W_037, W_042, W_044,
W_057, W_072, W_074, W_114`) are dropped from this step onward, on quality grounds
established during QC inspection (severe artifact contamination or dead/flat
channels) — both dyad members are excluded together.

Artifacts: 1a (per-participant PSD plots, one line per movie plus the average), 1b
(grand-average PSD by role). Movie-averaged PSDs and metadata are pickled to
`derived_data/` for Step 2.

## 3. Step 2 — specparam fitting (`02_run_specparam.py`)

Fits one `specparam.SpectralModel` per participant × channel on the movie-averaged
PSD, with:

| Parameter | Value |
|---|---|
| `freq_range` | (3, 15) Hz |
| `peak_width_limits` | (1, 3) Hz |
| `max_n_peaks` | 4 |
| `min_peak_height` | 0.05 (log power above the aperiodic fit) |
| `aperiodic_mode` | `'fixed'` (no spectral knee) |

Outputs: `all_peaks.csv` (one row per detected peak: `channel, center_freq, power,
bandwidth`, plus participant metadata) and a per-channel fit-quality table
(`r_squared`, `error`, `n_peaks`, aperiodic `exponent`/`offset`) — the two tables
every later step reads from. Artifacts: 2a (a gallery of individual fits, sampled to
`N_GALLERY_CHILDREN=61`/`N_GALLERY_CAREGIVERS=61` with `RANDOM_SEED=42`), 2b (fit
quality summary), 2c (a log of channels where no peak was detected in range — a
data-quality signal, not necessarily an error).

A standalone variant, `02_individual_specparam.py`, repeats Steps 1+2 end-to-end for
one participant without depending on the pickled Step 1 output — used for
spot-checking or debugging a single dyad member with identical parameters.

## 4. Step 3 — Peak inventory (`03_peak_inventory.py`)

Characterizes *where on the scalp and at what frequency* peaks are actually detected,
before committing to any band definition:

- **Prevalence** (`src.peaks.compute_peak_prevalence`): for each channel × frequency
  bin (`FREQ_BINS = [(3,5),(5,7),(7,9),(9,11),(11,13)]`), the fraction of
  participants (within a role, optionally within a diagnostic group) with at least
  one detected peak there.
- **Cross-channel clustering** (`src.peaks.cluster_peaks_across_channels`,
  `FREQ_TOLERANCE=1.0 Hz`): for one participant, greedily group all detected peaks
  (sorted by frequency) into clusters whose members are within tolerance of the
  cluster's running mean frequency — since peaks are processed in ascending order,
  comparing only against the most-recently-opened cluster is equivalent to comparing
  against every open cluster. A cluster spanning ≥ `MIN_CHANNELS=3` channels is
  flagged topographically "robust".
- **Within-ROI clustering** (`cluster_peaks_within_roi`/`cluster_peaks_all_rois`,
  `WITHIN_ROI_FREQ_TOLERANCE=1.0 Hz`): the same greedy algorithm restricted to one
  ROI's channels, with the running center a **power-weighted** mean (louder peaks
  pull the cluster center toward them) — this is the clustering approach later
  superseded, for the specific slow/fast band split, by the seeded k-means of Step 4
  (see its rationale there).

Artifacts: 3a (frequency histograms per scalp cluster/role/group), 3b (prevalence
topomaps), 3c (per-participant strongest-peak-frequency topomaps), 3d (peak
frequency vs. age scatter), 3e (per-participant cluster topomaps, `N_EXEMPLARS=3`
sampled participants), 3f (cluster summary table); a bimodality diagnostic over
`DIAGNOSTIC_ROIS = {sensorimotor, parietal, lateral-temporal, temporo-parietal}`
checks whether a scalp region shows a genuinely bimodal (two-rhythm) distribution
across the cohort, motivating Step 4's two-band model.

## 5. Step 4 — Slow/fast band assignment (`04_band_assignment.py`)

For each participant × ROI (`ROI_CHANNELS`: `frontal-midline={Fz}`,
`sensorimotor={C3,C4}`, `central-midline={Cz}`, `parietal={P3,Pz,P4}`,
`occipital={O1,O2}`, `lateral-temporal={T7,T8}`, `temporo-parietal={P7,P8}`), pools
that participant's raw per-channel peaks within the ROI and assigns **up to two**
individualized rhythm bands — slow and fast — via seeded, power-weighted k-means
(`src.bands.assign_two_bands_kmeans`, full algorithm documented in that function's
docstring):

1. Peaks are restricted to `[SLOW_CF_RANGE[0], FAST_CF_RANGE[1]] = [3.0, 13.0]` Hz.
2. Two seed centers are chosen — one from `SLOW_CF_RANGE=(3.0, 7.5)`, one from
   `FAST_CF_RANGE=(7.5, 13.0)` — each the candidate peak in its range whose
   `center_freq` is *closest to that range's midpoint* (ties: higher power, then
   lower frequency), not simply the highest-power peak in the range. This is a
   deliberate choice: a loud peak sitting right at the shared 7.5 Hz boundary is
   plausible fast-rhythm content that a "highest power" seed would misassign to
   the slow cluster.
3. Every kept peak is assigned to its nearest seed by absolute frequency distance
   (ties → cluster 0, the slow seed); centers are recomputed as each cluster's
   power-weighted mean; this repeats to convergence or `MAX_ITER=50`.
4. If only one seed range had a candidate peak, or the partition collapses to one
   cluster, or the two final centers are separated by less than `MIN_GAP=1.5 Hz`, the
   participant is assigned a **single** rhythm instead (the higher-summed-power
   cluster wins in the collapsed-gap case), labelled slow or fast by comparing its
   center to the shared 7.5 Hz boundary.
5. Each returned band's window is `[center − half_width, center + half_width]`,
   where `half_width` is chosen to cover every merged peak's own
   `center_freq ± bandwidth` extent, not just a fixed multiple of the center.

This replaces an earlier, simpler greedy-clustering + "keep the two strongest peaks"
approach: greedy clustering (1 Hz tolerance) could split one true oscillation across
two adjacent peaks, which a subsequent power-based top-2 selection would then
discard as noise.

Also computed here: individual alpha frequency (IAF) and dyadic (child–caregiver)
distance metrics (`src.bands.compute_iaf_metrics`), from `PRIMARY_ROI='parietal'`
falling back to `FALLBACK_ROI='sensorimotor'` when parietal has no detected rhythm.

Outputs: `band_assignments.csv` (one row per participant × ROI: `slow_cf`, `slow_bw`,
`fast_cf`, `fast_bw`, `assignment_note` ∈ `{two_rhythms, single_slow, single_fast,
no_peaks}`) and `iaf_metrics.csv` — **both consumed directly by the interbrain dDTF
pipeline** (see [methodology_multimodal_ddtf_pipeline.md](methodology_multimodal_ddtf_pipeline.md)).
Artifacts: 4a (assignment strip plot), 4b (summary table), 4c (IAF distance by
group), 4d (slow/fast gap distribution), 4e (dyadic gap scatter), 4f (merged-band
center-frequency histograms).

## 6. Step 5 — ROI viability validation (`05_roi_definition.py`)

Before trusting an ROI's averaged signal downstream, checks whether enough
participants actually show a peak there, separately for each band:

- `SLOW_FREQ_WINDOW=(3,7) Hz`, `FAST_FREQ_WINDOW=(7,14) Hz` — fixed reference ranges
  (not yet the individualized windows from Step 4) used purely to define the
  prevalence-counting bins.
- For every ROI × band × (role, group) subgroup (`GROUPS=['TD','ASD']`), the mean
  peak-detection prevalence across that ROI's channels must reach
  `MIN_PREVALENCE=0.50` (`src.roi.validate_roi_two_bands`) for the ROI to be marked
  viable for that band.
- Single-channel ROIs (`frontal-midline`, `central-midline`) are flagged
  `skip_rerun=True` — an ROI "average" over one channel is a no-op, so Step 6's
  ROI-averaged specparam rerun is skipped for them (their single-channel fit from
  Step 2 already *is* the ROI-level fit).

Output: the final ROI definitions (channels + per-band viability + `skip_rerun`)
consumed by Steps 6 and 7. Artifact 5a: viability heatmap.

## 7. Step 6 — ROI-averaged rerun + peak survival (`06_roi_specparam_rerun.py`)

For every multi-channel viable ROI (`ROI_LAYOUT` groups them for display:
`{frontal-midline, central-midline}`, `{sensorimotor, parietal, occipital}`,
`{lateral-temporal, temporo-parietal}`):

1. Average the movie-averaged PSD across the ROI's channels
   (`src.roi.average_psd_within_roi`), then refit specparam on that averaged PSD
   with the same settings as Step 2 (`freq_range=(3,14)`, `peak_width_limits=(1,3)`,
   `max_n_peaks=4`, `min_peak_height=0.1`, `aperiodic_mode='fixed'`) — note
   `min_peak_height` is stricter here (0.1 vs. 0.05 in Step 2), reflecting that
   ROI-averaging raises SNR and a lower threshold would otherwise let more spurious
   peaks through.
2. **Peak survival check** (`src.roi.check_peak_survival`): for the slow window
   `(3,7) Hz` and fast window `(7,14) Hz` independently, does a peak detected at the
   individual-channel level within the ROI still show up in the ROI-averaged fit,
   within `FREQ_TOLERANCE=1.0 Hz` of the same center frequency? A peak that only
   exists at one channel and washes out under averaging (destructive combination
   across channels with different phases/frequencies) does not survive.
3. Participants/ROIs with a survival rate below `SURVIVAL_FLAG_THRESHOLD=0.60` are
   flagged in `QUALITY_GATE_NOTES.md` as a caveat on using that ROI's averaged signal.

Artifacts: 6a (ROI-average vs. per-channel fit galleries with the band windows
overlaid, `N_GALLERY_CHILDREN=5`/`N_GALLERY_CAREGIVERS=5`), 6b (survival summary
table + bar chart).

## 8. Step 7 — Per-movie stability check (`07_movie_stability_check.py`)

Steps 1–6 all operate on the *movie-averaged* PSD; Step 7 checks whether that
averaging was justified — i.e. whether the slow and fast rhythms are actually stable
across the three individual movies (`MOVIES = ['Peppa','Incredibles','Brave']`),
checked **separately per band** since a stable fast rhythm does not imply a stable
slow one:

- Re-fits specparam per participant × ROI × movie
  (`STABILITY_ROI_NAMES = [sensorimotor, parietal, frontal-midline,
  central-midline, lateral-temporal, temporo-parietal]`, same `SPECPARAM_SETTINGS`
  as Step 6) and compares each movie's detected center frequency (within
  `SLOW_FREQ_WINDOW`/`FAST_FREQ_WINDOW`) against the movie-averaged value; a shift
  larger than `MAX_ACCEPTABLE_SHIFT=2.0 Hz` is flagged unstable.
- Bland-Altman-style agreement plots per ROI × band (7a), a stability summary
  table/heatmap across ROIs and movies (7b), and a per-participant detection
  consistency heatmap for `PRIMARY_ROI='parietal'` across the three movies (7c).

---

## 9. Downstream consumers

`band_assignments.csv` and `iaf_metrics.csv` are read directly by the interbrain
ffDTF + HRV pipeline (Stage 2, `src.bands.band_lookup`) to derive each participant's
individualized amplitude-envelope passband — see
[methodology_multimodal_ddtf_pipeline.md](methodology_multimodal_ddtf_pipeline.md).

## 10. Known limitations and deliberate design choices

- **Fixed thresholds throughout** (`min_peak_height`, `MIN_PREVALENCE=0.50`,
  `MIN_GAP=1.5 Hz`, `FREQ_TOLERANCE=1.0 Hz`, `SURVIVAL_FLAG_THRESHOLD=0.60`,
  `MAX_ACCEPTABLE_SHIFT=2.0 Hz`) are per-script config constants, chosen and recorded
  in `Exploratory_spectral_analysis/spectral_parameterization_report.md`, not
  re-derived per cohort.
- **`EXCLUDED_DYADS`** is a hand-curated list from visual QC, applied from Step 1
  onward; it is not recomputed automatically if new data quality issues are found
  later.
- **The two-band (slow/fast) model is a simplification**: a participant with a
  genuinely richer spectral structure (e.g. three distinct peaks) is still reduced to
  at most two bands by construction (`assign_two_bands_kmeans` always seeds exactly
  two clusters, never more).
- **Movie-averaging as the default unit of analysis** (Steps 1–6) is validated only
  after the fact (Step 7), not gated beforehand — a participant whose rhythm shifts
  by more than 2 Hz across movies still contributes a single averaged band estimate
  to Steps 4–6; Step 7 only flags this, it does not exclude the case automatically.
- ROI channel groupings are theory-driven scalp neighbourhoods
  (`src.roi.define_rois_theory`), not derived from independent-components or
  data-driven spatial clustering.
