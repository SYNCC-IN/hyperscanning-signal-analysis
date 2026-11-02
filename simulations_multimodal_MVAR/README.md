# Simulations S01–S11 (Table 2 of the methods paper)

Every script follows the numbering of Table 2 of `docs/draft_v02_ddtf_multimodal_hyperscanning_simulations.md`
and is run once per connectivity estimator, dDTF (the proposed method) or GPDC. Results go to
`simulations_multimodal_MVAR/dDTF/Results/<estimator>/S_xx/` (`xx` = simulation number).

```bash
python simulations_multimodal_MVAR/dDTF/S03_null_calibration.py quick          # smoke test (seconds), dDTF
python simulations_multimodal_MVAR/dDTF/S03_null_calibration.py full GPDC      # full protocol, GPDC
python simulations_multimodal_MVAR/dDTF/run_all.py quick                       # all scripts, dDTF and GPDC, then the comparison
python simulations_multimodal_MVAR/dDTF/run_all.py full 3 4 7                  # selected scripts, both estimators
python simulations_multimodal_MVAR/dDTF/run_all.py full GPDC                   # GPDC only (dDTF results already exist)
python simulations_multimodal_MVAR/dDTF/compare_estimators.py                  # rebuild the comparison from existing results
```

Each script has a configuration block at the top (`MODE`, `ESTIMATOR`, `N_JOBS`, `BASE_SEED`, `SETTINGS` for quick/full, model and
band parameters). Parameters that are not yet decided in the paper are marked `[TBD]`; acceptance thresholds are marked
`[proposed]` and must be fixed (ideally registered) before the full runs are interpreted. `SIM_NREP=<n>` overrides the number of
replicates (diagnostics, timing).

**Quick and full runs write to the same folders**: a quick run overwrites the full results of that estimator. Each
`summary.json` contains the `mode` it was produced with (and `Results/comparison/runs.csv` lists it per estimator).
Do not interpret quick-mode numbers. `Results/dDTF/` currently holds the full dDTF runs.

## Comparing dDTF and GPDC

The estimator replaces dDTF everywhere it is the method under test: the 4-node analysis of S01–S10, the
reference method of S11, the baseline of S08 (whose estimator variants then include dDTF), the envelope and phase
paths of S02 and S01 Part A (repository `gen_partial_directed_coherence`). The fixed comparisons (ffDTF and DTF in S05,
the competitors in S11) are computed in both runs and are identical. Both runs use the same seeds, so they analyse
the same simulated studies and every comparison is paired.

`run_all.py` ends with `compare_estimators.py`, which writes `Results/comparison/`:

- `criteria.csv`: every acceptance criterion, met or not, per estimator;
- `runs.csv`: mode and number of replicates of each run;
- `S_xx_<table>.csv`: the main readouts of each summary table, with the dDTF and GPDC columns next to each other;
- `comparison.md`: all of these as Markdown tables, ready to copy into the manuscript.

Labels that contain the estimator's name (`4-node dDTF`, `dDTF (4-node)`, criteria) appear as `[est]` so that the rows
of both runs line up. The readouts shown per table are listed in `TABLES` at the top of `compare_estimators.py`.

| ID | Script | Question | Level | Main outputs |
|---|---|---|---|---|
| S01 | `S01_recovery_orientation.py` | Are planted edges recovered and correctly oriented? Fast estimator = repo estimator? | 1 | `partB_summary.csv`, `fig_partA_*`, `fig_partB_recovery.png` |
| S02 | `S02_envelope_vs_phase.py` | Envelope vs phase estimators under amplitude and phase truths, centre-frequency gap | 2 | `summary.csv`, `fig_edge_values_by_truth.png`, `fig_direction_index.png` |
| S03 | `S03_null_calibration.py` | Is the surrogate test calibrated under zero coupling (N, segment length, pooled vs exclude null)? | 1 | `summary.csv`, `fig_fpr_by_cell.png`, `fig_z_distribution.png` |
| S04 | `S04_power.py` | Power and direction accuracy vs gain, N, segment length; one group and group contrast | 1 | `summary.csv`, `fig_power_curves.png` |
| S05 | `S05_conditioning.py` | Indirect paths (cascade) and common drivers: 4-node dDTF/ffDTF/DTF vs bivariate | 1 | `summary.csv`, `fig_absent_edge_false_positives.png` |
| S06 | `S06_smoothness.py` | Smooth channel as spurious source; effect of the surrogate correction and of z-scoring | 1 | `summary.csv`, `fig_smoothness_artefact.png` |
| S07 | `S07_shared_stimulus.py` | Common stimulus, group-dependent gain, pooled vs within-group null, wrong-film null | 1 | `summary.csv`, `fig_stimulus_false_positives.png` |
| S08 | `S08_sensitivity.py` | Estimator, order, window, overlap, detrending, Box-Cox, band (one factor at a time, paired) | 1 | `summary.csv`, `fig_variants.png` |
| S09 | `S09_cardiac_carrier_mismatch.py` | Cardiac carrier-frequency gap, coupling-band width (RSA oscillators, option A) | 2 | `summary.csv`, `fig_gap_effect.png` |
| S10 | `S10_coupling_type_visibility.py` | Which coupling types are visible with raw IBI (A) vs RSA amplitude (A′, B) | 2 | `summary.csv`, `fig_coupling_type_visibility.png` |
| S11 | `S11_comparison.py` | dDTF vs pairwise Granger, lagged cross-correlation, coherence, PLV | 1–2 | `summary_partA/B.csv`, `fig_partA/B_methods.png` |

Every folder also has `raw_results*.csv` (one row per replicate × condition × edge) and `summary.json` (settings and
the pass/fail status of the proposed acceptance criteria).

## Shared code (`src/`)

- `sim_generators.py` – VAR generators with resonant nodes, stimulus drive with person-specific gain, phase oscillators, observation noise.
- `sim_cardiac.py` – RSA-type cardiac generator, EEG envelope generator, representations A / A′ / B.
- `sim_estimation.py` – vectorized DTF/ffDTF/dDTF/GPDC and windowed MVAR; identical to `src.connectivity.Granger_estimator`
  (`tests/test_sim_estimation.py`). A callable `estimator` is accepted for competitor statistics (S11).
- `sim_inference.py` – film-matched surrogate dyads, delta and robust z, dyad/study/group-level tests, replicate-study driver.
- `sim_harness.py` – parallel runner, seeds, `Results/<estimator>/S_xx` folders.

## Conventions

Connectivity matrices are `[target, source]`. Node order in Level-1 designs: 0 child ROI, 1 caregiver ROI, 2 child HRV,
3 caregiver HRV. Edge names read `source->target`. None of the tests assumes a symmetric or normal null:
- study level: one-sided pairing permutation test (alpha = .05). Under H0 (the persons of a dyad are independent given the
  film) the real pairing is exchangeable with re-pairings of children and caregivers, so the mean real value is compared
  with the means of 1000 randomly re-paired studies (re-paired within groups for the within-group null);
- dyad level: empirical one-sided p-value of the real value among its surrogate values, p < .05;
- group level: permutation test (1000 label permutations) of the difference in group means of delta.

Delta and robust z (`real - median(null)`, divided by the scaled MAD) are kept as descriptive readouts. A common model
order (p = 4) is used for real and surrogate fits except in S08.

## Run time (4 cores, full mode, extrapolated from 2-replicate runs)

S01 ≈ 4 min, S02 ≈ 6 min, S03 ≈ 15 min, S04 ≈ 1.7 h, S05 ≈ 9 min, S06 ≈ 6 min, S07 ≈ 26 min, S08 ≈ 15 min,
S09 ≈ 9 min, S10 ≈ 6 min, S11 ≈ 9 min; about 3.5 h in total per estimator (about 7 h for dDTF and GPDC). S04 dominates (3 N × 3 segment lengths × 6 gains × 2 designs).

## Points to check before the full runs are interpreted

- S02: the Stage-0 phase path confounds the frequency gap with the width of the frequency grid (ffDTF normalization); the
  `phase_fixedgrid` path was added, but its value under truth `none` still declines with the gap in the quick run. Check in the full run.
- S01/S05/S09: reverse or indirect edges are detected more often than the nominal level (leakage). Whether this is acceptable has to be
  decided from the full runs; the criteria in S05 test it explicitly.
- S09/S10 use a simple RSA generator (carrier × slow log-amplitude) and an envelope-level EEG node; real IBI has more structure
  (trend, ectopic beats, interpolation).
- S11 part (iv) uses PLV only; phase-based DTF is covered in S02.
