# Simulations S01–S13 and the variants S04b, S06b (Table 2 of the methods paper)

Every script follows the numbering of Table 2 of `docs/draft_v02_ddtf_multimodal_hyperscanning_simulations.md`
and is run once per connectivity estimator, dDTF (the proposed method) or GPDC. Results go to
`simulations_multimodal_MVAR/dDTF/Results/<estimator>/S_xx/` (`xx` = simulation number). The `Results/` folder is not versioned (`.gitignore`); regenerate it with `run_all.py`.

```bash
python simulations_multimodal_MVAR/dDTF/S03_null_calibration.py quick          # smoke test (seconds), dDTF
python simulations_multimodal_MVAR/dDTF/S03_null_calibration.py full GPDC      # full protocol, GPDC
python simulations_multimodal_MVAR/dDTF/run_all.py quick                       # all scripts, dDTF and GPDC, then the comparison
python simulations_multimodal_MVAR/dDTF/run_all.py full 3 4 7                  # selected scripts, both estimators
python simulations_multimodal_MVAR/dDTF/run_all.py full 4b 6b                  # the variants (S04b partial coupling, S06b unequal SNR)
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

## Following a run

Every script prints a start line (mode, replicates, conditions, tasks, N_JOBS), a progress line every 5% of its tasks or
every 60 s (`[S04 GPDC] 312/1800 tasks (17%) | elapsed 0:41:10 | ETA 3:18:00 | last: n_dyads=25, seg_s=60, gain=0.3`),
a summary line with its main readouts and the number of criteria met, and an end line with the time and the results
folder; S13 adds one line per study and a warning for every Bayesian fit that fails the convergence gate. `run_all.py`
adds one line per finished run. All lines are also appended to `Results/progress.log`, and `Results/progress.json`
(rewritten after every update) holds the run_all state, the running tasks and the finished runs:

```bash
tail -f simulations_multimodal_MVAR/dDTF/Results/progress.log
cat simulations_multimodal_MVAR/dDTF/Results/progress.json
```

`SIM_QUIET=1` switches the per-task lines off (start, summary and end lines stay). Reporting never changes results.

## Comparing dDTF and GPDC

The estimator replaces dDTF everywhere it is the method under test: the 4-node analysis of S01–S10, the
reference method of S11, the baseline of S08 (whose estimator variants then include dDTF), the envelope and phase
paths of S02 and S01 Part A (repository `gen_partial_directed_coherence`). The fixed comparisons (ffDTF and DTF in S05,
the competitors in S11) are computed in both runs and are identical. Both runs use the same seeds, so they analyse
the same simulated studies and every comparison is paired.

`run_all.py` ends with `compare_estimators.py`, which writes `Results/comparison/`:

- `criteria.csv`: every acceptance criterion, met or not, per estimator, and `status` (`pass` = met by both estimators,
  `fail`; prefixed `proposed: ` for criteria marked `[proposed]`) for Supplementary Table S1;
- `runs.csv`: mode and number of replicates of each run;
- `S_xx_<table>.csv`: the main readouts of each summary table, with the dDTF and GPDC columns next to each other;
- `comparison.md`: all of these as Markdown tables, ready to copy into the manuscript.

Labels that contain the estimator's name (`4-node dDTF`, `dDTF (4-node)`, criteria) appear as `[est]` so that the rows
of both runs line up. The readouts shown per table are listed in `TABLES` at the top of `compare_estimators.py`.

| ID | Script | Question | Level | Main outputs |
|---|---|---|---|---|
| S01 | `S01_recovery_orientation.py` | Are planted edges recovered and correctly oriented? Fast estimator = repo estimator? | 1 | `partB_summary.csv`, `fig_partA_*`, `fig_partB_recovery.png` |
| S02 | `S02_envelope_vs_phase.py` | Envelope vs phase estimators under amplitude and phase truths, centre-frequency gap | 2 | `summary.csv`, `fig_edge_values_by_truth.png`, `fig_direction_index.png` |
| S03 | `S03_null_calibration.py` | Is the surrogate test calibrated under zero coupling (N, segment length, pooled vs exclude null)? The N = 10 cells and an N = 16 cell (8 vs 8) run to 1000 replicates (`summary_small_n.csv`). | 1 | `summary.csv`, `fig_fpr_by_cell.png`, `fig_z_distribution.png` |
| S04 | `S04_power.py` | Power and direction accuracy vs gain, N, segment length; one group and group contrast. Design B at 5 vs 5, 60 s, gains 0/0.2/0.4 runs to 1000 replicates (`summary_small_n_group.csv`). | 1 | `summary.csv`, `fig_power_curves.png` |
| S04b | `S04b_partial_coupling.py` | Partial coupling (design A, N = 25, 60 s): only a share q of the dyads coupled, or coupling on during part of the 60 s (one continuous simulation); power per edge and after max-T, reverse rate, dyad-level rate in coupled vs uncoupled dyads | 1 | `S_04/partial/summary.csv`, `fig_partial_coupling` |
| S05 | `S05_conditioning.py` | Indirect paths (cascade), cross-person common driver, omitted dyad-level driver (with/without it as a 5th node): 4-node estimator/ffDTF/DTF vs bivariate. Collinearity diagnostics (condition number, caregiver-heart VIF) and an omitted-driver sweep over segment length and model order (`summary_sweep.csv`, `fig_collinearity_sweep`). | 1 | `summary.csv`, `fig_absent_edge_false_positives.png` |
| S06 | `S06_smoothness.py` | Smooth channel as spurious source; effect of the surrogate correction and of z-scoring | 1 | `summary.csv`, `fig_smoothness_artefact.png` |
| S06b | `S06b_unequal_snr.py` | Unequal SNR: white observation noise (SD 0–2 × node SD) on the caregiver heart or the child brain (no coupling, S6 design), or on the caregiver nodes (S1 edges); asymmetry, FPR, power | 1 | `S_06/noise/summary_no_coupling.csv`, `summary_coupled.csv`, `fig_unequal_snr` |
| S07 | `S07_shared_stimulus.py` | Common stimulus, group-dependent gain, pooled vs within-group null, wrong-film null | 1 | `summary.csv`, `fig_stimulus_false_positives.png` |
| S08 | `S08_sensitivity.py` | Estimator, order, window, overlap, detrending, Box-Cox, band (one factor at a time, paired) | 1 | `summary.csv`, `fig_variants.png` |
| S09 | `S09_cardiac_carrier_mismatch.py` | Cardiac carrier-frequency gap, coupling-band width (RSA oscillators, option A) | 2 | `summary.csv`, `fig_gap_effect.png` |
| S10 | `S10_coupling_type_visibility.py` | Which coupling types are visible with raw IBI (A) vs RSA amplitude (A′, B) | 2 | `summary.csv`, `fig_coupling_type_visibility.png` |
| S11 | `S11_comparison.py` | dDTF vs pairwise Granger, lagged cross-correlation, coherence, PLV | 1–2 | `summary_partA/B.csv`, `fig_partA/B_methods.png` |
| S12 | `S12_mutual_coupling.py` | Mutual (bidirectional) coupling: detection of both directions, false 'mutual' from one-way links, dominant direction and a sign-flip asymmetry test | 1 | `summary.csv`, `fig_mutual_coupling.png` |
| S13 | `S13_group_film_bayes.py` | Group and film effects across the 8 cross-person edges: permutation max-T vs Bayesian models with edge as a grouping factor (per edge, `(1\|edge)`, `(film*group\|edge)`); detection, family-wise rate on null edges, shrinkage and leakage. Film effect (sign-flip, max-T) and film × group interaction (label permutation, max-T) on the contrast film 1 − mean(films 2, 3); scenarios a–f. | 1 | `summary_detection.csv`, `summary_familywise_null.csv`, `summary_shrinkage.csv`, `summary_gate.csv`, `fig_group_film_bayes.png` |

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
- group level: permutation test (1000 label permutations) of the difference in group means of delta. When a test has
  no more distinct rearrangements than N_PERM (5 vs 5 dyads: 252 relabellings), all of them are used (exact test;
  `label_permutations`, `caregiver_permutations`, `sign_flips` in `src/sim_inference.py`);
- film contrasts (S13): sign-flip test of the per-dyad contrast (`signflip_maxT`), interaction by label permutation;
- multiplicity: the 8 cross-person edges form one family, tested on one shared set of permutations; max-T (per-edge
  standardization, Westfall-Young step-down), min-p and Bonferroni adjusted p-values (`pairing_permutation_maxT`,
  `group_permutation_maxT`). Within-person edges cannot be tested against pseudo-dyads and are never in a family.

Delta and robust z (`real - median(null)`, divided by the scaled MAD) are kept as descriptive readouts. A common model
order (p = 4) is used for real and surrogate fits except in S08.

## Run time (measured 2026-11-10/11, Apple M2 Pro, full mode, dDTF / GPDC)

S03 12 min / 10 min, S04 1.6 h / 1.3 h, S04b 9 min / 7 min, S01 29 s / 22 s, S12 4 min / 3 min, S05 27 min / 23 min, S06 4 min / 3 min, S06b 5 min / 4 min, S07 29 min / 24 min, S08 24 min / 19 min, S02 5 min / 5 min, S09 10 min / 8 min, S10 7 min / 6 min, S11 20 min / 17 min. In total S1–S12 with the variants took 7 h 44 min (one script at a time, 4 processes), with S13 running alongside:
1200 studies (100 replicates × 6 scenarios × 2 estimators) in 9 h 7 min with 5, then 9 processes (≈ 3.3 min per study,
almost all Bayesian fitting). `Results/run_full_rerun.sh` runs both streams; times are in `Results/run_all_timing_full.json`.

## Points to check before the full runs are interpreted

- S02: the Stage-0 phase path confounds the frequency gap with the width of the frequency grid (ffDTF normalization); the
  `phase_fixedgrid` path was added, but its value under truth `none` still declines with the gap in the quick run. Check in the full run.
- S01/S05/S09: reverse or indirect edges are detected more often than the nominal level (leakage). Whether this is acceptable has to be
  decided from the full runs; the criteria in S05 test it explicitly.
- S09/S10 use a simple RSA generator (carrier × slow log-amplitude) and an envelope-level EEG node; real IBI has more structure
  (trend, ectopic beats, interpolation).
- S11 part (iv) uses PLV only; phase-based DTF is covered in S02.
