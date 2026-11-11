# Full re-run of S1–S13 with the pending code changes (2026-11-10)

Status: **done.** Code pushed (`22bdd3f..37c118b`, plus the `wilson_interval` fix `f84540d`); full re-run 2026-11-10 19:31 to
2026-11-11 04:38 (S1–S12 and variants 7 h 44 min; S13 9 h 7 min alongside). Every simulation ran in full mode for both
estimators: 200 replicates (S2, S9–S11, S13: 100), 1000 in the small-N cells. Tables: `dDTF/Results/comparison/`
(`comparison.md`, `criteria.csv` with `status`); copy for Drive in `dDTF/Results/for_drive/` (`~/mnt/2026_Multimodal_MVAR`
does not exist on this machine).

The manuscript DRAFT v0.4 is not in the repository (only `docs/draft_v02_…`), so "old vs new" (Section 2) lists the
main readouts per simulation from `Results/comparison/` rather than the exact numbers cited in v0.4.

## 1. Implementation notes for the manuscript (Sections 2.5 and 2.6)

**Sign-flip test of film contrasts (2.5, "Stimulus contrasts").** For each dyad and edge we formed the film contrast
c = Δ(film 1) − ½[Δ(film 2) + Δ(film 3)], with Δ taken against the within-group null of each film; a dyad missing a
film was left out for that edge only. Under the null hypothesis of no film effect each dyad's contrast is symmetric
about zero, so its sign is exchangeable: one matrix of random signs (1000 × number of dyads) was drawn and applied to
all eight cross-person edges, and the statistic of an edge was |½(mean_TD c + mean_ASD c)|, the film effect averaged
over groups. Family-wise error was controlled by max-T step-down over the eight edges. The film × group interaction
was tested with the same group-label permutation as the group contrast, applied to c (statistic |mean_ASD c −
mean_TD c|), again with max-T over the eight edges.

**Exact enumeration (2.5).** When a permutation test has no more distinct rearrangements than the number of
permutations (1000), all of them are used instead of random draws: with 5 vs 5 dyads the C(10, 5) = 252 group
relabellings, and likewise all sign vectors when 2ⁿ ≤ 1000 and all re-pairings when n! ≤ 1000. The observed
arrangement is excluded from the enumerated set, so the usual (1 + #{T_b ≥ T_obs}) / (1 + B) equals the exact
p-value #{arrangements with T ≥ T_obs} / (number of arrangements). The smallest attainable two-sided p-value with
5 vs 5 dyads is therefore 2/252 ≈ 0.008.

**Time-varying coupling (2.6, partial coupling).** To simulate coupling that is present during only part of a
recording, the coupling coefficient of the planted edge was switched on during one contiguous episode covering
100, 75, 50 or 25 % of the 60 s, starting at a random time in each dyad, and was zero otherwise. The process was
simulated as one continuous VAR whose coefficient matrix changes at the episode boundaries, not by concatenating
coupled and uncoupled segments, so there are no discontinuities in the signals. In the second variant a random
share q of the dyads (1, 0.75, 0.5, 0.25) was coupled for the whole recording and the others had gain 0.

**Observation noise (2.6, unequal signal-to-noise ratio).** White Gaussian noise was added to selected nodes after
the VAR simulation, so it is measurement noise and not part of the generating model. Its standard deviation was set
relative to the standard deviation of the node in that dyad (0, 0.25, 0.5, 1 or 2 times, i.e. SNR ∞ to 0.5), and
the series were z-scored afterwards, as in the pipeline. Noise was placed either on the caregiver heart or on the
child brain (no coupling), or on both caregiver nodes (with the three S1 edges planted).

## Reproducibility

New random draws use separate streams (`[seed, 1]` max-T, `[seed, 2]` partial coupling, `[seed, 3]` observation noise,
`spawn_seeds([BASE_SEED, 1])` for extra replicates and cells), so every readout of an unchanged design is reproduced.
Checked in the pilot on full settings (first 20 replicates, both estimators), against the full runs of 2026-11-10
(kept in `Results/_baseline_2026-11-10/`):

| Simulation | Readouts compared | Identical |
|---|---|---|
| S1 | study p, dyad rate, median Δ, max-T p (240 rows per estimator) | all |
| S3, N = 10, 60 s | study p, dyad rate, median z, study max-T p (480 rows) | all |
| S3, N = 10, 60 s | group p (per edge and max-T) | no — **expected**: exact enumeration (252 relabellings) replaces 1000 random draws. Group decisions changed in 6/480 (dDTF; rate 0.067 → 0.075 over these 20 replicates) and 0/480 (GPDC). |
| S12, first cell | asymmetry p, dyad rates, mean d, forward/reverse rejections | all |
| S4b, q = 1 and 100 % | study p, max-T p vs S4 design A (quick) | all |
| S6b, noise SD 0 | study p, median Δ vs S6 (r = 0.9, z-scored); study p and max-T p vs S1 | all |

Exact enumeration affects only cells with ≤ 1000 relabellings: N = 10 (5 vs 5) in S3, S4 design B and S13 is not
affected (15 vs 15). Its effect on the 200- and 1000-replicate N = 10 rates is in Section 3.

## Pilot (Part C)

- `run_all.py quick` over all 15 simulations × 2 estimators: 2 min 48 s; every run finished; progress lines,
  `progress.log` and `progress.json` worked (quick-mode numbers are not meant to be read).
- S13, full settings, 5 replicates × scenarios a–f per estimator: 19 min with 5 processes per estimator, i.e.
  ≈ 190 s per study and process (simulation and Δ 1 s, all permutation tests 0.1 s, M2 ≈ 90 s, M3 ≈ 20 s,
  M4 ≈ 62 s, fallback ≈ 31 s). Convergence-gate failures after the R-hat fix: M2 1–4 %, M3 7 %, M4 0–7 %
  (before the fix: 10–80 %). Pilot files: `Results/<est>/S_13/pilot_v2_2026-11-10/`.
- Nothing changed unexpectedly.

Actual full-run time: S1–S12 and variants 7 h 44 min, S13 9 h 7 min (alongside). Estimated beforehand: S1–S12 with the variants ≈ 5–6 h (previous ≈ 4.1 h plus ≈ 0.3 h of new
cells and variants, slowed by S13 running alongside); S13 100 replicates × 6 scenarios × 2 estimators = 1200 studies,
≈ 6.3 h for the first 50 replicates (5 processes) and ≈ 3.5 h for the second 50 (9 processes): ≈ 10 h in total.

## How the run was started and how to follow it

```bash
nohup zsh simulations_multimodal_MVAR/dDTF/Results/run_full_rerun.sh > /dev/null 2>&1 &
tail -f simulations_multimodal_MVAR/dDTF/Results/progress.log
cat simulations_multimodal_MVAR/dDTF/Results/progress.json
```

The script runs `run_all.py full 3 4 4b 1 12 5 6 6b 7 8 2 9 10 11` (both estimators) and, alongside it, S13 in chunks
(`SIM_REPS=0-49`, then `50-99`, dDTF and GPDC), then `compare_estimators.py`; its last line in `progress.log` is
`[run_full_rerun] all done`.

## 2. Old vs new

Old = full runs of 2026-11-09/10 (`Results/_baseline_2026-11-10/`), new = this run, same seeds.

**Unchanged designs: identical.** Every readout of every summary table of S1, S2, S5, S6, S7, S8, S9, S10, S11 and S12 is
identical to the old run, for both estimators (all rows, all readouts in `compare_estimators.TABLES`). In S3 and S4,
replicates 0–199 are identical in every study-level, dyad-level and z readout (43 200 and 172 800 raw rows); the only
differences are the group-level tests at N = 10 (5 vs 5), which now use exact enumeration (S3: 14 279 of 43 200 group p
differ, all at N = 10; S4: design B at N = 10 only). Summary rates of the N = 10 cells also change because they now pool
1000 replicates. **No other number differs; nothing to flag.**

**N = 10 group contrast, old (200 replicates, random relabelling) vs new (1000, exact).** S3, mean over the 8 edges:

| Estimator | Null | 60 s old → new | 120 s old → new | 300 s old → new |
|---|---|---|---|---|
| dDTF | pooled | 0.051 → 0.049 | 0.049 → 0.048 | 0.052 → 0.051 |
| dDTF | exclude | 0.049 → 0.050 | 0.048 → 0.050 | 0.056 → 0.053 |
| dDTF | within-group | 0.066 → 0.069 | 0.063 → 0.073 | 0.077 → 0.075 |
| GPDC | pooled | 0.041 → 0.047 | 0.046 → 0.050 | 0.048 → 0.049 |
| GPDC | exclude | 0.041 → 0.046 | 0.045 → 0.048 | 0.053 → 0.048 |
| GPDC | within-group | 0.073 → 0.081 | 0.077 → 0.080 | 0.082 → 0.082 |

S4 design B, 5 vs 5, 60 s (power of the planted edge; gain 0 = false-positive rate):

| Estimator | Gain | per edge old → new | max-T old → new | min-p old → new | Bonferroni old → new |
|---|---|---|---|---|---|
| dDTF | 0 | 0.030 → 0.056 | 0.015 → 0.013 | 0.010 → 0.002 | 0.005 → 0 |
| dDTF | 0.2 | 0.405 → 0.449 | 0.140 → 0.178 | 0.105 → 0.006 | 0.040 → 0 |
| dDTF | 0.4 | 0.985 → 0.981 | 0.845 → 0.841 | 0.505 → 0.045 | 0.115 → 0 |
| GPDC | 0 | 0.085 → 0.074 | 0.020 → 0.010 | 0.020 → 0.002 | 0.015 → 0.002 |
| GPDC | 0.2 | 0.450 → 0.494 | 0.180 → 0.200 | 0.135 → 0.029 | 0.060 → 0.028 |
| GPDC | 0.4 | 1.000 → 0.986 | 0.845 → 0.858 | 0.595 → 0.122 | 0.235 → 0.104 |

**Flag (min-p and Bonferroni at 5 vs 5).** With exact enumeration, min-p and Bonferroni lose almost all power at 5 vs 5.
This is the true discreteness of the test, which the random draws had hidden: the smallest attainable two-sided p of an
edge is 2/252 = 0.0079 (a relabelling and its swap give the same |difference|), so Bonferroni (× 8 = 0.063) can never
reject, and about 16 of the 251 relabellings reach that minimum across the 8 edges, so the min-p adjusted p cannot fall
much below 0.067. Max-T, which compares standardized statistics rather than p-values, is unaffected. The remaining
non-zero values (mostly GPDC) presumably come from studies in which a dyad drops out of an edge (non-finite z), which
breaks the swap symmetry and halves the smallest attainable p. Recommendation: in the
manuscript, mention min-p only as a sensitivity check at larger N, and state that it and Bonferroni cannot be used at
5 vs 5.

**Power thresholds (Section 3.3 / Table 3): unchanged.** Smallest gain with ≥ 80 % power at 60 s, both estimators:
design A 0.2 (N = 10), 0.2 (N = 25), 0.1 (N = 50), the same after max-T; design B 0.3 / 0.2 / 0.2, after max-T
0.4 / 0.3 / 0.2. No threshold in any cell changed (per edge, max-T, min-p, Bonferroni).

**Criteria: no existing verdict changed.** All criteria of S1–S12 have the same pass/fail as in the old run; the new
criteria are those of S04b, S06b and S13 (Section 4).

Draft v0.4 is not in the repository, so the numbers above are the readouts that changed or that the manuscript is
known to cite (thresholds, S3 calibration, S4 design B); every other number is unchanged (see above).

## 3. New results

### 3.1 S5: omitted driver — sweep and collinearity (Section 3.4)

Study-level rate of the heart–heart edge (no true edge) in the 5-node model with Z, N = 25, and the median
diagnostics of the 5-node fit (real dyads; pseudo-dyads are within 0–5 % of these):

| Estimator | Gain | 60 s, p = 4 | 120 s | 300 s | p = 2 (60 s) | p = 6 (60 s) |
|---|---|---|---|---|---|---|
| dDTF | 0.4 | 0.065 | 0.180 | 0.710 | 0.135 | 0.015 |
| dDTF | 0.6 | 0.525 | 0.825 | 1.000 | 0.645 | 0.180 |
| GPDC | 0.4 | 0.105 | 0.360 | 0.860 | 0.120 | 0.185 |
| GPDC | 0.6 | 0.640 | 0.950 | 1.000 | 0.290 | 0.540 |
| VIF, caregiver-heart lags | 0.6 | 41 | 39 | 39 | 5.2 | 58 |
| condition number | 0.6 | 25 | 24 | 23 | 5.9 | 37 |

(Main design, 60 s, p = 4: the 4-node model without Z detects the edge in 100 % of studies at every gain; the 5-node
model with Z in 1.5 / 6.5 / 45.5 % (dDTF) and 4.5 / 13.5 / 62.5 % (GPDC) at gains 0.2 / 0.4 / 0.6; corr(Z, caregiver
heart) at lag 0 is 0.11–0.16.)

*Draft text.* The residual heart–heart detection with Z in the model is not a small-sample collinearity effect: it grows
with segment length (at gain 0.6, from 0.53 at 60 s to 1.00 at 300 s for dDTF) and therefore reflects a systematic
difference between real and pseudo-dyads that more data resolve better, not noise. It does not track the collinearity
diagnostics: the VIF of the caregiver-heart lags and the condition number barely change with segment length, rise
steeply with model order, and are the same in pseudo-dyads as in real dyads. A higher model order reduced the excess for
dDTF (0.18 at p = 6) but not consistently for GPDC (0.29 / 0.64 / 0.54 at p = 2 / 4 / 6). Conditioning on a measured
dyad-level driver therefore removes most, but at strong drive and long recordings not all, of its spurious heart–heart
edge. *[Mechanism not established here; candidates are the per-window detrending of a smooth, strongly driven process
and the finite model order; worth a short targeted check before the text is final.]*

### 3.2 Small-N group contrast at 1000 replicates (Section 3.2, Box 2)

Group-contrast false-positive rate (S3), with 95 % Wilson intervals. Family-wise max-T rate over the 8 edges (one value
per study, so the interval is exact for the study as the unit):

| Estimator | N | Null | Per-edge rate (mean of 8) | Max-T family-wise rate [95 % CI] |
|---|---|---|---|---|
| dDTF | 10 (5 vs 5), 60 s | pooled / exclude | 0.049 / 0.050 | 0.061 [0.048, 0.078] / 0.063 [0.050, 0.080] |
| dDTF | 10, 60 / 120 / 300 s | within-group | 0.069 / 0.073 / 0.075 | 0.082 [0.067, 0.101] / 0.089 [0.073, 0.108] / 0.100 [0.083, 0.120] |
| dDTF | 16 (8 vs 8), 60 s | within-group | 0.058 | 0.071 [0.057, 0.089] |
| GPDC | 10, 60 s | pooled / exclude | 0.047 / 0.046 | 0.060 [0.047, 0.076] / 0.063 [0.050, 0.080] |
| GPDC | 10, 60 / 120 / 300 s | within-group | 0.081 / 0.080 / 0.082 | 0.113 [0.095, 0.134] / 0.104 [0.087, 0.124] / 0.117 [0.099, 0.138] |
| GPDC | 16, 60 s | within-group | 0.074 | 0.081 [0.066, 0.100] |

At 200 replicates, within-group null, per-edge rate: dDTF 0.048–0.061 at N = 25 and 0.038–0.049 at N = 50; GPDC
0.061–0.073 at N = 25 and 0.051–0.066 at N = 50. S4 design B at 5 vs 5, gain 0 (within-group null, planted edge):
dDTF 0.056 [0.043, 0.072], GPDC 0.074 [0.059, 0.092].

*Draft text.* With the pooled or exclude null the group contrast is calibrated already at 5 vs 5 dyads (per-edge rate
0.046–0.053, intervals containing 0.05). With the within-group null, which builds each group's null from its own
dyads only, the contrast is liberal at small N: the interval excludes 0.05 at 5 vs 5 for both estimators (dDTF ≈ 0.07,
GPDC ≈ 0.08 per edge) and still at 8 vs 8 (0.058 and 0.074). For dDTF the rate is nominal from about 12–13 dyads per
group; for GPDC it stays at 0.06–0.07 even with 25 per group. *Recommendation:* with the within-group null, use at least
12–13 dyads per group for dDTF; for GPDC the limitation "Group contrasts with GPDC" stays (the interval excludes 0.05
at every N tested up to 25 per group), so the manuscript should keep it.

### 3.3 Partial coupling (Section 3.3)

S4b, design A, N = 25, 60 s, 200 replicates. Power per edge / after max-T; reverse-edge rate per edge.

| Estimator | Variant | Gain | 100 % | 75 % | 50 % | 25 % |
|---|---|---|---|---|---|---|
| dDTF | share of dyads coupled (q) | 0.3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.97 / 0.82 |
| dDTF | share of dyads coupled (q) | 0.4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 |
| dDTF | share of time coupled | 0.3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.55 / 0.25 |
| dDTF | share of time coupled | 0.4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.83 / 0.60 |
| GPDC | share of dyads coupled (q) | 0.3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.86 / 0.49 |
| GPDC | share of dyads coupled (q) | 0.4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.95 / 0.80 |
| GPDC | share of time coupled | 0.3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.46 / 0.17 |
| GPDC | share of time coupled | 0.4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.83 / 0.60 |

Median Δ of the planted edge scales with the coupled share (dDTF gain 0.4: 0.47 / 0.40 / 0.18 / 0.06 for q = 1 / 0.75 /
0.5 / 0.25; GPDC 0.86 / 0.79 / 0.44 / 0.15). Dyad-level detection: coupled dyads 0.76–0.80 (gain 0.3) and 0.92–0.95 (gain 0.4)
at every q; uncoupled dyads 0.047–0.060 (upper 95 % limit ≤ 0.069) — criterion ≤ 0.10 met. Reverse edge (dDTF only)
falls with q (gain 0.4: 1.00 / 1.00 / 0.96 / 0.42); GPDC ≤ 0.075 throughout. q = 1 and 100 % reproduce S4 design A
exactly (criterion met).

*Draft text.* The study-level test stays powerful when only part of the sample is coupled: with N = 25 and gain ≥ 0.3,
power is 1.00 down to half of the dyads coupled and 0.86–1.00 with a quarter (fewer after max-T, 0.49–0.99). Coupling
present during only part of the recording costs more: half of the 60 s still gives full power, a quarter gives 0.46–0.83.
The dyad-level test does not spread coupling to uncoupled dyads (false-positive rate 0.05–0.06), and the median Δ
shrinks in proportion to the coupled share, so a moderate group-level Δ can reflect strong coupling in a subset of dyads.

### 3.4 Unequal signal-to-noise ratio (Section 3.5)

S6b, no coupling (S6 design, caregiver-heart pole radius 0.9), N = 25, 60 s, 200 replicates. Noise SD relative to the
node SD: 0 / 0.25 / 0.5 / 1 / 2.

| Estimator | Noise on | Uncorrected asymmetry | Corrected asymmetry | FPR smooth→rough / rough→smooth (range) |
|---|---|---|---|---|
| dDTF | caregiver heart | 0.183 / 0.100 / 0.031 / −0.021 / −0.031 | 0.002 to −0.000 (all \|·\| ≤ 0.002) | 0.055–0.075 / 0.020–0.050 |
| dDTF | child brain | 0.183 / 0.190 / 0.200 / 0.204 / 0.196 | −0.002 to 0.002 | 0.015–0.075 / 0.030–0.045 |
| GPDC | caregiver heart | 0.011 / −0.013 / −0.019 / −0.013 / −0.028 | 0.001 to 0.014 | 0.035–0.080 / 0.035–0.065 |
| GPDC | child brain | 0.011 / 0.012 / 0.019 / 0.037 / 0.063 | −0.003 to 0.003 | 0.035–0.065 / 0.035–0.055 |

With coupling (S1 edges, noise on both caregiver nodes, SD 0 / 0.5 / 1): power and direction accuracy 1.00 for all three
edges with both estimators. Reverse-edge rate: dDTF 0.98–1.00 at every noise level (as without noise); GPDC rises with
noise for the heart-source edges (cg:HRV→child:HRV 0.02 → 0.035 → 0.19; cg:HRV→child:ROI 0.00 → 0.015 → 0.22; after
max-T ≤ 0.07).

*Draft text.* Unequal measurement noise changes the raw directional asymmetry — noise on the smooth caregiver heart
removes, and at SD ≥ 1 reverses, the smoothness artefact of dDTF (0.18 → −0.03), and noise on the child brain enlarges
the GPDC asymmetry (0.01 → 0.06) — but the surrogate correction cancels it in every condition (corrected asymmetry within
±0.014, criterion ±0.02 met) and the corrected false-positive rate stays at 0.02–0.08 (criterion ≤ 0.10 met). With
coupling, noise on the source person leaves power and direction accuracy at 1.00 but lets GPDC detect the reverse of
heart-source edges in about a fifth of studies at noise SD equal to the signal SD (0.06–0.07 after max-T), the familiar
noise-induced reversal of Granger causality.

### 3.5 S13: group, film and interaction effects (new Section 3.10)

2 × 15 dyads, 3 films × 60 s, 100 replicates per scenario. Detection on planted edges (Δ; Bayesian models on the
within-edge standardized Δ), dDTF / GPDC:

| Scenario: effect | max-T within film | max-T over 24 | max-T on film mean | sign-flip / interaction max-T | M2 per edge | M3 (1\|edge) | M4 (film·group\|edge) |
|---|---|---|---|---|---|---|---|
| c: group, brain–brain | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | – | 1.00 / 1.00 | 0.73 / 0.52 | 1.00 / 1.00 |
| d: group, both edges | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | – | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| e: group (film 1 only) | 0.97 / 0.98 | 0.92 / 0.95 | 0.20 / 0.58 | – | 0.53 / 0.91 | 0.12 / 0.06 | 0.34 / 0.34 |
| e: film | – | – | – | 0.67 / 0.78 | 1.00 / 0.99 | 0.28 / 0.17 | 0.59 / 0.69 |
| e: interaction | – | – | – | 0.91 / 0.94 | 0.98 / 0.99 | 0.36 / 0.15 | 0.65 / 0.64 |
| f: film | – | – | – | 1.00 / 1.00 | 1.00 / 1.00 | 0.88 / 0.57 | 1.00 / 1.00 |

Family-wise rate on null edges, scenario a (no effect anywhere), dDTF / GPDC: group contrast max-T per film 0.057 /
0.103 (per film; 0.16 / 0.28 if any of the 3 films counts), max-T over 24 0.07 / 0.12, max-T on film mean 0.05 / 0.08;
film effect (sign-flip max-T) 0.05 / 0.07; interaction max-T 0.08 / 0.12; uncorrected 0.31–0.85. Bayesian (95 % HDI
excludes 0 on any null edge): M2 0.20–0.44, M3 0.02–0.12, M4 0.04–0.08.

Leakage (scenario c, group effect on any null edge): M4 0.93 / 0.41 over all null edges, 0.21 / 0.34 excluding the
reverse of the planted edge; per-film max-T 0.65 / 0.22 and 0.14 / 0.20; film-mean max-T 0.71 / 0.08 and 0.06 / 0.08.
In scenario f the sign-flip test detects the film effect on the reverse edge with dDTF (family-wise 0.71, 0.07 without
the reverse edge; GPDC 0.07).

Shrinkage of the standardized group effect on planted edges (bias, RMSE): M2 ≈ unbiased (|bias| ≤ 0.06, RMSE 0.10–0.18,
as the unpooled estimate); M4 shrinks toward 0 by 0.05–0.23 (RMSE 0.11–0.28), most in scenario e; M3 cannot localize an
edge-specific effect (bias 0.4–1.3). Convergence gate (R-hat < 1.01, ESS > 400, no divergences; after the R-hat fix):
failures M2 1.5–2.5 %, M3 1.8–2.8 %, M4 9.3–9.8 % (divergences in 6.7–8.0 %), fallback 6.7–9.6 %. Runtime per study
≈ 3.3 min: M2 83–96 s (8 fits), M3 20–22 s, M4 59–67 s, fallback 30–34 s; simulation 1 s; all permutation tests 0.1 s.

*Draft text.* Permutation tests with max-T detected every planted group effect that was present in all films (power
1.00), and the film-specific group difference of scenario e when tested within film (0.97–0.98) but not on the film mean
(0.20–0.58). The new sign-flip test detected a film effect shared by both groups in every study (scenario f) and the film
× group interaction of scenario e in 0.91–0.94, with family-wise rates near nominal for dDTF (0.05–0.08) and somewhat
above it for GPDC (0.07–0.12), in line with the liberal within-group null of GPDC (Section 3.2). Bayesian per-edge models
estimated effects without bias but, lacking multiplicity control, flagged at least one null edge in 20–44 % of studies;
partial pooling over edges (M4) controlled that (4–8 %) at the cost of shrinking a single strong edge by up to 40 % and
of lower power for effects confined to one film; a random intercept per edge alone (M3) cannot represent edge-specific
effects. Leakage of a group effect into other edges under M4 is mostly the dDTF reverse-edge bias (Section 3), which is in
the data, not created by pooling (0.21 without the reverse edge).

### 3.6 max-T (Section 4.5 TBDs)

S3, family-wise false-positive rate over the 8 cross-person edges, 10 cells (N × segment, incl. N = 16) × 3 nulls:
uncorrected 0.24–0.43 (dDTF) / 0.28–0.50 (GPDC). Max-T step-down, study level: 0.025–0.085 (mean 0.052–0.054, dDTF) and
0.025–0.070 (mean 0.054–0.056, GPDC); group contrast, pooled and exclude nulls: 0.040–0.070 (dDTF) and 0.025–0.085
(GPDC); within-group null: 0.040–0.100 (mean 0.063, dDTF) and 0.040–0.117 (mean 0.078, GPDC), the excess at N ≤ 16 for
dDTF and mostly at N ≤ 16 for GPDC (Section 3.2). S4 power after max-T: thresholds in Section 2 (design A unchanged, design B +0.1 at N = 10 and 25).
Non-planted edges after max-T: the dDTF reverse edge stays high (Section 3 of the earlier report; unchanged).

## 4. Criteria

`Results/comparison/criteria.csv`, column `status` (`pass` = met by both estimators; `proposed: ` marks criteria whose
thresholds are not yet fixed). Met dDTF / GPDC:

| Sim | Met | Not met (dDTF / GPDC) |
|---|---|---|
| S1 | 4/4, 4/4 | – |
| S2 | 3/3, 3/3 | – |
| S3 | 2/5, 2/5 | group-level FPR within band in every cell (✗/✗, within-group null at N ≤ 16; GPDC also N = 25, 300 s); max-T FWER within band in every cell, study level (✗/✗) and group level (✗/✗) |
| S4 | 3/3, 3/3 | – |
| S4b | 2/2, 2/2 [proposed] | – |
| S5 | 5/7, 6/7 | cascade absent indirect edge ≤ 0.1 (✗/✓); bivariate worse than 4-node in the cascade (✗/✓); omitted driver with Z ≤ 0.1 (✓/✗) |
| S6 | 4/4, 2/4 | GPDC: artefact present / grows with smoothness (expected: GPDC has no smoothness artefact) |
| S6b | 2/2, 2/2 [proposed] | – |
| S7 | 4/4, 4/4 | – |
| S8 | 3/3, 3/3 | – |
| S9 | 2/4, 2/4 | pulling: power at the largest gap below gap 0; amplitude-only link not seen by raw IBI |
| S10 | 4/6, 4/6 | A does not detect amp_to_eeg; A′ does not detect linear |
| S11 | 6/7, 7/7 | dDTF: absent indirect edge ≤ 0.1 |
| S12 | 4/5, 5/5 | dDTF: reverse edge of one-way links ≤ 0.1 |
| S13 | 5/7, 4/7 [proposed] | M4 leakage ≤ 0.1 (✗/✗); interaction max-T FWER in band, scenario a (✗ 0.08 / ✗ 0.12); GPDC: group max-T within film in band (0.103) |

**Criteria that should change.**
- S3 "max-T FWER within 0.03–0.07 in every cell": fails by chance at 200 replicates (a calibrated test misses the band in
  about 1 cell in 5); replace by "the Wilson 95 % interval overlaps the band in every cell" (study level: met for both;
  group level: fails only for the within-group null at N ≤ 16, which is a real effect, Section 3.2).
- S3 "group-level FPR within band in every cell": report the within-group null separately; for the pooled and exclude
  nulls it is met in every cell.
- S6 "artefact present / grows" for GPDC: GPDC has no smoothness artefact by construction; state these two as dDTF-only.
- S13 "M4 leakage ≤ 0.1": score on null edges excluding the reverse of the planted edge (0.21 / 0.34 — still above 0.1;
  consider ≤ 0.25, or report leakage descriptively).
- S13 interaction and GPDC FWER criteria: at 100 replicates the interval is wide (dDTF interaction 0.08 [0.04, 0.15]);
  use "interval overlaps the band", as for S3.
- S5 omitted driver: the 5-node criterion is met at gain 0.4 for dDTF only; given Section 3.1, state the limitation for
  strong drive and long recordings instead of a pass/fail at one gain.

## 5. Commits

| Commit | Content |
|---|---|
| `8054cbd` | A2: exact enumeration in small permutation tests |
| `0c2b5d3` | A1: S13 sign-flip test of film contrasts, interaction test, scenario f |
| `0b1d943` | A3: S5 collinearity diagnostics and omitted-driver sweep |
| `d1029c7` | A4: S3 and S4 small-N cells with 1000 replicates |
| `8b29807` | A5: S4 variant with partial coupling (S04b) |
| `a6381ce` | A6: S6 variant with unequal signal-to-noise ratio (S06b) |
| `ce8722f` | A7: S7 figure with the other-film null; band comments; registration of the variants |
| `37c118b` | A8: progress reporting for the simulation runs |
| `f84540d` | `wilson_interval`: bounds kept within [0, p] and [p, 1] (fixes the S13 summary crash after the second chunk) |

Commit to cite in the Data availability statement: **`f84540d`** — the simulations ran on `37c118b`; `f84540d` changes only
how Wilson bounds are clipped (≤ 1e-16) and is the commit that produced the final S13 summaries. The report commit follows.

Deviations from the task text: the film-contrast methods are named `perm_contrast_maxT` / `perm_contrast_unadj`
because `perm_film_*` already names the per-film group test in S13; `criteria.csv` `status` is `pass` / `fail`, with
`proposed: ` in front for criteria marked [proposed], so proposed criteria keep their pass/fail verdict.
