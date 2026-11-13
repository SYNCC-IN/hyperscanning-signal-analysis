# Diagnostic checks before submission: A2 (group contrast null) and A4 (omitted driver)

Date: 2026-11-13. Branch `simulations_multimodal_MVAR`. The manuscript is not edited. Existing functions and results are
unchanged; new behaviour comes only from new options, all with defaults that keep the old behaviour.

Code:
- `src/sim_inference.py`: option `renull` in `group_permutation_maxT`, `edge_summary`, `multiplicity_records`,
  `score_study` and `run_study`. New helpers: `group_null_medians`, `renull_delta`, `group_mean_difference`,
  `renull_statistics`. The film × group interaction is tested by `group_permutation_maxT(..., renull=True, combine=...)`,
  with the films stacked as columns and `combine` mapping them to F1 − mean(F2, F3).
- `src/sim_estimation.py`: `detrend_type="none"` in `fit_mvar` / `edge_matrix` (windows fitted as cut), and
  `conditional_gc` (estimated conditional Granger F from the least-squares fit).
- `tests/test_sim_renull.py`: 5 tests, including the uniformity test.
- Scripts: `simulations_multimodal_MVAR/dDTF/CHK_A2_renull.py` and `CHK_A4_omitted_driver.py`.
- Outputs (raw CSVs, summaries, figures): `simulations_multimodal_MVAR/checks_2026-11/A2/` and `.../A4/`.

All rates come with 95% Wilson intervals. "Current" means the existing test (labels permuted on a fixed delta).
"renull" means the within-group null median of every dyad is recomputed from the full pair table under every permuted
labelling. Both versions use the same data and the same permutations.

---

## A2: is the within-group group contrast liberal because the null medians are fixed?

### Unit test (exchangeable synthetic data)

Setup: i.i.d. N(0,1) real values and pair table, 8 vs 8, 2000 replicates, 200 permutations.

| version | rate (p < 0.05) | KS p (uniform) |
|---|---|---|
| current | 0.0725 | 8.3e-05 |
| renull | 0.0530 | 0.125 |

The current test is liberal even on perfectly exchangeable data. That is the predicted mechanism. A rough calculation
gives the same size: the variance of the observed difference is σ²/4 + 2·Var(median of 56 pairs) ≈ 0.31σ², while the
permutation variance is ≈ 0.25σ². That predicts about 7.5%. `renull` is exactly calibrated.

### Reruns of the null cells

Data:
- **S3, N = 16:** the same 1000 studies as S3 (S03's extra seed stream).
- **S3, N = 25 and 50:** S03's 200 studies plus a new stream.
- **S13 scenario a:** S13's own seeds, reps 0–199. Reps 0–99 are the studies of the full run.
- Per-edge rates pool 8 edges × studies.

| cell | measure | dDTF current | dDTF renull | GPDC current | GPDC renull |
|---|---|---|---|---|---|
| S3, 8 vs 8 (1000) | per-edge | 0.061 (0.056–0.066) | **0.053** (0.048–0.058) | 0.074 (0.069–0.080) | **0.050** (0.045–0.055) |
| S3, 8 vs 8 | FWER, max-T over 8 | 0.069 (0.055–0.086) | 0.069 (0.055–0.086) | 0.084 (0.068–0.103) | **0.054** (0.042–0.070) |
| S3, 12 vs 13 (500) | per-edge | 0.052 (0.045–0.059) | 0.053 (0.047–0.061) | 0.066 (0.059–0.074) | **0.050** (0.043–0.057) |
| S3, 12 vs 13 | FWER, max-T | 0.064 (0.046–0.089) | 0.044 (0.029–0.066) | 0.068 (0.049–0.094) | 0.052 (0.036–0.075) |
| S3, 25 vs 25 (500) | per-edge | 0.043 (0.037–0.050) | 0.053 (0.047–0.061) | 0.056 (0.050–0.064) | 0.049 (0.042–0.056) |
| S3, 25 vs 25 | FWER, max-T | 0.030 (0.018–0.049) | 0.046 (0.031–0.068) | 0.048 (0.033–0.070) | 0.042 (0.028–0.063) |
| S13 a, 15 vs 15 (200) | group, max-T within film (600 film × study) | 0.050 (0.035–0.071) | 0.068 (0.051–0.091) | 0.087 (0.067–0.112) | **0.053** (0.038–0.074) |
| S13 a | group, any of 3 films (per study) | 0.135 (0.095–0.189) | 0.185 (0.137–0.245) | 0.240 (0.186–0.304) | 0.140 (0.099–0.195) |
| S13 a | interaction, max-T over 8 | 0.075 (0.046–0.120) | 0.050 (0.027–0.090) | 0.085 (0.054–0.132) | 0.070 (0.042–0.114) |

The current-version numbers reproduce the earlier results within Monte Carlo error:
- **S3 at 8 vs 8:** 5.8% / 7.4% before, 6.1% / 7.4% here.
- **S13 a, per film:** 5.7% / 10.3% before. This run (200 studies) gives 5.0% / 8.7%.

The film × study rate counts each study three times, so its interval is somewhat too narrow.

### Diagnostic: variance of the observed statistic vs. permutation variance

The statistic is the signed group difference of delta. For each edge, its variance across null studies is divided by the
mean (over studies) of the variance of its permutation distribution. The table shows the mean of that ratio over the 8
edges, with the range over edges in brackets.

| cell | dDTF current | dDTF renull | GPDC current | GPDC renull |
|---|---|---|---|---|
| 8 vs 8 | 1.10 (0.99–1.17) | 1.02 (0.93–1.12) | **1.21** (1.13–1.27) | **1.01** (0.95–1.06) |
| 12 vs 13 | 1.03 (0.91–1.14) | 1.02 (0.89–1.14) | 1.15 (0.98–1.25) | 1.02 (0.87–1.11) |
| 25 vs 25 | 0.93 (0.84–1.04) | 0.99 (0.92–1.07) | 1.05 (0.97–1.12) | 1.01 (0.92–1.07) |

The current version gives ratios above 1 that shrink with group size and are larger for GPDC. `renull` brings every
cell to about 1.

### Power cost (S4 design B, cg:ROI→child:ROI planted in one group, 60 s, 200 studies)

| cell | gain | measure | dDTF current | dDTF renull | GPDC current | GPDC renull |
|---|---|---|---|---|---|---|
| 12 vs 13 | 0.1 | uncorrected | 0.220 | 0.215 | 0.210 | 0.180 |
| 12 vs 13 | 0.1 | max-T | 0.055 | 0.060 | 0.075 | 0.070 |
| 12 vs 13 | 0.2 | uncorrected | 0.930 | 0.945 | 0.910 | 0.875 |
| 12 vs 13 | 0.2 | max-T | 0.690 | 0.700 | 0.705 | 0.660 |
| 25 vs 25 | 0.1 | uncorrected | 0.350 | 0.360 | 0.330 | 0.310 |
| 25 vs 25 | 0.1 | max-T | 0.100 | 0.120 | 0.115 | 0.090 |
| 25 vs 25 | 0.2 | uncorrected | 0.995 | 1.000 | 1.000 | 0.995 |
| 25 vs 25 | 0.2 | max-T | 0.980 | 0.990 | 0.980 | 0.985 |

Wilson intervals are in `checks_2026-11/A2/summary_power.csv`. Every difference is within about ±4 points, well inside
the ±6–7 point intervals.

Figure: `checks_2026-11/A2/fig_A2_renull.png` / `.svg`. Panels A and B show rate vs. dyads per group: current solid,
renull dashed, dDTF filled, GPDC open, 3–7% band. Panel C shows S13 a.

### Verdict A2

**The hypothesis is confirmed.** Fixed group null medians make the permutation distribution too narrow:
- On exchangeable data the current test runs at 7.25%; renull runs at 5.3% and is uniform.
- In S3, the observed-to-permutation variance ratio is above 1 for the current test (up to 1.21 for GPDC at 8 vs 8).
  It shrinks with group size, as a median-noise effect should, and is about 1 with renull.

**renull restores nominal rates where the current test was liberal:**
- GPDC per-edge: 7.4 / 6.6 / 5.6% → 5.0 / 5.0 / 4.9%.
- GPDC max-T FWER at 8 vs 8: 8.4% → 5.4%.
- S13 GPDC group contrast within film: 8.7% → 5.3%.
- dDTF per-edge at 8 vs 8: 6.1% → 5.3%.

**Two cells stay at or above the band edge with renull, both dDTF:**
- S3 max-T FWER at 8 vs 8: 6.9% with both versions. The median mechanism has no effect there, so the residual comes from
  something else. A plausible candidate is the standardization of max-T with only 8 dyads per group, which this check
  did not test.
- S13 group contrast within film: 5.0% → 6.8%. The intervals overlap, so this is not a clear increase.

**The power cost is negligible:** no systematic loss for dDTF, and at most −3.5 points for GPDC (within Monte Carlo
error).

**Recommendation:** use `renull=True` for group contrasts and the interaction with the within-group null. It is not yet
the default; switching it on is a decision for the manuscript.

---

## A4: omitted-driver edge with Z in the model

Generator: S05 omitted driver, 5-node model, N = 25, 100 studies per cell, with the same seeds for every variant of a
cell. The table gives the study-level rate (pairing permutation test) of cg heart → child heart, which has no true edge.

| variant | dDTF 60 s, g 0.4 | dDTF 300 s, g 0.4 | dDTF 60 s, g 0.6 | dDTF 300 s, g 0.6 | GPDC 60 s, g 0.4 | GPDC 300 s, g 0.4 | GPDC 60 s, g 0.6 | GPDC 300 s, g 0.6 |
|---|---|---|---|---|---|---|---|---|
| C0 default | 0.03 (0.01–0.08) | **0.64** (0.54–0.73) | 0.55 (0.45–0.64) | **1.00** (0.96–1.00) | 0.12 (0.07–0.20) | **0.88** (0.80–0.93) | 0.68 (0.58–0.76) | **1.00** (0.96–1.00) |
| C1 constant detrend | 0.02 (0.01–0.07) | 0.05 (0.02–0.11) | 0.27 (0.19–0.36) | 0.64 (0.54–0.73) | 0.03 (0.01–0.08) | 0.11 (0.06–0.19) | 0.16 (0.10–0.24) | 0.64 (0.54–0.73) |
| C2 no detrending | 0.01 (0.00–0.05) | **0.01** (0.00–0.05) | 0.20 (0.13–0.29) | **0.17** (0.11–0.26) | 0.05 (0.02–0.11) | **0.07** (0.03–0.14) | 0.13 (0.08–0.21) | **0.11** (0.06–0.19) |
| C3a one window, linear | 0.02 (0.01–0.07) | 0.02 (0.01–0.07) | 0.25 (0.18–0.34) | 0.18 (0.12–0.27) | 0.04 (0.02–0.10) | 0.06 (0.03–0.13) | 0.20 (0.13–0.29) | 0.15 (0.09–0.23) |
| C3b one window, none | 0.02 (0.01–0.07) | 0.02 (0.01–0.07) | 0.24 (0.17–0.33) | 0.16 (0.10–0.24) | 0.03 (0.01–0.08) | 0.05 (0.02–0.11) | 0.17 (0.11–0.26) | 0.14 (0.09–0.22) |
| C4 p = 8 | – | 0.21 (0.14–0.30) | – | 0.74 (0.65–0.82) | – | 0.72 (0.63–0.80) | – | 0.99 (0.95–1.00) |
| C4 p = 12 | – | 0.15 (0.09–0.23) | – | 0.46 (0.37–0.56) | – | 0.49 (0.39–0.59) | – | 0.93 (0.86–0.97) |
| C5 Z → both hearts at lag 1 | 0.08 (0.04–0.15) | 0.57 (0.47–0.66) | 0.52 (0.42–0.62) | 1.00 (0.96–1.00) | 0.11 (0.06–0.19) | 0.72 (0.63–0.80) | 0.58 (0.48–0.67) | 1.00 (0.96–1.00) |
| C6 Z pole radius 0.5 | 0.05 (0.02–0.11) | 0.11 (0.06–0.19) | 0.11 (0.06–0.19) | 0.44 (0.35–0.54) | 0.00 (0.00–0.04) | 0.09 (0.05–0.16) | 0.13 (0.08–0.21) | 0.29 (0.21–0.39) |

C0 reproduces the S5 sweep: 6.5% / 10.5% at 60 s and 71% / 86% at 300 s (gain 0.4), and 52.5% / 64% → 100% (gain 0.6).
Rates of the reverse edge are in `checks_2026-11/A4/summary_rates.csv`.

Figures:
- `checks_2026-11/A4/fig_A4_rates.png` / `.svg`: rate vs. segment length per variant.
- `fig_A4_granger.png` / `.svg`: median Granger F vs. length.

### Real vs. pseudo vs. independent dyads

The table gives the mean edge value of cg heart → child heart on the analysis scale (Box-Cox 0.25, band-averaged).
Independent dyads have no heart–heart link and an independent Z per person, with the caregiver's Z in the model.

| estimator | variant | 300 s, g 0.4: real / pseudo / independent | 300 s, g 0.6: real / pseudo / independent |
|---|---|---|---|
| dDTF | C0 | −3.816 / −3.840 / −3.840 | −3.779 / −3.844 / −3.843 |
| dDTF | C2 | −3.841 / −3.839 / −3.839 | −3.834 / −3.842 / −3.841 |
| GPDC | C0 | −1.851 / −1.983 / −1.987 | −1.716 / −2.027 / −2.022 |
| GPDC | C2 | −1.964 / −1.979 / −1.980 | −1.999 / −2.025 / −2.023 |

Pseudo-dyads and independent dyads agree in every cell, so the null construction itself is sound. The excess sits in the
real dyads. Under C0 it is large and grows with gain. Under C2 it is small: 0.00 / 0.015 at gain 0.4 and 0.008 / 0.026 (dDTF / GPDC)
at gain 0.6. The values for 60 s are in `summary_rates.csv`.

### Time-domain check: conditional Granger F, cg heart → child heart, 5-node least-squares fit of each real dyad

Each entry is the median over studies of the study median over 25 dyads.

| variant | gain | 60 s | 120 s | 300 s | 600 s | F × T at 600 s / 60 s |
|---|---|---|---|---|---|---|
| C0 default | 0.4 | 0.0321 | 0.0157 | 0.0071 | 0.0045 | 1.4 |
| C0 default | 0.6 | 0.0372 | 0.0206 | 0.0111 | **0.0086** | 2.3 |
| C2 no detrending | 0.4 | 0.0306 | 0.0138 | 0.0050 | 0.0024 | 0.78 |
| C2 no detrending | 0.6 | 0.0309 | 0.0139 | 0.0051 | 0.0025 | 0.81 |
| C3b one window, none | 0.6 | 0.0277 | 0.0124 | 0.0046 | 0.0023 | 0.83 |

Under C2 and C3, F falls as about 1/T and does not depend on gain: pure small-sample bias toward zero. Under C0, F × T
grows with length and gain, so C0 has a plateau above the small-sample level: about 0.002 at gain 0.4 and 0.006 at gain
0.6 at 600 s. That plateau is a misspecification. C1 (constant detrending) lies in between, at 0.0035 at 600 s with
gain 0.6.

### Oracle

For dDTF and GPDC of cg heart → child heart, computed from the true 5-node coefficients (identity noise covariance, band
0.15–0.5 Hz), the maximum |value| over 1000 parameter draws per gain is exactly 0. This holds for the C0, C5 and C6
generators. The population conditional GC from the exact autocovariances is ≤ 2e-13. The edge is truly absent.

### Verdict A4

**Per-window linear detrending causes the residual coupling.** Detrending 25-sample (10 s) windows of the slow heart and
Z processes (pole radius 0.80–0.92) changes their lagged cross-covariances. The stacked-window regression then no longer
satisfies the true VAR. In real dyads this leaves a constant bias on the heart → heart coefficient. Pseudo-dyads do not
share it, because there Z does not drive the child heart. A constant bias against shrinking noise is why the rate grows
with recording length.

**Removing the detrending removes the growth:**
- At gain 0.4, C2, C3a and C3b give 1–7% at both 60 and 300 s (dDTF and GPDC).
- C1 (constant detrending) only partly helps: 64% at 300 s with gain 0.6.
- A longer model order (C4) or a single lag (C5) does not help.
- A less persistent Z (C6) reduces the bias but does not remove it.

**A smaller residual remains at gain 0.6 with every variant:** 11–25%, the same at 60 and 300 s. It does not grow, and
its Granger F decays as 1/T. So it is not misspecification: it is a finite-sample effect specific to real dyads, where
Z and the caregiver heart are strongly collinear. Pseudo-dyads and independent dyads do not show it. No variant here
removes it, and per the handoff it was not pursued further.

**Should the default pipeline change?** Yes, for the estimation step: fit the 10 s windows without per-window linear
detrending (C2, `detrend_type="none"`). The windowed averaging stays; one whole-segment fit (C3) is equivalent here.
Two caveats remain open:
- The global z-scoring then becomes the only drift handling, and real IBI drift may need a separate high-pass step
  before windowing.
- The change affects every simulation that uses the default estimator.

At minimum S1, S3 and S8 need a rerun; S5 does too, for the omitted-driver numbers. They were not run here.

---

## Five-line summary

1. A2 is confirmed: with fixed group null medians the permutation variance is too small (observed/permutation variance
   up to 1.21 for GPDC at 8 vs 8; 7.25% on exchangeable synthetic data).
2. `renull=True` restores about 5% in S3 (GPDC 7.4 → 5.0% per edge; max-T 8.4 → 5.4%) and in S13 GPDC (8.7 → 5.3%), at
   no measurable power cost. One dDTF cell (max-T at 8 vs 8, 6.9%) is unaffected and has another cause.
3. A4's oracle is exactly 0: the cg heart → child heart edge is truly absent, and pseudo-dyads match independent dyads.
4. The growth with length comes from per-window linear detrending of 10 s windows (a constant bias; Granger F plateaus).
   Without detrending, or with one window, the rate is 1–7% at gain 0.4 for any length.
5. Recommendation: drop per-window linear detrending (`detrend_type="none"`) and rerun S1, S3, S5 and S8. A
   non-growing, real-dyad-only 11–25% residual at gain 0.6 remains (likely the collinearity of Z and the caregiver heart; not
   tested).

---

## Rerun of S1, S3, S5 and S8 with the new default (no per-window detrending)

Following the A4 verdict, which was approved, the default of `fit_mvar`, `edge_matrix`, `regressor_diagnostics` and
`conditional_gc` in `src/sim_estimation.py` is now `detrend_type="none"`: the 10 s windows (50% overlap) are fitted as
cut. The old behaviour is still available as `detrend_type="linear"`. In S8 the baseline changed accordingly, and the old
default was added as the last variant, `detrend=linear`. It comes last so that every other variant draws the same
permutations as before.

The two implementation checks against `src.connectivity.Granger_estimator`, which detrends linearly, now pass `"linear"`
explicitly. These are S01's `implementation_check` and `tests/test_sim_estimation.py`.

S1, S3, S5 and S8 were rerun in full mode for both estimators with the full-run seeds, so they analyse the same
simulated studies. Old results are kept in `Results/<est>/S_NN_linear_detrend/` (local, gitignored); new results are in
`Results/<est>/S_NN/`. Comparison tables with all cells: `checks_2026-11/nodetrend/compare_S0{1,3,5,8}.csv`, built by
`dDTF/CHK_nodetrend_compare.py`. All rates come with 95% Wilson intervals. Rates pooled over edges use
n = studies × edges, so their intervals are somewhat too narrow.

A consistency check confirms the rerun used identical data: the new S8 variant `detrend=linear` reproduces the old
baseline (null FPR dDTF 0.057 vs. 0.059, GPDC 0.050 vs. 0.052). The variants that never detrended ("window=whole
segment") or detrend constantly are identical old vs. new.

### S1 recovery (25 dyads, 60 s, 200 studies)

- **Planted edges:** detected in 100% of studies in both versions and for both estimators, uncorrected and after max-T
  (0.981–1.000).
- **Absent cross-person edges:** 0–2% (old) and 0–1% (new).
- **Reverse edges:** for dDTF, the reverse edges of the planted edges are detected in 100% of studies in both versions.
  This is the known dDTF reverse-edge pattern, and it is unchanged.
- **Median z of the planted edges:**
  - dDTF: 3.40 / 4.18 / 3.79 → 3.65 / 4.16 / 3.95.
  - GPDC: 2.56 / 2.63 / 2.49 → 2.67 / 2.57 / 2.60.

### S3 null calibration (200 studies per cell; 1000 for N = 10 and N = 16)

Per-edge rates (mean over 8 edges, per cell and null mode) are unchanged.

| estimator | level | old range over cells × modes | new range |
|---|---|---|---|
| dDTF | study | 0.037–0.060 | 0.042–0.058 |
| dDTF | group | 0.038–0.075 | 0.040–0.074 |
| GPDC | study | 0.043–0.055 | 0.039–0.054 |
| GPDC | group | 0.040–0.082 | 0.043–0.083 |

The within-group group contrast at N = 10 stays at 7–8% in both versions: this is the A2 effect, not affected by
detrending. |median z| ≤ 0.016 everywhere.

The single edge × cell extremes widened slightly, from 0.020–0.095 to 0.010–0.125. The new maximum (0.125, 95% CI
0.085–0.178, at N = 25, 120 s, cg:ROI→child:ROI, exclude null) is one of 288 edge × cell rates from 200 studies. The
same cell was the old maximum region (0.090).

Max-T family-wise rates, selected cells:

| cell | estimator | level, null | old | new |
|---|---|---|---|---|
| N10, 60 s | dDTF | study, pooled | 0.063 (0.050–0.080) | 0.059 (0.046–0.075) |
| N16, 60 s | dDTF | study, pooled | 0.061 (0.048–0.078) | 0.063 (0.050–0.080) |
| N25, 60 s | dDTF | study, pooled | 0.035 (0.017–0.070) | 0.045 (0.024–0.083) |
| N50, 60 s | dDTF | study, pooled | 0.080 (0.050–0.126) | 0.075 (0.046–0.120) |
| N10, 300 s | dDTF | group, within_group | 0.100 (0.083–0.120) | 0.124 (0.105–0.146) |
| N16, 60 s | dDTF | group, within_group | 0.071 (0.057–0.089) | 0.062 (0.049–0.079) |
| N10, 60 s | GPDC | study, pooled | 0.061 (0.048–0.078) | 0.053 (0.041–0.069) |
| N16, 60 s | GPDC | study, pooled | 0.059 (0.046–0.075) | 0.053 (0.041–0.069) |
| N10, 300 s | GPDC | group, within_group | 0.117 (0.099–0.138) | 0.131 (0.111–0.153) |
| N16, 60 s | GPDC | group, within_group | 0.081 (0.066–0.100) | 0.102 (0.085–0.122) |

Study-level max-T is within 2.5–8.5% in every cell in both versions. The GPDC criterion "max-T within band in every cell
(study level)" now passes; it failed before.

The within-group group contrast at N = 10–16 is still above the band (10–13%) in both versions, and slightly higher in
the new run. This is the problem A2 addresses with `renull` (not used in these scripts). All other S3 criteria have the
same outcome as before.

### S5 omitted driver (sweep: 200 studies per cell, N = 25)

Rate of the absent edge cg:HRV → child:HRV in the 5-node model with Z:

| gain | segment, p | dDTF old | dDTF new | GPDC old | GPDC new |
|---|---|---|---|---|---|
| 0.4 | 60 s, p 4 | 0.065 (0.038–0.108) | 0.030 (0.014–0.064) | 0.105 (0.070–0.155) | 0.075 (0.046–0.120) |
| 0.4 | 120 s, p 4 | 0.180 (0.133–0.239) | **0.040** (0.020–0.077) | 0.360 (0.297–0.429) | **0.095** (0.062–0.144) |
| 0.4 | 300 s, p 4 | 0.710 (0.644–0.768) | **0.010** (0.003–0.036) | 0.860 (0.805–0.901) | **0.085** (0.054–0.132) |
| 0.4 | 60 s, p 2 / p 6 | 0.135 / 0.015 | 0.030 / 0.015 | 0.120 / 0.185 | 0.080 / 0.070 |
| 0.6 | 60 s, p 4 | 0.525 (0.456–0.593) | 0.165 (0.120–0.223) | 0.640 (0.571–0.703) | 0.140 (0.099–0.195) |
| 0.6 | 120 s, p 4 | 0.825 (0.766–0.871) | 0.175 (0.129–0.234) | 0.950 (0.910–0.973) | 0.140 (0.099–0.195) |
| 0.6 | 300 s, p 4 | 1.000 (0.981–1.000) | 0.150 (0.107–0.206) | 1.000 (0.981–1.000) | 0.160 (0.116–0.217) |

The growth with length is gone. At gain 0.4 the rate is 1–9.5% at every length. At gain 0.6 a flat residual of 14–17%
remains, as predicted by A4. The 4-node model without Z still detects the dyad-level edge in 100% of studies, as
intended.

Main run (60 s, gain 0.4):
- **Cascade:** absent indirect edge, 4-node estimator: dDTF 1.00 → 1.00 (the known dDTF cascade limitation, unchanged);
  GPDC 0.005 → 0.010. Planted power is 1.00 everywhere.
- **Common driver:** absent edge, 4-node estimator: dDTF 0.020 → 0.005, GPDC 0.085 → 0.065.
- **Omitted driver, 5-node with Z:** dDTF 0.065 → 0.040, GPDC 0.135 → 0.075.
- **Criteria:** dDTF 5 of 7 met in both versions, the same two failing (both dDTF cascade criteria). GPDC goes from 6 of
  7 to 7 of 7: the omitted-driver criterion (5-node with Z ≤ 0.10) now passes.

### S8 sensitivity (25 dyads, 60 s, 200 null + 200 planted studies)

| estimator | variant | null FPR old | null FPR new | power old → new |
|---|---|---|---|---|
| dDTF | baseline | 0.059 (0.047–0.074) | 0.051 (0.040–0.065) | 1.00 → 1.00 |
| dDTF | detrend=linear (old default) | – | 0.057 (0.045–0.071) | – → 1.00 |
| dDTF | range over all variants | 0.038–0.059 | 0.038–0.057 | 1.00 everywhere |
| GPDC | baseline | 0.052 (0.041–0.066) | 0.058 (0.046–0.072) | 1.00 → 1.00 |
| GPDC | detrend=linear (old default) | – | 0.050 (0.039–0.064) | – → 1.00 |
| GPDC | range over all variants | 0.043–0.057 | 0.038–0.060 | 1.00 everywhere |

All S8 criteria are met in both versions (3 of 3).

### Verdict of the rerun

Calibration and power are unchanged:
- **S1:** recovery is 100%, absent edges ≤ 2%, and median z is slightly higher.
- **S3:** per-edge and study-level family-wise rates are within the same ranges as before. The one S3 issue that
  remains, the within-group group contrast at N = 10–16, is the A2 problem and is independent of detrending.
- **S8:** every variant keeps its null FPR at 3.8–6.0% with full power.

The only material change is the intended one in S5: with Z in the model, the spurious heart → heart edge no longer grows
with recording length.

**Not rerun:** S2, S4, S4b, S6, S6b, S7, S9–S13, the Drive copy of the results and the manuscript figures. They still
reflect per-window linear detrending. `fig2_model_and_null.py` also still annotates the windows as "each linearly
detrended".
