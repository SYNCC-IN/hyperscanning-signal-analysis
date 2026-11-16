# Pipeline alignment with the manuscript (v0.7): report, 2026-11-16

Branch `pipeline_manuscript_alignment` (from `multimodal_MVAR` at eae74b7). The real-data pipeline now estimates and
tests on the simulations' code path (`src/sim_estimation.py`, `src/sim_inference.py`, `src/perm_anova.py`); stage
scripts are configuration, orchestration and reporting.

## Decisions taken before the run (user, 2026-11-16)

- Primary estimator for edge-wise claims: **GPDC**; dDTF as companion ("is there coupling", forward–reverse contrast).
- Max-T family: **all 8 cross-person edges of one film** (`FAMILY_EDGES: null`).
- Planned film contrasts: Incredibles vs (Peppa+Brave)/2 **plus the pairwise contrasts** Incredibles−Peppa,
  Incredibles−Brave, Peppa−Brave (one max-T family over 4 contrasts × 8 edges).
- Earlier stage-06 outputs (BH-FDR primary family, asymmetry track): **archived** to `06_group_archive_<date>/` on the
  first run of the new stage 06.
- Respiration: not recorded (the loaders read SVAROG EEG/ECG, Polar H10 IBI and Pupil Labs eye tracking only; no
  respiration channel in `src/`). The stage-07 gate says so next to the shared-driver caveat.

## Old vs new settings

| setting | before | now (manuscript) |
|---|---|---|
| MVAR fit | Yule–Walker on window-averaged ACF (`mtmvar.ar_coeff`) | least squares on stacked windows (`fit_mvar(method="ols")`) |
| per-window detrending | linear | none (`DETREND_TYPE="none"`) |
| order | stage 4: per-case criterion order; stage 5: p = 4 | p = 4 everywhere (criterion order descriptive only) |
| window | 10 s, 50 % | 10 s, 50 % (unchanged) |
| coupling band / Box–Cox / fs | 0.15–0.5 Hz / 0.25 / 2.5 Hz | unchanged |
| design band-pass | 0.05–1.0 Hz, Butterworth 2, zero phase | unchanged (verified in stage 02) |
| estimator | dDTF only | GPDC and dDTF from one fit, reported dDTF / GPDC |
| stability check | YW fit (almost never triggered); unstable real dyads kept | same OLS fit as the estimate; unstable real dyad dropped with its row and column; unstable pseudo-dyad = empty cell |
| stage-5 output | Δ, z against pooled "film" null (reference) + within-group sensitivity | full pairing table T per film × estimator (`.nc`), Δ / z / dyad p for pooled, exclude and within-group nulls; within-group is the default |
| primary inference | Bayesian per-edge models + BH-FDR on {H2, H4} | permutation tests with max-T (new stage 07) |
| asymmetry | Bayesian model of z difference | sign-flip test of mean d = Δfwd − Δrev |
| film contrast | Bayesian cell means | re-pairing test B; sign flip of Δ as sensitivity |
| covariates | Bayesian toggles | exploratory only (hybrid renull + SS_hyp, `ADD_AGE_COVARIATE`) |
| Bayesian model | primary | optional effect sizes: pooled film×group + (film×group | edge) + (1 | dyad) |

## Validation

1. **pytest.** `tests/test_pipeline_alignment.py` (10 tests): the pipeline's band-averaged edge matrix (stage-4 path and
   `pair_tables`, real and pseudo-dyads) equals `sim_estimation.edge_matrix` to 1e-12 for GPDC and dDTF; the legacy
   YW/linear defaults of `Granger_estimator` still equal the old `mtmvar` path (so S01 is unchanged); helper layouts.
   Full suite: 164 passed, 12 skipped, 9 failed — the same 9 failures as on eae74b7 (`test_dataloader`,
   `test_export`, `test_secore_loader`), unrelated to this work.
2. **Synthetic end-to-end** (`stage00c_end_to_end.py`: simulated studies written as stage-02 files and run through the
   unchanged stage 05 and 07; 25 dyads, 60 s, groups of 12 / 13, three films; 20 studies with the S1 planted edges,
   50 without coupling). Per-film tests: one replicate per film (n = 60 / 150); cross-film tests: one per study.
   - Planted (S1): test 1 detects the three planted edges in 100 % of films with both estimators. GPDC: the reverse
     and absent edges 0–1.7 % (max-T). dDTF: the reverse edges of the planted pairs are detected in 98–100 %, as in
     S1 (dDTF reverse edges inherit the forward coupling). Same pattern as `Results/<est>/S_01` (table in
     `00c_end_to_end/end_to_end_comparison.csv`; the simulation used the pooled null with one group).
   - No coupling (S3): family-wise rates (max-T) — test 1: GPDC 6.0 % [3.2, 11.0], dDTF 4.7 % [2.3, 9.3]
     (S_03, within-group, 25 dyads, 60 s: 4.0 % / 2.5 %); test 4 per film: 4.7 % / 2.7 % (S_03: 3.5 % / 3.5 %);
     test 3: GPDC 5.3 %, dDTF 0.7 % (conservative); tests 4b/4c, 5, 6: 0–6 % (n = 50 each); 5b (sign-flip
     sensitivity): 6 % / 8 %. Per-edge unadjusted rates: mean 3.8–7.6 % per test, range 0–14 % (n = 50–150 per cell).
     All family-wise Wilson intervals contain 5 % except dDTF test 3 (below).
3. **Real data.** Stages 03–05 and 07 run on the 42 dyads of `02_envelopes` (stage-02 outputs reused: settings
   verified, not re-run). Stage 06: see below.

### Stable fits (p = 4, OLS)

| film | dyads | stable real fits (TD / ASD) | real dyads dropped | unstable pseudo-dyads |
|---|---|---|---|---|
| Peppa | 42 | 42 (24 / 18) | none | 0 / 1722 |
| Incredibles | 42 | 41 (24 / 17) | W_116 | 9 / 1722 |
| Brave | 42 | 41 (24 / 17) | W_116 | 41 / 1722 |

Cross-film tests use 42 dyads (W_116 contributes to Peppa only); film-contrast tests and the factorial block use the
41 dyads present in all films.

### Main results (stage 07; N_PERM = 1000, alpha = 0.05, seed 20261116)

**No test is significant after max-T correction, with either estimator** (smallest max-T adjusted p: dDTF 0.205,
test 1, Brave, cg:HRV→child:HRV; GPDC 0.258, test 6). Unadjusted p < 0.05 occurred in 6 of the 412 rows (both
estimators): dDTF test 1 Brave cg:HRV→child:HRV (p = 0.034) and test 5 Peppa−Brave cg:HRV→child:HRV (0.035); GPDC
test 6 child:ROI→cg:HRV in three contrasts (0.017–0.045); and the GPDC factorial term G×source (p = 0.007, one of
6 terms, each reported without correction). These are reported, not interpreted. Full rows: `07_permutation/<estimator>/stage07_tests.csv`; gate:
`07_permutation/permutation_gate.html`.

#### Test 1: edge existence (mean T of real dyads; within-group re-pairing; max-T over the 8 cross-person edges of a film)

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| Peppa | child:ROI->cg:ROI | -3.730 | 0.349 | 0.926 | -1.482 | 0.628 | 0.993 |
| Peppa | child:ROI->cg:HRV | -3.819 | 0.530 | 0.982 | -1.420 | 0.675 | 0.993 |
| Peppa | cg:ROI->child:ROI | -3.735 | 0.703 | 0.982 | -1.476 | 0.660 | 0.993 |
| Peppa | cg:ROI->child:HRV | -3.770 | 0.581 | 0.982 | -1.438 | 0.368 | 0.945 |
| Peppa | child:HRV->cg:ROI | -3.599 | 0.881 | 0.982 | -1.455 | 0.888 | 0.993 |
| Peppa | child:HRV->cg:HRV | -3.699 | 0.539 | 0.982 | -1.296 | 0.358 | 0.945 |
| Peppa | cg:HRV->child:ROI | -3.423 | 0.230 | 0.839 | -1.327 | 0.153 | 0.740 |
| Peppa | cg:HRV->child:HRV | -3.530 | 0.813 | 0.982 | -1.412 | 0.801 | 0.993 |
| Incredibles | child:ROI->cg:ROI | -3.726 | 0.648 | 0.956 | -1.466 | 0.603 | 0.973 |
| Incredibles | child:ROI->cg:HRV | -3.817 | 0.611 | 0.956 | -1.480 | 0.895 | 0.973 |
| Incredibles | cg:ROI->child:ROI | -3.714 | 0.347 | 0.872 | -1.412 | 0.254 | 0.904 |
| Incredibles | cg:ROI->child:HRV | -3.771 | 0.703 | 0.956 | -1.453 | 0.491 | 0.973 |
| Incredibles | child:HRV->cg:ROI | -3.599 | 0.779 | 0.956 | -1.421 | 0.717 | 0.973 |
| Incredibles | child:HRV->cg:HRV | -3.691 | 0.129 | 0.579 | -1.313 | 0.419 | 0.965 |
| Incredibles | cg:HRV->child:ROI | -3.445 | 0.495 | 0.932 | -1.390 | 0.494 | 0.973 |
| Incredibles | cg:HRV->child:HRV | -3.492 | 0.162 | 0.657 | -1.354 | 0.458 | 0.973 |
| Brave | child:ROI->cg:ROI | -3.723 | 0.529 | 0.977 | -1.407 | 0.270 | 0.883 |
| Brave | child:ROI->cg:HRV | -3.819 | 0.653 | 0.988 | -1.469 | 0.826 | 1.000 |
| Brave | cg:ROI->child:ROI | -3.736 | 0.929 | 0.998 | -1.515 | 0.930 | 1.000 |
| Brave | cg:ROI->child:HRV | -3.787 | 0.965 | 0.998 | -1.512 | 0.811 | 1.000 |
| Brave | child:HRV->cg:ROI | -3.609 | 0.944 | 0.998 | -1.427 | 0.922 | 1.000 |
| Brave | child:HRV->cg:HRV | -3.701 | 0.253 | 0.855 | -1.319 | 0.375 | 0.946 |
| Brave | cg:HRV->child:ROI | -3.450 | 0.699 | 0.988 | -1.418 | 0.810 | 1.000 |
| Brave | cg:HRV->child:HRV | -3.469 | 0.034 | 0.205 | -1.330 | 0.243 | 0.876 |

#### Test 3: dominant direction (mean d = Δ(cg→child) − Δ(child→cg); sign flip; max-T over the 4 pairs)

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| Peppa | cg:ROI->child:ROI vs child:ROI->cg:ROI | -0.0162 | 0.301 | 0.694 | -0.0053 | 0.935 | 0.935 |
| Peppa | cg:ROI->child:HRV vs child:HRV->cg:ROI | +0.0169 | 0.493 | 0.732 | +0.0578 | 0.386 | 0.772 |
| Peppa | cg:HRV->child:ROI vs child:ROI->cg:HRV | +0.0033 | 0.919 | 0.919 | +0.0818 | 0.211 | 0.640 |
| Peppa | cg:HRV->child:HRV vs child:HRV->cg:HRV | -0.0368 | 0.280 | 0.686 | -0.0460 | 0.445 | 0.772 |
| Incredibles | cg:ROI->child:ROI vs child:ROI->cg:ROI | +0.0108 | 0.494 | 0.934 | +0.0570 | 0.418 | 0.790 |
| Incredibles | cg:ROI->child:HRV vs child:HRV->cg:ROI | +0.0105 | 0.665 | 0.953 | +0.0134 | 0.836 | 0.956 |
| Incredibles | cg:HRV->child:ROI vs child:ROI->cg:HRV | -0.0052 | 0.838 | 0.971 | +0.0817 | 0.266 | 0.688 |
| Incredibles | cg:HRV->child:HRV vs child:HRV->cg:HRV | -0.0031 | 0.925 | 0.971 | -0.0166 | 0.797 | 0.956 |
| Brave | cg:ROI->child:ROI vs child:ROI->cg:ROI | -0.0172 | 0.284 | 0.606 | -0.1040 | 0.091 | 0.324 |
| Brave | cg:ROI->child:HRV vs child:HRV->cg:ROI | +0.0088 | 0.709 | 0.927 | +0.0165 | 0.833 | 0.976 |
| Brave | cg:HRV->child:ROI vs child:ROI->cg:HRV | -0.0099 | 0.733 | 0.927 | +0.0287 | 0.725 | 0.976 |
| Brave | cg:HRV->child:HRV vs child:HRV->cg:HRV | +0.0456 | 0.151 | 0.459 | +0.0204 | 0.759 | 0.976 |

#### Test 4: group contrast ASD − TD of within-group Δ (re-nulled; max-T within film)

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| Peppa | child:ROI->cg:ROI | -0.0129 | 0.692 | 0.993 | -0.0747 | 0.528 | 0.908 |
| Peppa | child:ROI->cg:HRV | -0.0410 | 0.067 | 0.394 | -0.2080 | 0.052 | 0.340 |
| Peppa | cg:ROI->child:ROI | +0.0050 | 0.866 | 0.993 | +0.0383 | 0.728 | 0.927 |
| Peppa | cg:ROI->child:HRV | -0.0085 | 0.761 | 0.993 | -0.0897 | 0.456 | 0.908 |
| Peppa | child:HRV->cg:ROI | +0.0373 | 0.376 | 0.933 | +0.0721 | 0.395 | 0.896 |
| Peppa | child:HRV->cg:HRV | +0.0513 | 0.091 | 0.451 | +0.1346 | 0.177 | 0.749 |
| Peppa | cg:HRV->child:ROI | -0.0027 | 0.943 | 0.993 | +0.0900 | 0.249 | 0.811 |
| Peppa | cg:HRV->child:HRV | -0.0189 | 0.655 | 0.993 | +0.0183 | 0.839 | 0.927 |
| Incredibles | child:ROI->cg:ROI | -0.0227 | 0.517 | 0.966 | -0.0988 | 0.371 | 0.963 |
| Incredibles | child:ROI->cg:HRV | +0.0030 | 0.910 | 0.982 | +0.1567 | 0.224 | 0.863 |
| Incredibles | cg:ROI->child:ROI | -0.0108 | 0.729 | 0.980 | -0.0339 | 0.772 | 0.993 |
| Incredibles | cg:ROI->child:HRV | -0.0161 | 0.575 | 0.971 | -0.0898 | 0.366 | 0.963 |
| Incredibles | child:HRV->cg:ROI | -0.0626 | 0.152 | 0.615 | -0.0381 | 0.680 | 0.993 |
| Incredibles | child:HRV->cg:HRV | -0.0261 | 0.448 | 0.960 | -0.0249 | 0.825 | 0.993 |
| Incredibles | cg:HRV->child:ROI | -0.0676 | 0.175 | 0.652 | -0.0619 | 0.475 | 0.963 |
| Incredibles | cg:HRV->child:HRV | -0.0081 | 0.871 | 0.982 | -0.0421 | 0.717 | 0.993 |
| Brave | child:ROI->cg:ROI | -0.0082 | 0.801 | 0.948 | -0.0087 | 0.922 | 0.969 |
| Brave | child:ROI->cg:HRV | -0.0152 | 0.496 | 0.924 | -0.1698 | 0.139 | 0.664 |
| Brave | cg:ROI->child:ROI | -0.0530 | 0.073 | 0.362 | -0.1548 | 0.123 | 0.664 |
| Brave | cg:ROI->child:HRV | -0.0219 | 0.328 | 0.899 | -0.1050 | 0.350 | 0.821 |
| Brave | child:HRV->cg:ROI | -0.0045 | 0.926 | 0.948 | -0.0230 | 0.830 | 0.969 |
| Brave | child:HRV->cg:HRV | +0.0255 | 0.445 | 0.924 | +0.1167 | 0.172 | 0.677 |
| Brave | cg:HRV->child:ROI | +0.0685 | 0.219 | 0.765 | +0.1059 | 0.235 | 0.714 |
| Brave | cg:HRV->child:HRV | +0.0368 | 0.549 | 0.924 | +0.0810 | 0.516 | 0.881 |

#### Test 4c: group contrast of Δ averaged over films

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| mean | child:ROI->cg:ROI | -0.0099 | 0.627 | 0.972 | -0.0580 | 0.440 | 0.894 |
| mean | child:ROI->cg:HRV | -0.0158 | 0.300 | 0.918 | -0.0815 | 0.234 | 0.801 |
| mean | cg:ROI->child:ROI | -0.0154 | 0.401 | 0.961 | -0.0521 | 0.406 | 0.894 |
| mean | cg:ROI->child:HRV | -0.0111 | 0.479 | 0.972 | -0.0953 | 0.138 | 0.640 |
| mean | child:HRV->cg:ROI | -0.0173 | 0.469 | 0.972 | +0.0134 | 0.793 | 0.927 |
| mean | child:HRV->cg:HRV | +0.0112 | 0.489 | 0.972 | +0.0944 | 0.120 | 0.640 |
| mean | cg:HRV->child:ROI | +0.0095 | 0.759 | 0.972 | +0.0451 | 0.384 | 0.894 |
| mean | cg:HRV->child:HRV | +0.0124 | 0.659 | 0.972 | +0.0203 | 0.720 | 0.927 |

#### Test 5: film contrasts, re-pairing (test B; max-T over 4 contrasts × 8 edges)

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| Incredibles_vs_rest | child:ROI->cg:ROI | -0.0062 | 0.636 | 1.000 | -0.0306 | 0.719 | 1.000 |
| Incredibles_vs_rest | child:ROI->cg:HRV | +0.0021 | 0.862 | 1.000 | -0.0052 | 0.999 | 1.000 |
| Incredibles_vs_rest | cg:ROI->child:ROI | +0.0195 | 0.243 | 0.991 | +0.0894 | 0.169 | 0.970 |
| Incredibles_vs_rest | cg:ROI->child:HRV | +0.0036 | 0.709 | 1.000 | +0.0243 | 0.800 | 1.000 |
| Incredibles_vs_rest | child:HRV->cg:ROI | +0.0036 | 0.620 | 1.000 | +0.0097 | 0.635 | 1.000 |
| Incredibles_vs_rest | child:HRV->cg:HRV | +0.0062 | 0.562 | 1.000 | -0.0284 | 0.786 | 1.000 |
| Incredibles_vs_rest | cg:HRV->child:ROI | -0.0231 | 0.643 | 1.000 | -0.0309 | 0.794 | 1.000 |
| Incredibles_vs_rest | cg:HRV->child:HRV | -0.0033 | 0.856 | 1.000 | +0.0062 | 0.998 | 1.000 |
| Incredibles_vs_Peppa | child:ROI->cg:ROI | -0.0046 | 0.545 | 1.000 | +0.0086 | 0.996 | 1.000 |
| Incredibles_vs_Peppa | child:ROI->cg:HRV | +0.0022 | 0.946 | 1.000 | -0.0227 | 0.916 | 1.000 |
| Incredibles_vs_Peppa | cg:ROI->child:ROI | +0.0156 | 0.580 | 1.000 | +0.0672 | 0.479 | 1.000 |
| Incredibles_vs_Peppa | cg:ROI->child:HRV | -0.0081 | 0.827 | 1.000 | -0.0113 | 0.821 | 1.000 |
| Incredibles_vs_Peppa | child:HRV->cg:ROI | +0.0041 | 0.754 | 1.000 | +0.0162 | 0.758 | 1.000 |
| Incredibles_vs_Peppa | child:HRV->cg:HRV | +0.0096 | 0.387 | 1.000 | -0.0540 | 0.798 | 1.000 |
| Incredibles_vs_Peppa | cg:HRV->child:ROI | -0.0412 | 0.439 | 1.000 | -0.0742 | 0.398 | 1.000 |
| Incredibles_vs_Peppa | cg:HRV->child:HRV | +0.0236 | 0.233 | 0.991 | +0.0526 | 0.523 | 1.000 |
| Incredibles_vs_Brave | child:ROI->cg:ROI | -0.0078 | 0.783 | 1.000 | -0.0699 | 0.544 | 1.000 |
| Incredibles_vs_Brave | child:ROI->cg:HRV | +0.0021 | 0.832 | 1.000 | +0.0124 | 0.910 | 1.000 |
| Incredibles_vs_Brave | cg:ROI->child:ROI | +0.0234 | 0.114 | 0.888 | +0.1117 | 0.098 | 0.872 |
| Incredibles_vs_Brave | cg:ROI->child:HRV | +0.0153 | 0.377 | 1.000 | +0.0598 | 0.477 | 1.000 |
| Incredibles_vs_Brave | child:HRV->cg:ROI | +0.0032 | 0.602 | 1.000 | +0.0032 | 0.595 | 1.000 |
| Incredibles_vs_Brave | child:HRV->cg:HRV | +0.0028 | 0.881 | 1.000 | -0.0028 | 0.854 | 1.000 |
| Incredibles_vs_Brave | cg:HRV->child:ROI | -0.0051 | 1.000 | 1.000 | +0.0123 | 0.686 | 1.000 |
| Incredibles_vs_Brave | cg:HRV->child:HRV | -0.0302 | 0.375 | 1.000 | -0.0403 | 0.505 | 1.000 |
| Peppa_vs_Brave | child:ROI->cg:ROI | -0.0032 | 0.761 | 1.000 | -0.0785 | 0.492 | 1.000 |
| Peppa_vs_Brave | child:ROI->cg:HRV | -0.0000 | 0.877 | 1.000 | +0.0351 | 0.823 | 1.000 |
| Peppa_vs_Brave | cg:ROI->child:ROI | +0.0078 | 0.383 | 1.000 | +0.0445 | 0.368 | 1.000 |
| Peppa_vs_Brave | cg:ROI->child:HRV | +0.0234 | 0.234 | 0.991 | +0.0710 | 0.369 | 1.000 |
| Peppa_vs_Brave | child:HRV->cg:ROI | -0.0008 | 0.826 | 1.000 | -0.0130 | 0.836 | 1.000 |
| Peppa_vs_Brave | child:HRV->cg:HRV | -0.0068 | 0.479 | 1.000 | +0.0512 | 0.926 | 1.000 |
| Peppa_vs_Brave | cg:HRV->child:ROI | +0.0360 | 0.437 | 1.000 | +0.0865 | 0.195 | 0.990 |
| Peppa_vs_Brave | cg:HRV->child:HRV | -0.0538 | 0.035 | 0.557 | -0.0928 | 0.222 | 0.990 |

#### Test 6: film × group interaction (re-nulled; max-T over 4 contrasts × 8 edges)

| film / contrast | edge | dDTF stat | dDTF p | dDTF p max-T | GPDC stat | GPDC p | GPDC p max-T |
|---|---|---|---|---|---|---|---|
| Incredibles_vs_rest | child:ROI->cg:ROI | -0.0191 | 0.635 | 1.000 | -0.0612 | 0.640 | 1.000 |
| Incredibles_vs_rest | child:ROI->cg:HRV | +0.0282 | 0.349 | 0.999 | +0.3572 | 0.017 | 0.258 |
| Incredibles_vs_rest | cg:ROI->child:ROI | +0.0068 | 0.863 | 1.000 | +0.0274 | 0.825 | 1.000 |
| Incredibles_vs_rest | cg:ROI->child:HRV | -0.0075 | 0.852 | 1.000 | +0.0082 | 0.957 | 1.000 |
| Incredibles_vs_rest | child:HRV->cg:ROI | -0.0679 | 0.177 | 0.950 | -0.0772 | 0.486 | 1.000 |
| Incredibles_vs_rest | child:HRV->cg:HRV | -0.0558 | 0.189 | 0.959 | -0.1789 | 0.124 | 0.926 |
| Incredibles_vs_rest | cg:HRV->child:ROI | -0.1156 | 0.064 | 0.682 | -0.1605 | 0.109 | 0.892 |
| Incredibles_vs_rest | cg:HRV->child:HRV | -0.0307 | 0.671 | 1.000 | -0.0936 | 0.534 | 1.000 |
| Incredibles_vs_Peppa | child:ROI->cg:ROI | -0.0238 | 0.611 | 1.000 | -0.0324 | 0.827 | 1.000 |
| Incredibles_vs_Peppa | child:ROI->cg:HRV | +0.0382 | 0.271 | 0.993 | +0.3880 | 0.029 | 0.429 |
| Incredibles_vs_Peppa | cg:ROI->child:ROI | -0.0286 | 0.569 | 1.000 | -0.0662 | 0.627 | 1.000 |
| Incredibles_vs_Peppa | cg:ROI->child:HRV | -0.0208 | 0.639 | 1.000 | +0.0013 | 0.992 | 1.000 |
| Incredibles_vs_Peppa | child:HRV->cg:ROI | -0.0777 | 0.134 | 0.915 | -0.1392 | 0.234 | 0.977 |
| Incredibles_vs_Peppa | child:HRV->cg:HRV | -0.0601 | 0.184 | 0.957 | -0.2163 | 0.120 | 0.918 |
| Incredibles_vs_Peppa | cg:HRV->child:ROI | -0.0950 | 0.144 | 0.920 | -0.1532 | 0.149 | 0.952 |
| Incredibles_vs_Peppa | cg:HRV->child:HRV | -0.0166 | 0.817 | 1.000 | -0.0641 | 0.692 | 1.000 |
| Incredibles_vs_Brave | child:ROI->cg:ROI | -0.0145 | 0.782 | 1.000 | -0.0901 | 0.533 | 1.000 |
| Incredibles_vs_Brave | child:ROI->cg:HRV | +0.0182 | 0.584 | 1.000 | +0.3264 | 0.045 | 0.628 |
| Incredibles_vs_Brave | cg:ROI->child:ROI | +0.0421 | 0.273 | 0.993 | +0.1209 | 0.405 | 0.999 |
| Incredibles_vs_Brave | cg:ROI->child:HRV | +0.0057 | 0.898 | 1.000 | +0.0152 | 0.938 | 1.000 |
| Incredibles_vs_Brave | child:HRV->cg:ROI | -0.0581 | 0.388 | 1.000 | -0.0151 | 0.922 | 1.000 |
| Incredibles_vs_Brave | child:HRV->cg:HRV | -0.0515 | 0.326 | 0.998 | -0.1416 | 0.278 | 0.992 |
| Incredibles_vs_Brave | cg:HRV->child:ROI | -0.1361 | 0.088 | 0.809 | -0.1678 | 0.190 | 0.959 |
| Incredibles_vs_Brave | cg:HRV->child:HRV | -0.0448 | 0.612 | 1.000 | -0.1231 | 0.460 | 1.000 |
| Peppa_vs_Brave | child:ROI->cg:ROI | +0.0093 | 0.857 | 1.000 | -0.0577 | 0.680 | 1.000 |
| Peppa_vs_Brave | child:ROI->cg:HRV | -0.0200 | 0.547 | 1.000 | -0.0616 | 0.704 | 1.000 |
| Peppa_vs_Brave | cg:ROI->child:ROI | +0.0707 | 0.121 | 0.905 | +0.1871 | 0.185 | 0.959 |
| Peppa_vs_Brave | cg:ROI->child:HRV | +0.0265 | 0.434 | 1.000 | +0.0138 | 0.928 | 1.000 |
| Peppa_vs_Brave | child:HRV->cg:ROI | +0.0196 | 0.791 | 1.000 | +0.1241 | 0.438 | 1.000 |
| Peppa_vs_Brave | child:HRV->cg:HRV | +0.0086 | 0.846 | 1.000 | +0.0747 | 0.570 | 1.000 |
| Peppa_vs_Brave | cg:HRV->child:ROI | -0.0410 | 0.595 | 1.000 | -0.0146 | 0.911 | 1.000 |
| Peppa_vs_Brave | cg:HRV->child:HRV | -0.0282 | 0.648 | 1.000 | -0.0590 | 0.660 | 1.000 |

#### Factorial group terms (renull + tr SS_hyp; each term unadjusted)

| term | dDTF SS_hyp | dDTF p | GPDC SS_hyp | GPDC p |
|---|---|---|---|---|
| G | 0.00 | 0.692 | 0.05 | 0.597 |
| GxF | 0.06 | 0.230 | 0.05 | 0.808 |
| GxE | 0.04 | 0.869 | 0.95 | 0.247 |
| Gxdirection | 0.00 | 0.596 | 0.01 | 0.733 |
| Gxsource | 0.02 | 0.206 | 0.79 | 0.007 |
| Gxtarget | 0.00 | 0.614 | 0.00 | 0.963 |

### Stage 06 (optional Bayesian effect sizes)

Running at the time of writing; results to follow (`06_group/<estimator>/stage06_effects.csv`, `06_group/group_model_gate.html`).

## Notes and limitations

- **Film-contrast dyad set.** Tests 5/5b and the factorial block use the dyads present in every film weighted by any
  configured contrast (one rearrangement set for the whole contrast family), not a per-contrast set. With these data
  the sets are identical (W_116 misses Incredibles and Brave). Test 6 (`combine` matrix) likewise drops a dyad
  missing any film of a contrast.
- **dDTF dominant-direction test** was conservative in the end-to-end null run (0.7 %, CI 0.1–3.7 %).
- **Stage 00b** (smoothness demo) keeps its earlier settings (YW, linear, p = 2, 0.2–1.0 Hz) and was not re-run; its
  docstring now says so.
- **Stale files** from the earlier runs remain beside the new ones in `Interbrain_ffDTF_analysis/`: `04_ffdtf/`
  (now `04_connectivity/`), and in `05_surrogate/` the old `*_null*.npz`, `*_delta*.npz` and
  `stage05_delta_table.csv` (pooled "film" scope). They are not read by any stage.
- `stage04_ffdtf.py` was renamed `stage04_connectivity.py` (config key `stage04_connectivity`). The analysis folder name
  `Interbrain_ffDTF_analysis` and the directory typo `mlutimodal` were kept.
- `DRIVE_ROOT` can be set by `SYNCC_DRIVE_ROOT` or a git-ignored `pipeline_config.local.json`; `SYNCC_ANALYSIS_ROOT`
  redirects stages 05–07 (used by the end-to-end check).

## Commands (from the repository root)

```bash
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage02_envelopes.py   # only if envelopes must be rebuilt
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage03_mvar_order.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage04_connectivity.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage05_surrogate.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage07_permutation_tests.py
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage06_group_model.py      # optional, long
.venv/bin/python scripts/pipeline_exploratory_mlutimodal_dDTF/stage00c_end_to_end.py 20 50
```
