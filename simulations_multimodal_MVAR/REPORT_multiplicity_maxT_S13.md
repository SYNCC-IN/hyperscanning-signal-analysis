# Multiplicity across edges (max-T) and the Bayesian complement (S13) — report (2026-11-10)

Branch `simulations_multimodal_MVAR` (local). Code: `294ba9e` (max-T / min-p, S01/S03/S04/S05/S12 readouts),
`87d2c9c` (S13). Full re-runs of S01, S03, S04, S05, S12 for dDTF and GPDC (200 replicates, N_PERM = 1000).
**S13: only a 5-replicate pilot (dDTF) so far** — the full run is pending (see the end of this report).

## 1. max-T: implementation (text for Section 2.5)

The confirmatory tests are made for a family of the 8 cross-person directed edges of the 4-node model; within-person
edges cannot be tested against pseudo-dyads and are never in a family. Every edge of the family is scored on **one shared
set of permutations**, so that the joint null distribution of the 8 statistics — including their dependence — is
preserved. At the study level the permutations are random re-pairings of caregivers with children (within groups for the
within-group null), and the statistic of an edge is the mean of its paired values; pairs whose surrogate fit was unstable
(NaN) are left out of that mean, separately for every permutation and edge. For the group contrast the permutations are
relabellings of dyads, shared by all edges, and the statistic is |mean₁ − mean₀| of Δ (dyads with a missing value of an
edge are left out of that edge only). Because the edges are measured on different scales, each statistic is standardized
by its own permutation distribution, u = (T − mean_perm)/sd_perm, before the maximum over the family, M_b = max_e u_e,b,
is taken for every permutation b. The single-step adjusted p-value of edge e is (1 + #{M_b ≥ u_e,obs})/(1 + B). In the
Westfall–Young step-down variant (Westfall & Young, 1993; Nichols & Holmes, 2002), edges are ordered by decreasing
u_obs, the maximum is taken only over the edges not yet tested (successive maxima), and the adjusted p-values are made
monotone in that order; step-down rejects a superset of single-step and controls the family-wise error rate (FWER) in
the strong sense under subset pivotality. Because the nulls of bounded measures are skewed, standardization by the SD may
be inexact; as a sensitivity check we also computed the rank-based min-p procedure, in which every permuted statistic is
converted to a per-edge p-value by its rank among the observed and permuted values of that edge, and
p_e = (1 + #{min_e p_b,e ≤ p_e,obs})/(1 + B) (with the same step-down). Unadjusted per-edge p-values from the same
permutations and Bonferroni (8 × p) are reported for comparison. Implementation: `src/sim_inference.py`
(`pairing_permutation_maxT`, `group_permutation_maxT`, `maxT_adjust`, `minP_adjust`).

Note for the text: the family tests use a random stream separate from the per-edge tests, so every per-edge number of the
earlier runs is reproduced exactly (Section 6).

## 2. S3 — family-wise false-positive rate under zero coupling

Range over the 9 cells N ∈ {10, 25, 50} × length ∈ {60, 120, 300 s}; 200 studies per cell; family = 8 edges.
"per-edge after max-T" = mean rejection rate of a single edge after correction.

| estimator | level | null | uncorrected | Bonferroni | max-T (step-down) | min-p | per-edge, uncorrected | per-edge after max-T |
|---|---|---|---|---|---|---|---|---|
| dDTF | study | pooled | 0.265–0.370 | 0.035–0.065 | 0.035–0.080 | 0.035–0.065 | 0.041–0.064 | 0.004–0.010 |
| dDTF | study | exclude | 0.265–0.365 | 0.025–0.060 | 0.035–0.080 | 0.025–0.060 | 0.041–0.061 | 0.005–0.010 |
| dDTF | study | within_group | 0.240–0.345 | 0.020–0.065 | 0.025–0.085 | 0.025–0.065 | 0.037–0.056 | 0.003–0.011 |
| dDTF | group | pooled | 0.280–0.355 | 0.005–0.060 | 0.025–0.065 | 0.040–0.060 | 0.040–0.056 | 0.003–0.009 |
| dDTF | group | exclude | 0.285–0.370 | 0.010–0.070 | 0.030–0.070 | 0.025–0.070 | 0.040–0.058 | 0.004–0.009 |
| dDTF | group | within_group | 0.245–0.430 | 0.015–0.055 | 0.040–0.090 | 0.020–0.085 | 0.038–0.077 | 0.006–0.013 |
| GPDC | study | pooled | 0.285–0.355 | 0.030–0.065 | 0.030–0.070 | 0.030–0.065 | 0.043–0.052 | 0.004–0.009 |
| GPDC | study | exclude | 0.275–0.365 | 0.020–0.060 | 0.030–0.070 | 0.020–0.060 | 0.043–0.052 | 0.004–0.009 |
| GPDC | study | within_group | 0.275–0.350 | 0.025–0.075 | 0.025–0.065 | 0.025–0.075 | 0.042–0.052 | 0.003–0.009 |
| GPDC | group | pooled | 0.275–0.435 | 0.005–0.075 | 0.025–0.085 | 0.020–0.075 | 0.040–0.066 | 0.003–0.011 |
| GPDC | group | exclude | 0.285–0.440 | 0.000–0.070 | 0.025–0.080 | 0.025–0.070 | 0.041–0.067 | 0.003–0.010 |
| GPDC | group | within_group | 0.355–0.490 | 0.005–0.075 | 0.040–0.095 | 0.025–0.095 | 0.051–0.082 | 0.006–0.013 |

- Uncorrected, any of 8 edges is rejected in about a third of null studies (1 − 0.95⁸ = 0.34).
- max-T single-step gave the same family-wise rate as step-down in every cell (under the global null the first step is
  identical), so it is not listed separately; mean max-T FWER over cells 0.049–0.067.
- Bonferroni is conservative, most at N = 10 in the group contrast: with 5 vs 5 dyads there are only 252 relabellings, the
  smallest attainable p is 2/252 ≈ 0.008 > 0.05/8, so Bonferroni can never reject (FWER 0.000–0.005 in those cells).
- The within-group group contrast is slightly liberal already per edge (0.073–0.082 at N = 10, GPDC; 0.077 at N = 10,
  300 s, dDTF); this carries into its max-T FWER (up to 0.090–0.095). This is a property of the within-group null at
  small N (5 dyads per group, 20 surrogate pairs per group), not of max-T.

## 3. S4 — power of the planted edge before and after max-T

Planted edge cg:ROI→child:ROI; family = all 8 cross-person edges. Design A = study level (pooled null), design B = group
contrast (within-group null). 60 s, power (200 studies):

| design | gain | dDTF N=10: unc / max-T / min-p | dDTF N=25 | dDTF N=50 | GPDC N=10 | GPDC N=25 | GPDC N=50 |
|---|---|---|---|---|---|---|---|
| A | 0.0 | 0.02 / 0.00 / 0.00 | 0.07 / 0.00 / 0.00 | 0.06 / 0.00 / 0.00 | 0.03 / 0.00 / 0.00 | 0.06 / 0.00 / 0.01 | 0.06 / 0.00 / 0.00 |
| A | 0.1 | 0.47 / 0.18 / 0.17 | 0.80 / 0.50 / 0.47 | 0.94 / 0.83 / 0.82 | 0.44 / 0.18 / 0.17 | 0.72 / 0.40 / 0.38 | 0.92 / 0.80 / 0.79 |
| A | 0.2 | 1.00 / 0.96 / 0.96 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 0.94 / 0.92 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| B | 0.0 | 0.03 / 0.02 / 0.01 | 0.06 / 0.00 / 0.00 | 0.05 / 0.00 / 0.00 | 0.08 / 0.02 / 0.02 | 0.08 / 0.00 / 0.00 | 0.06 / 0.00 / 0.00 |
| B | 0.1 | 0.12 / 0.04 / 0.01 | 0.22 / 0.05 / 0.05 | 0.36 / 0.15 / 0.12 | 0.16 / 0.04 / 0.04 | 0.20 / 0.06 / 0.04 | 0.32 / 0.12 / 0.10 |
| B | 0.2 | 0.40 / 0.14 / 0.10 | 0.90 / 0.62 / 0.57 | 1.00 / 0.99 / 0.98 | 0.45 / 0.18 / 0.14 | 0.90 / 0.61 / 0.60 | 1.00 / 0.99 / 0.99 |
| B | 0.3 | 0.84 / 0.54 / 0.36 | 1.00 / 1.00 / 1.00 | 1.00 | 0.86 / 0.56 / 0.40 | 1.00 / 0.99 / 0.99 | 1.00 |
| B | 0.4 | 0.98 / 0.84 / 0.50 | 1.00 | 1.00 | 1.00 / 0.84 / 0.60 | 1.00 | 1.00 |
| B | 0.5 | 1.00 / 0.89 / 0.57 | 1.00 | 1.00 | 0.99 / 0.92 / 0.63 | 1.00 | 1.00 |

**Proposed readout for the text — smallest gain with ≥ 80 % power, 60 s** (gain grid 0.1 steps; identical for dDTF and GPDC):

| design | N | uncorrected | max-T | min-p |
|---|---|---|---|---|
| A (study level) | 10 | 0.2 | 0.2 | 0.2 |
| A (study level) | 25 | 0.2 | 0.2 | 0.2 |
| B (group contrast) | 10 | 0.3 | 0.4 | not reached (≤ 0.5) |
| B (group contrast) | 25 | 0.2 | 0.3 | 0.3 |

At 120 s max-T raises the threshold by one step in A at N = 10 (0.1 → 0.2), in B at N = 10 (0.2 → 0.3) and in B at
N = 50 (0.1 → 0.2); at 300 s nothing changes (`summary.json` → `min_gain_for_80pct_power_adjusted`). min-p never reaches
80 % in design B at N = 10: with 5 vs 5 dyads its rank-based p-values are too coarse (see S3).

**Non-planted edges after max-T** (`summary_nonplanted_maxT.csv`). The 6 edges unrelated to the planted one stay at
0.00–0.03 (design A) and 0.00–0.055 (design B) in every cell, for both estimators. The **reverse edge child:ROI→cg:ROI
stays high for dDTF**: after max-T, design A at N = 25, 60 s, it is rejected in 0.61 / 0.99 / 1.00 of studies at gains
0.2 / 0.3 / 0.4, and in 1.00 at 300 s already from gain 0.2 (design B: 0.10–0.79 at N = 25, 60 s). This is a bias of
the measure — the reverse dDTF of a strong one-way link is inflated by the estimation noise of the partial coherence
(see HANDOFF, "przeciek kierunkowy") — and max-T does not remove it: max-T controls false positives under the null of no
coupling, not the bias of a non-null measure. For GPDC the reverse edge after max-T is ≤ 0.07 at 60 and 120 s and rises
only at 300 s with N ≥ 25 (up to 0.22 at N = 25 and 0.55 at N = 50, gain 0.5).

## 4. S1, S5, S12 after max-T (8 cross-person edges; no criteria)

- **S1** (planted cg:ROI→child:ROI 0.4, cg:HRV→child:HRV 0.3, cg:HRV→child:ROI 0.3): planted edges 1.00 before and after
  max-T (both estimators). Absent edges after max-T — dDTF: child:ROI→cg:ROI, child:HRV→cg:HRV and child:ROI→cg:HRV
  1.00 (unchanged; reverse/indirect leakage), the other two 0.005–0.010; GPDC: all absent edges ≤ 0.005 (uncorrected ≤ 0.02).
- **S5** (4-node model; absent-edge rate uncorrected → max-T): cascade, absent cg:ROI→child:HRV — dDTF 0.235 → 0.075
  (gain 0.2), 1.00 → 1.00 (0.4, 0.6); GPDC 0.04 / 0.005 / 0.005 → 0.000. Common driver, absent cg:HRV→child:ROI — dDTF
  0.025 / 0.020 / 0.005 → 0.000; GPDC 0.050 / 0.085 / 0.050 → 0.000 / 0.025 / 0.010. Omitted driver (a true dyad-level
  path, expected to be detected): dDTF heart edge 0.975–1.00 after max-T, reverse 0.59 (0.2) and 1.00; GPDC 0.90–1.00,
  reverse 0.07 / 0.665 / 0.915. Planted edges 1.00 throughout.
- **S12** (mutual coupling, N = 25): forward 1.00 everywhere. Reverse after max-T — GPDC: one-way 0.3 → false reverse
  0.06 → 0.01 (brain) and 0.010 → 0.005 (heart); weak true reverse 0.1 detected 0.905 → 0.640 (fwd 0.2) and
  0.930 → 0.790 (fwd 0.3); 0.15–0.3 reverse stays 1.00. dDTF: the false "mutual" from a one-way link stays 0.995 after
  max-T (brain, 0.3/0.0) and 1.00 (heart). Any of the other 6 edges rejected: 0.015–0.080.

## 5. S13 — group and film effects across edges (PILOT, dDTF, 5 studies per scenario)

**Provisional; do not quote.** With 5 studies per scenario every rate has a Wilson interval of about ±0.35.
Methods: perm film max-T = group contrast per film, max-T over 8 edges (edge counted as detected if any film rejects);
perm 24 = max-T over the 24 edge × film tests; perm mean = max-T over 8 edges on the film-averaged Δ; M2 per edge; M3
(1|edge); M4 (film*group|edge). Bayesian rule: 95 % HDI excludes 0.

Detection on the planted edge(s):

| scenario | effect | perm film max-T | perm 24 | perm mean | M2 | M3 | M4 |
|---|---|---|---|---|---|---|---|
| c | group | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 |
| d | group (brain, heart) | 1.0, 1.0 | 1.0, 1.0 | 1.0, 1.0 | 1.0, 1.0 | 1.0, 1.0 | 1.0, 1.0 |
| e | group | 0.8 | 0.8 | 0.0 | 0.6 | 0.2 | 0.4 |
| e | film | – | – | – | 1.0 | 0.4 | 0.6 |
| e | interaction | – | – | – | 1.0 | 0.2 | 0.6 |

Family-wise rate of a group effect on edges without a planted group effect (all null edges / without the reverse of the
planted edge): scenario a — M2 0.4, every max-T variant and M3/M4 0.0; c — perm film max-T 0.4/0.2, perm 24 0.0/0.0, M2
0.8/0.4, M3 0.8/0.8, M4 0.8/0.4; d — M3 1.0/1.0, M4 1.0/0.4.

Shrinkage and leakage (standardized group effect, mean over the 5 studies; "sample" = unpooled difference of standardized
group means, the Monte Carlo reference): planted edge in c — sample −1.28, M2 −1.27, M4 −1.14, M3 −0.24; in e — sample
−0.44, M2 −0.45, M4 −0.27, M3 −0.10. On null edges of c, M3 puts its common effect (−0.24) on every edge by construction;
M4 follows the sample estimates (−0.05 to +0.04) except on the reverse edge child:ROI→cg:ROI (M4 −0.59, sample −0.65):
that "leakage" is in the data (the dDTF reverse-edge bias of Section 3 differs between groups because the forward coupling
differs), not produced by partial pooling.

Convergence: with `convergence_row` 10–80 % of fits fail the gate, but 71 of the 78 failures have no divergences and
ESS > 400 and fail only because `convergence_row` takes R-hat from `arviz.summary`, which rounds it to 2 decimals (any
R-hat ≥ 1.005 shows as 1.01 and fails "< 1.01"). S13 now records the same gate on unrounded diagnostics too (`pass_exact`)
and triggers the M4 fallback on it. **This rounding also affects `src/group_model.convergence_row` in the `main`
pipeline (stage 6); not changed here.**

Runtime (pilot, 2 processes while 8 other simulation processes ran): 25 studies in 2325 s. Per study: simulation and Δ
0.9 s, all permutation tests 0.04 s, M2 85 s (8 fits), M3 19 s, M4 60 s, M4 fallback 29 s when triggered.

## 6. Criteria

| sim | criterion | dDTF | GPDC | comment |
|---|---|---|---|---|
| S3 | max-T FWER within 0.03–0.07 in every cell, study level [proposed] | fail | fail | 8 (dDTF) / 9 (GPDC) of 54 study+group cells outside (0.025–0.095); every Wilson CI overlaps the band |
| S3 | same, group contrast [proposed] | fail | fail | as above |
| S3 | per-edge study-level FPR in band (mean over edges) | pass | pass | unchanged |
| S3 | per-edge group-level FPR in band (mean over edges) | **fail** (was pass) | **fail** (was pass) | only because the within_group cells are now included (Section 7) |
| S3 | median z | pass | pass | unchanged |
| S4 | the three existing criteria | pass | pass | unchanged |
| S13 | a: max-T FWER within 0.03–0.07 [proposed] | pending | pending | pilot 0/15 film-studies |
| S13 | c: max-T power ≥ 80 % [proposed] | pending | pending | pilot 14/15 |
| S13 | c: M4 leakage ≤ 0.10 [proposed] | pending | pending | pilot 0.8 (all null edges) — will fail for dDTF because of the reverse edge |

**Criteria that need changing**
1. *S3 "max-T FWER within 0.03–0.07 in every cell"* cannot be met reliably with 200 studies per cell: the binomial SD at
   0.05 is 0.015, so a correctly calibrated test lands outside the band in about 19 % of cells (≈ 10 of 54 expected; we see
   8–9). Proposal: "the Wilson 95 % CI of every cell overlaps 0.03–0.07, and the FWER pooled over cells is within
   0.04–0.06" (the CI part is met everywhere; the pooled means are 0.049–0.057 for every level × null except the GPDC
   within-group group contrast, 0.067 — the per-edge excess noted in Section 2), or ≥ 1000 studies per cell. The same holds for the
   per-edge criteria, which pass only because they average 8 edges.
2. *S13 c "M4 leakage into any null edge ≤ 0.10"*: for dDTF the reverse edge carries a real group difference of Δ, so it
   is not a null edge for the measure. Evaluate leakage on the null edges excluding the reverse of planted edges (column
   scope "null edges except reverse of planted" in `summary_familywise_null.csv`), and report the reverse edge separately.
3. *S13 a "max-T FWER within 0.03–0.07"* with 100 replicates (300 film-studies) has the same problem as (1); use the
   Wilson-overlap version.

## 7. Old vs new (manuscript numbers)

- **No per-edge number changes.** Every readout of `S_01/partB_summary.csv`, `S_03/summary.csv` (pooled, exclude),
  `S_04/summary.csv`, `S_05/summary.csv` and `S_12/summary.csv` is identical to the previous full runs (Nov 9–10) to
  machine precision, for both estimators: data and per-edge permutations use the same random stream as before, and the
  family tests draw from a separate one.
- S3 now also has the within-group null (72 new rows per estimator). Its per-edge group-contrast FPR is 0.073–0.082 at
  N = 10 (GPDC) and 0.077 at N = 10, 300 s (dDTF), which flips the existing criterion "group-level FPR within band in
  every cell" to False. If the manuscript quotes "group-level FPR 0.04–0.07 in every cell", restrict it to the pooled
  and exclude nulls or add the within-group range.
- S4 design B data are unchanged (the extra 6 edges enter only the family test), so its per-edge power is unchanged.

## Pending: S13 full run

The S13 full run has not been made yet. Estimated cost ≈ 170 s per study on one core, i.e. ≈ 2.6 h for 50 replicates
× 5 scenarios × 2 estimators on 10 cores (≈ 5.2 h for 100). Commands (replicates are seeded by index, so chunks can be
run in any order and are merged by `summarize`):

```bash
SIM_REPS=0-49 SIM_NJOBS=5 .venv/bin/python simulations_multimodal_MVAR/dDTF/S13_group_film_bayes.py full dDTF
SIM_REPS=0-49 SIM_NJOBS=5 .venv/bin/python simulations_multimodal_MVAR/dDTF/S13_group_film_bayes.py full GPDC
SIM_REPS=50-99 SIM_NJOBS=5 .venv/bin/python simulations_multimodal_MVAR/dDTF/S13_group_film_bayes.py full dDTF
SIM_REPS=50-99 SIM_NJOBS=5 .venv/bin/python simulations_multimodal_MVAR/dDTF/S13_group_film_bayes.py full GPDC
.venv/bin/python simulations_multimodal_MVAR/dDTF/compare_estimators.py
```
