# Publication figures 2, 6–14 (2026-11-13)

**Code:** commit `ce297ae` on `simulations_multimodal_MVAR`, pushed. It changes `simulations_multimodal_MVAR/dDTF/figures_manuscript.py` (functions `fig02`, `fig06`–`fig14`) and `fig2_model_and_null.py`.

**Command:** `python simulations_multimodal_MVAR/dDTF/figures_manuscript.py all` regenerates every figure in about 5 s. It also redraws the older `fig_level1_signals`, `fig_level2_signals`, `fig3_S1_recovery` and `fig_gain_conversion` with unchanged code. No simulation was run.

**Inputs:** `TXT/2026_Multimodal_MVAR/Results/`. **Outputs:** `TXT/2026_Multimodal_MVAR/Figures/`. The figures themselves are not in git.

## Files

| Figure | Files | Size |
|---|---|---|
| Fig. 2 | `fig02_model_and_null.svg/.png` | 190 × 163 mm |
| Fig. 6 | `fig06_calibration.svg/.png` | 190 × 160 mm |
| Fig. 7 | `fig07_power.svg/.png` | 190 × 193 mm |
| Fig. 8 | `fig08_specificity.svg/.png` | 190 × 198 mm |
| Fig. 9 | `fig09_confounds.svg/.png` | 190 × 193 mm |
| Fig. 10 | `fig10_sensitivity.svg/.png` | 190 × 137 mm |
| Fig. 11 | `fig11_envelope_vs_phase.svg/.png` | 190 × 66 mm |
| Fig. 12 | `fig12_cardiac.svg/.png` | 190 × 168 mm |
| Fig. 13 | `fig13_mutual_coupling.svg/.png` | 190 × 74 mm |
| Fig. 14 | `fig14_group_film.svg/.png` | 190 × 168 mm |
| Captions | `captions.md` | one caption per figure |

PNGs are 300 dpi. SVGs keep their text editable (`svg.fonttype = "none"`, DejaVu Sans).

## Data used per panel

All paths are relative to `Results/`. "cmp" means `comparison/`.

- **Fig. 2.** Schematic with illustrative data from `fig2_model_and_null.py`.
  - Only change: the panel-A legend moved below the heart nodes. The panel letter A moved left because the taller panel became narrower.
- **Fig. 6.**
  - A: `cmp/S_03_summary.csv`, the study, dyad and group rates averaged over the 8 edges.
  - B: same file, within-group null at N = 16, 25, 50.
  - C: `cmp/S_03_summary_fwer.csv`, methods `none` and `maxT`. Study level from the pooled-null rows; group level for the pooled and within-group nulls.
  - D: `<est>/S_03/z_values_pooled.npz`, all cells and edges.
- **Fig. 7.**
  - A, B, C: `cmp/S_04_summary.csv`.
  - D, E, F: `cmp/S_04b_summary.csv`.
- **Fig. 8.**
  - A: `cmp/S_08_summary.csv`, `reverse_detection`.
  - B, D: `cmp/S_05_summary.csv`.
  - C: `cmp/S_11_summary_partA.csv`.
  - E: `cmp/S_05_summary_sweep.csv` (detection, `vif_cg_heart_real`, `cond_real`).
- **Fig. 9.**
  - A, B: `cmp/S_06_summary.csv`.
  - C, D: `cmp/S_07_summary.csv`, log-SD 0.5 rows.
  - E: `cmp/S_06b_summary_no_coupling.csv`.
  - F: `cmp/S_06b_summary_coupled.csv`.
- **Fig. 10.** `cmp/S_08_summary.csv` (`null_fpr`, `power_planted`, `reverse_detection`).
- **Fig. 11.** `cmp/S_11_summary_partB.csv`.
- **Fig. 12.**
  - A: `cmp/S_09_summary.csv`, mid band.
  - B: `cmp/S_10_summary.csv`.
- **Fig. 13.** `cmp/S_12_summary.csv`.
- **Fig. 14.**
  - A: `<est>/S_13/summary_detection.csv` (planted rows; DV `delta` for permutation tests, `delta_std` for Bayesian models).
  - B: `cmp/S_13_summary_familywise_null.csv`, scenario a, all null edges. The "max-T within film" value is computed from `<est>/S_13/raw_results_full_reps_*.csv` as the per-film family-wise rate averaged over the three films.
  - C: `cmp/S_13_summary_shrinkage.csv`, planted edges of c, d, e.
  - D: `cmp/S_13_summary_familywise_null.csv`, scenario c, both scopes.

## Choices and what could not be drawn exactly as specified

- **5 vs. 5 cells.**
  - S4 design B at N = 10 is dropped.
  - In S3 I dropped the N = 10 group level for every null (pooled, exclude and within-group), not only the within-group null, because all of them are 5 vs. 5 contrasts. Study- and dyad-level results at N = 10 stay.
- **Wilson intervals for pooled rates.** Several summary rates are means over edges (S3, S8) or over dyads (dyad level). For these the interval treats every study × edge (or dyad) decision as one trial, which overstates its precision. The captions say this. Family-wise rates (one value per study) and single-edge rates use the number of studies.
- **Fig. 6B, segment length.** 8 vs. 8 dyads exists only for 60 s, so that point has no 120 or 300 s companion.
- **Fig. 6C.** The exclude null is left out of the family-wise panel to keep it readable; it lies within 0.03 (dDTF) and 0.015 (GPDC) of the pooled null in `S_03_summary_fwer.csv`.
- **Fig. 7C.** Max-T vs. per-edge power is a scatter (one point per design × N × gain at 60 s), not a set of curves.
- **Fig. 7F.** The reverse-edge rate is shown for both measures (GPDC as the comparison), at gain 0.4.
- **Fig. 8E.** The collinearity diagnostics are a small panel, not an inset. The VIF and condition number are measure-independent (the same in both runs), so one line is drawn.
- **Fig. 9C.** Only the between-person gain spread of 0.5 is shown; the CSV also has spread 0. Wilson bands are drawn only for the gain ratio of 3, to avoid clutter.
- **Fig. 9F.** All three planted edges had power 1.00, so one line per measure shows the lowest of the three.
- **Fig. 12A.** The narrow and wide bands are omitted (mid band only, as the handoff allows).
- **Fig. 12B.** A heatmap, not grouped bars. The no-coupling rows have one value.
- **Fig. 14A.** For "max-T within film" the rate is detection in any film. For scenario d it is the mean over the two planted edges. The pooled-effects fallback model is not shown.
- Nothing requested had to be left out for lack of data.

## Numbers in captions vs. the manuscript text

The captions carry design numbers only: N, segment lengths, studies, gains, ratios, bands, model order. All of them agree with the manuscript text (DRAFT in Google Docs, read 2026-11-13).

The only result number I checked against the text is the family-wise rate of the group contrast with max-T within film in scenario a (Fig. 14B). Computed from the raw files it is 0.057 / 0.103 per film, matching the text's 5.7% / 10.3%. `S_13_summary_familywise_null.csv` gives 0.16 / 0.28 for the same method, because it counts a rejection in any of the three films. Fig. 14B uses the per-film value, as the text does.

No other discrepancy between caption numbers and the text.

## Not done

- Placing the figures into the Google Doc.
- Replacing the `[FIG n]` placeholders with these captions.
