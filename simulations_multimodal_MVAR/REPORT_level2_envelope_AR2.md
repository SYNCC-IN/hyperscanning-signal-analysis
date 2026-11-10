# Level-2 envelopes as resonant AR(2): old vs. new values (S2, S10, S11)

Code commit `ae5f7c7` ("Level-2 envelopes: resonant AR(2) log-amplitude"). The full runs used 100 replicates per condition with both estimators. **Old** is the working-tree results before this change (the least-squares re-run of 2026-11-07/08). **New** is the AR(2) re-run of 2026-11-09. All values are given as dDTF / GPDC.

## Envelope parameters used (for Methods, Section 2.6, Level 2)

- **S2, S11 Part B (truths "none" and "amplitude"):** each person's log-amplitude VAR at 2 Hz has an AR(2) self-resonance.
  - The pole radius is drawn per person from U(0.70, 0.85), and the pole frequency from U(0.20, 0.40) Hz.
  - The draws use a random stream separate from the signal noise.
  - The cross-person edge is unchanged: caregiver → child, lag 1, gain 0.5.
  - The innovation SD is 1/SNR = 0.25, as before.
  - For example, with r = 0.8 and f₀ = 0.3 Hz the log-amplitude SD is 0.40 (it was 0.26 with the AR(1) self-term 0.3). The modulation depth of the envelope is 41% (it was 27%), and the spectral peak is at 0.30 Hz.
- **S10 (child EEG envelope):** the log-envelope is generated at 10 Hz.
  - The fast component, formerly an OU process with τ = 1 s, is now a resonant AR(2).
  - Its pole radius is drawn per child from U(0.70^0.2, 0.85^0.2) = U(0.931, 0.968) at 10 Hz. This is the same decay as 0.70–0.85 at 2 Hz.
  - Its pole frequency is drawn from U(0.20, 0.40) Hz.
  - Its SD is kept at 0.5, so the coupling strengths 0.3/0.6/1.0 keep their size relative to the envelope's own variability.
  - The slow component (OU, τ = 15 s, SD 0.5) is unchanged.
- **Phase truth (S2, S11):** unchanged. It uses independent amplitude noise (OU, τ = 10 s, SD 0.2), and its old and new rows are identical.
- **Stability:** `generate_var_process` now asserts that all companion roots have modulus below 1. No run failed this check.
- **Pilot (15 replicates of S10 and S11, both estimators):** detection did not collapse or saturate, and no gains were retuned.

## Section 3.7: S2 (descriptive sweep, gap 0–9.5 Hz)

| Readout | Old | New |
|---|---|---|
| Amplitude truth, envelope path: relative value across gaps | 99–111% / 99–102% | 100–107% / 99–101% |
| Amplitude truth, envelope path: direction index | 0.89–0.91 / 0.68–0.72 | 0.82–0.85 / 0.60–0.64 |
| Amplitude truth, phase path: relative value at gap 5 Hz | 1.7% / 55% | 4.3% / 65% |
| No coupling, phase path: relative value at gap 5 Hz | 2.0% / 55% | 3.4% / 69% |
| No coupling, phase path: direction index range | −1.00 to 0.01 / 0.01 to 0.53 | −1.00 to 0.02 / 0.01 to 0.52 |
| Amplitude truth, fixed-grid phase path: relative value range | 0–541% / 23–252% | 0–318% / 30–247% |
| Phase truth (unchanged generator) | identical | identical |

The phase path behaves the same under amplitude coupling and under no coupling, in both the old and the new runs. The conclusion that its decay is a normalization artefact still holds.

## Section 3.7: S11 Part B (study-level detection, *N* = 25, 120 s)

| Truth | Gap | Envelope MVAR [est] | Coherence | Lagged xcorr | Pairwise GC | PLV |
|---|---|---|---|---|---|---|
| amplitude | 0 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.09 → 0.10 |
| amplitude | 3 Hz | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 1.00 → 1.00 | 0.05 → 0.06 |
| none | 0 | 0.05/0.04 → 0.01/0.02 | 0.03 → 0.02 | 0.01 → 0.03 | 0.05 → 0.04 | 0.02 → 0.01 |
| none | 3 Hz | 0.07/0.06 → 0.06/0.09 | 0.05 → 0.07 | 0.05 → 0.07 | 0.08 → 0.08 | 0.03 → 0.03 |
| phase | 0 / 3 Hz | unchanged (0.05/0.07, 0.03/0.05) | 0.05, 0.02 | 0.07, 0.06 | 0.14, 0.08 | 1.00, 1.00 |

**Manuscript text, old → new:**
- Envelope methods detect the amplitude truth in 100% of studies (unchanged).
- PLV detects the amplitude truth in 5–9% → 6–10%.
- Phase coupling: PLV 100%, envelope methods 2–14% (unchanged).
- No coupling: ≤ 8% → ≤ 9%. The new maximum is GPDC at gap 3 Hz.

## Section 3.8 / Table 4: S10 (caregiver cardiac → child EEG, strengths 0.3 / 0.6 / 1.0)

**Forward edge**

| Option | Truth | Old | New |
|---|---|---|---|
| A | linear | 1.00 at every strength (both) | 1.00 at every strength (both) |
| A | amp_to_eeg | 0.28/0.67/0.94 · 0.22/0.52/0.87 | 0.24/0.77/0.94 · 0.20/0.63/0.79 |
| A | none | 0.06 · 0.07 | 0.07 · 0.04 |
| A′ | linear | 0.15/0.92/1.00 · 0.17/0.92/1.00 | 0.16/0.91/1.00 · 0.13/0.90/1.00 |
| A′ | amp_to_eeg | 1.00 at every strength (both) | 1.00 at every strength (both) |
| A′ | none | 0.04 · 0.04 | 0.03 · 0.02 |
| B | linear | 0.12/0.73/1.00 · 0.14/0.65/1.00 | 0.18/0.85/1.00 · 0.16/0.69/1.00 |
| B | amp_to_eeg | 0.97/1.00/1.00 · 0.90/1.00/1.00 | 1.00/1.00/1.00 · 1.00/1.00/1.00 |
| B | none | 0.05 · 0.08 | 0.07 · 0.03 |

**Reverse edge (child EEG → caregiver cardiac)**

| Option | Truth | Old | New |
|---|---|---|---|
| A | linear | 1.00/1.00/1.00 · 0.08/0.33/0.58 | 0.95/1.00/1.00 · 0.13/0.34/0.56 |
| A | amp_to_eeg | 0.21/0.50/0.73 · 0.16/0.32/0.58 | 0.24/0.59/0.75 · 0.22/0.34/0.58 |
| A′ | linear | 0.21/0.73/1.00 · 0.15/0.32/0.68 | 0.13/0.84/1.00 · 0.05/0.36/0.55 |
| A′ | amp_to_eeg | 0.99/1.00/1.00 · 0.75/1.00/1.00 | 1.00/1.00/1.00 · 0.84/1.00/1.00 |
| B | linear | 0.15/0.46/0.99 · 0.12/0.35/0.73 | 0.15/0.56/0.99 · 0.15/0.39/0.75 |
| B | amp_to_eeg | 0.88/1.00/1.00 · 0.65/1.00/1.00 | 0.94/1.00/1.00 · 0.69/1.00/1.00 |
| none | (all options) | 0.03–0.06 · 0.02–0.05 | 0.06–0.07 · 0.06–0.07 |

**Manuscript text, old → new:** with option A under linear coupling, the reverse edge was detected in "dDTF 100%, GPDC 8–58%" → "dDTF 95–100%, GPDC 13–56%". The qualitative conclusion is unchanged. Each option is most sensitive to its own coupling type but also detects the other type at strong coupling, so the representation does not identify the coupling type.

## Acceptance criteria (Supplementary Table S1)

No criterion changed status for S2, S10 or S11.

| Simulation | dDTF | GPDC |
|---|---|---|
| S2 | 3/3 | 3/3 |
| S10 | 4/6 | 4/6 (both fail "A does not detect amp_to_eeg" and "A′ does not detect linear") |
| S11 | 6/7 (fails "absent indirect edge ≤ 0.1", from Part A) | 7/7 |

## Files

- Per-estimator results: `simulations_multimodal_MVAR/dDTF/Results/{dDTF,GPDC}/S_{02,10,11}/`.
- Comparison: `Results/comparison/S_02_summary.csv`, `S_10_summary.csv`, `S_11_summary_part{A,B}.csv`. `comparison.md`, `criteria.csv` and `runs.csv` were rebuilt too, but they also hold the other simulations' uncommitted least-squares results.
- Figures: `Results/fig_level2_signals.png` (Fig. 4, AR(2) envelopes, nperseg 128, theoretical AR(2) spectra dashed). `fig_level1_signals.png` and `fig3_S1_recovery.png` were regenerated, unchanged.
- `~/mnt/2026_Multimodal_MVAR/` does not exist on this machine, so nothing was copied there.
