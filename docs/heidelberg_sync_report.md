# Heidelberg BI-Vital ↔ EEG synchronization: strategy and results

Source of the numbers: `export_report.csv` of the batch export (34 dyads listed).

## Strategy

BI-Vital ECG (200 Hz) and BrainVision EEG (1 kHz) run on separate clocks. For each wearer the pipeline models `t_dev = a + (1 + b) · t_eeg` (offset `a`, drift `b`) and estimates it from the cardiac artefact present in the wearer's own EEG:

1. EEG channels of the wearer (incl. VEOG) and the ECG are band-passed 5–25 Hz at 200 Hz.
2. Windowed cross-correlation (120 s windows, 60 s hop) gives a track of offsets per EEG channel. Channels are ranked by median |r| and by agreement of the offsets across windows; the proxy is the sign-aligned mean of the best consistent channels.
3. The coarse lag is refined with the proxy and `a`, `b` are fitted with a robust Theil–Sen line.
4. Search strategies are tried in order until one gives a consistent fit: (1) around the header start-time lag (±30 s), (2) same with drift-aware agreement, (3) wide search ±300 s re-centred on the channel consensus, (4) full range ±4 h. The strategy used is stored in `*_sync_summary.csv` (`search_strategy`); a fallback prints a `WARNING`.
5. **If every strategy fails**, the dyad is still exported: EEG as usual, while ECG/IBI/RMSSD of that member are all NaN and every `.nc` carries `sync_status = failed` and `sync_failure_reason`. The report status reads `exported, EEG only for: …`. The channel ranking and `*_sync_failures.csv` are saved in `sync_QC/<dyad>`.
6. After synchronization ECG, IBI and RMSSD are placed on the 128 Hz EEG grid (NaN-aware, gaps are not bridged).

Typical successful fit (H033, validated): offsets 3744.88 s (adult) and 3744.65 s (child), i.e. about 7 s away from the header start-time lag; drift 1.5 and 1.1 ms/min; residual SD 0.5–1 ms.

## Results

| | Dyads |
|---|---|
| Listed in report | 34 |
| Skipped (no EEG or no BI-Vital files): H016, H017, H041 | 3 |
| Exported | 31 |
| – ECG synchronized for both members | 18 |
| – one member without ECG (EEG only for that member) | 9 |
| – both members without ECG | 4 |

At device level 45 of 62 BI-Vital recordings (73%) were synchronized; 17 failed (9 adults, 8 children).

Dyads with full synchronization: H009, H014, H020, H024, H026, H030–H035, H037–H039, H042–H044, H046.

Dyads with partial or no ECG:

- both members failed: H010, H011, H027, H029
- adult (cg) only failed: H012, H019, H021, H022, H040
- child (ch) only failed: H013, H018, H023, H025

## Why the failures occur

- **No detectable cardiac artefact (majority).** All four strategies find no EEG channel with a consistent offset (agreement ≈ 0). The best channels are frontal (Fp1, Fp2, VEOG), i.e. blinks rather than ECG, and the full-range search only reaches r ≈ 0.2–0.4, the level of chance maxima. For H010 this was also checked with a drift-robust envelope method and with PCA/ICA and left–right differences: no repeatable offset, whereas the same test recovers the known offset in H033. For these recordings the artefact is either absent in the EEG or too weak next to noisy BI-Vital ECG (H010 adult).
- **ECG ends before the EEG** (H010 child, H018 child, H021 adult): the start-time strategies cannot use all windows; the remaining strategies then also find no artefact.
- **Implausible header start time** (H025 child: lag ≈ 1.78·10⁹ s): the BI-Vital header time of this recording is wrong or not parsed correctly; it needs a look at the file.
- **Partially consistent (H040 adult):** agreement 0.38–0.54 for VEOG/FT10/TP10 with the start-time strategy but a failing robust fit, which suggests a nonlinear clock behavior or a mislocated segment. This is the best candidate for manual review.

## Notes and open points

- The old firmware (first dyads, no `Drift Correction` header field) is read correctly (column mapping by header name), and H009 synchronizes.
- Dyad H025 with split BI-Vital parts is exported, but its child recording failed (see above).
- Accelerometer columns missing in some files no longer stop the export (they become NaN in the QC table).
- The report does not record which strategy was used for successful members; it is in `sync_QC/<dyad>/*_sync_summary.csv`. Recommended QC: review dyads where `search_strategy` is not `start-time lag`.
- If ECG of the failed members is needed, options are a manual offset from another source or using the header start time with an uncertainty flag (not recommended: unknown timing error would bias IBI/RMSSD).
