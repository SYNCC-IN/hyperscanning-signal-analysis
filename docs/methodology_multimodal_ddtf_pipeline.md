# Methodology: Multimodal Interbrain ffDTF + HRV Pipeline

Technical description of the eight-stage pipeline (`stage00`–`stage06`) that estimates
directed, interpersonal (caregiver ↔ child) brain and autonomic connectivity from
cleaned EEG and interbeat-interval (IBI) data, tests it against a surrogate-dyad null,
and models group differences (TD vs. ASD) at the population level.

Code: [`scripts/pipeline_exploratory_mlutimodal_dDTF/`](../scripts/pipeline_exploratory_mlutimodal_dDTF/)
(`stage00_synthetic_validation.py` … `stage06_group_model.py`), built on
[`src/design.py`](../src/design.py), [`src/assemble.py`](../src/assemble.py),
[`src/mvar_diag.py`](../src/mvar_diag.py), [`src/connectivity.py`](../src/connectivity.py),
[`src/surrogate.py`](../src/surrogate.py), [`src/group_model.py`](../src/group_model.py),
[`src/synthetic_mvar.py`](../src/synthetic_mvar.py), [`src/reporting.py`](../src/reporting.py),
[`src/mtmvar.py`](../src/mtmvar.py) (the underlying MVAR/DTF math),
[`src/pipeline_config.py`](../src/pipeline_config.py) (config loader) and
`pipeline_config.json` (all settings, single source of truth).
Design rationale and locked/open decisions: `DTF_analysis_notes/pipeline_plan.md` and
`DTF_analysis_notes/notatka_projekt_DTF_HRV_H2_H4.md` (the original scientific note).

Input: cleaned, individually band-parameterized EEG
(see [methodology_ica_cleaning_pipeline.md](methodology_ica_cleaning_pipeline.md) and
[methodology_spectral_analysis_pipeline.md](methodology_spectral_analysis_pipeline.md))
plus interpolated IBI on the same time grid.

---

## 1. Scientific goal

The pipeline targets two hypotheses about caregiver–child passive co-viewing:

- **H2 — social-attention coupling**: directed EEG-envelope connectivity
  caregiver → child over temporo-parietal cortex (P7/P8, a TPJ proxy), predicted to
  index joint/social-attention alignment and to be reduced in ASD.
- **H4 — autonomic co-regulation**: directed HRV connectivity caregiver → child,
  predicted to index autonomic co-regulation and to differ (sign not assumed a
  priori) in ASD — the literature review behind this pipeline explicitly flags that
  negative interpersonal HRV synchrony can be *adaptive*, so `TD > ASD` is never
  hard-coded as the expected direction.

H2 and H4 share one MVAR structure and are fit **together** in one model so each
hypothesis's estimate is conditioned on the other modality's dynamics — a strictly
2-variable model per hypothesis could not do this, and it is also what gives access to
the most exploratory edges (autonomic → cortical "scaffolding", caregiver HRV →
child ROI and vice versa). The three film stimuli (`Peppa`, `Incredibles`, `Brave`)
are treated as three **qualitatively different, unordered** valence conditions
(happy/low-arousal, conflict/high-arousal, mixed — borrowed and validated by prior
fNIRS work, Esposito et al.), never averaged together or treated as a scale.

## 2. Node topology: the single source of truth

Unlike an earlier, hard-wired 4-variable design, the pipeline is **config-driven**
over an arbitrary set of "nodes" declared once, in `pipeline_config.json`'s
`"shared".nodes`:

```json
{"name": "child:ROI", "role": "child",     "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"},
{"name": "cg:ROI",    "role": "caregiver", "signal": "roi_envelope", "roi_channels": ["P7"], "roi_label": "temporo-parietal"},
{"name": "child:HRV", "role": "child",     "signal": "raw_ibi"},
{"name": "cg:HRV",    "role": "caregiver", "signal": "raw_ibi"}
```

Every stage derives its MVAR row order from `src.design.node_names(nodes)` and its
signal-type sub-blocks from `src.design.rows_for_signal(nodes, signal)`; no stage
script hard-codes a fixed 4-node/2-role shape. `edge_topology` (also in the shared
config) declares the directed edges of scientific interest and their class label
(`H2_primary`, `H2_reverse`, `H4_primary`, `H4_reverse`, `exploratory`); every stage
that reads it validates at load time (`src.design.assert_edges_known`) that each edge
references a node that actually exists, raising a message naming the offending edge
rather than a bare index error. This makes the two-node-per-role default shown above
one *instance* of a general design that also supports, e.g., several ROI nodes per
role or HRV-only roles — the four production variables (`child:ROI`, `cg:ROI`,
`child:HRV`, `cg:HRV`) are simply the currently committed configuration.

**Variable definitions** (`signal` field):

- `roi_envelope`: the participant's individualized-band **instantaneous amplitude
  envelope**, on `roi_channels`, using the individualized band at `roi_label` from
  `band_assignments.csv` (see [methodology_spectral_analysis_pipeline.md](methodology_spectral_analysis_pipeline.md)).
  Envelope, not raw filtered signal, because phase-locking measures (PLV, phase-based
  DTF) are systematically biased when the two people's center frequencies differ
  (empirically confirmed for child ~8–10 Hz vs. adult ~10–12 Hz gaps in a Stage 0
  simulation, §4); the envelope compares slow power fluctuations instead of
  instantaneous phase and is robust to this mismatch. Consequence: this variable
  models slow (~1–2 s) amplitude co-ordination, not fast oscillatory phase coupling.
- `raw_ibi`: the **raw (interpolated) IBI**, downsampled only — no band-pass, no
  Hilbert envelope. This reverses an earlier design choice (an HF-band IBI envelope,
  intended to be conceptually parallel to the EEG envelope) after inspecting real
  data: EEG-rhythm envelopes fluctuate in a band (~0.2–1 Hz) that overlaps the *raw*
  IBI's RSA content, whereas the *envelope* of HF-IBI is a second-order, much slower
  signal that no longer sits in that band. Feeding the raw IBI keeps both modalities
  in a comparable band for one shared low-rate MVAR. Accepted consequence: the EEG
  side is a second-order quantity (amplitude envelope of a fast rhythm) while the HRV
  side is a first-order oscillation (the IBI itself) — internally consistent within
  each modality, relevant when interpreting the exploratory cross brain–heart edges.

All node signals are downsampled to a common rate (`TARGET_SFREQ = 2.5 Hz`, chosen so
Nyquist, 1.25 Hz, clears the child HF-reference band's upper edge, ~1.04 Hz) and
additionally pass through one identical shared band-pass
(`DESIGN_HIGHPASS_HZ`–`DESIGN_LOWPASS_HZ`, default 0.05–1.0 Hz, 2nd-order Butterworth,
zero-phase) so every variable's group delay matches.

---

## 3. Stage 0 / 0b — Synthetic validation (no real data)

Before touching real data, `stage00_synthetic_validation.py` builds known-truth
signals via `src.synthetic_mvar` (`edges_to_coupling`, `generate_var_process`,
`generate_coupled_oscillators`) and runs them through the *real* estimator code
path (`src.mtmvar.direct_dtf`/`mvar_plot`), checking three things:

1. **Orientation**: injecting a directed edge `node0 ← node1` only and confirming
   empirically which axis of the returned cube is source vs. target
   (`direct_dtf`'s output is `[target, source, freq]`) — this convention is then
   relied upon, not re-derived, by every later stage.
2. **4-node recovery**: with a known coupling matrix mimicking the real design
   (child/caregiver ROI + HRV, plus one novel cross edge), the injected edges are
   recovered with higher DTF than their reverse.
3. **Envelope vs. phase robustness**: as the injected child/caregiver center-frequency
   gap widens from 0 to 10 Hz, the envelope-based ffDTF path stays within 50% of its
   zero-gap baseline while a phase-based path's recovered coupling collapses — the
   empirical justification for the envelope-not-phase design decision in §2.

`stage00b_smoothness_artifact.py` is a companion methods check
(`self_ar2_coeffs`, `two_channel_ar2_coupling`, `simulate_two_channel_dyads`) probing
whether a spurious apparent coupling can arise purely from one channel being smoother
(higher spectral concentration) than another, independent of any injected true
coupling — a sanity check against a specific class of estimator artifact.

Both stages are fully synthetic (no real-data dependency) and are re-run whenever the
estimator core changes, as an acceptance gate.

## 4. Stage 1 — Coverage (`stage01_coverage.py`)

Composes existing loaders (`src.io_utils.get_participant_files`,
`src.assemble.assemble_dyad`) into one row per `(dyad_id, role, film, modality)`,
recording: whether the file exists, whether every expected ROI channel was found and
none was interpolated (`roi_ok`), and the film's actual duration against
`EXPECTED_FILM_LEN_S` (QC range around the nominal ~60 s). No signal processing,
alignment, or MVAR happens here — this stage only decides which dyad × film × modality
cells are usable.

The actual set of dyads carried into Stage 2 is **hand-curated**: this script
*displays* a suggestion (group ∈ `INCLUDED_GROUPS`, `roi_ok` all-True, every
film/modality/role present) but does not write it automatically — a researcher reviews
the coverage gate and edits `pipeline_config.json`'s `"shared".included_dyads` list by
hand, mirroring the manual-review checkpoint already required by the ICA-cleaning
pipeline (see [methodology_ica_cleaning_pipeline.md](methodology_ica_cleaning_pipeline.md)).

## 5. Stage 2 — Per-node envelopes / design variables (`stage02_envelopes.py`)

For each included dyad × film, builds one continuous signal per node (§2), then
segments it to the film's window:

1. **Union channel load**: the dyad's EEG is loaded once with the *union* of every ROI
   node's `roi_channels` (`src.assemble.assemble_dyad`); each node then reselects its
   own channel subset (`src.assemble.select_roi_channels`) — so two ROI nodes on the
   same role with different channel sets/individualized bands (e.g. a second ROI
   comparison track) get genuinely different envelopes, not duplicates.
2. **Per-node continuous signal** (`src.design.roi_band_envelope` for `roi_envelope`
   nodes; plain downsampling for `raw_ibi` nodes), computed on the whole continuous
   `passive_movies` chunk — filter/Hilbert/downsample first, segment to the film
   window *afterward* — so filter and Hilbert edge transients land in the discarded
   pre/post margins and inter-film gaps rather than inside the retained data
   (`src.design.segment_signal`).
3. **Stacking** (`src.design.stack_design`) into one `xarray.DataArray`
   (`dims=(variable, time)`, `variable` = `node_names(nodes)`), written to
   `02_envelopes/<dyad_id>_<film>.nc` in physical units (not yet z-scored — that is a
   Stage 3 concern) with per-node band metadata in `attrs` (`f"{name}_cf"`,
   `f"{name}_bw_half"`, `f"{name}_roi_channels"` for ROI nodes).
4. **QC gate**: one figure set per node (ROI node → raw/filtered/envelope trace + a
   PSD with the individualized band overlaid; HRV node → a raw-IBI trace), plus a
   combined design-variable PSD (z-scored for display only) confirming every node
   occupies a comparable, non-aliased frequency band at the shared rate.

## 6. Stage 3 — Design matrix, order selection, windowed-ACF MVAR fit (`stage03_mvar_order.py`)

1. **Design matrix** (`src.design.assemble_design_matrix`): selects/orders Stage 2's
   file by `node_names(nodes)` and z-scores every row across time (`ddof=0`) — once,
   globally per film, *before* windowing (never per-window, which would remove the
   cross-window variance drift the averaged-ACF fit below is meant to absorb).
2. **Model-order selection** (`src.mvar_diag.select_p_used` → `mtmvar.mvar_criterion`):
   evaluated on the **global 2-D** signal only (`mvar_criterion` does not support 3-D
   input, and is not patched to — a documented, accepted limitation). Three
   information criteria (AIC, HQ, SC) are computed:
   `AIC = ln|Σ| + 2·p·k²/n`, `HQ = ln|Σ| + 2·ln(ln n)·p·k²/n`,
   `SC = ln|Σ| + ln(n)·p·k²/n` (`Σ` = residual covariance at order `p`, `k` =
   number of channels, `n` = sample count), minimized over `p = 1..MAX_MODEL_ORDER`.
   The primary criterion (`PRIMARY_CRIT`, default `SC`) fit on the full joint system
   is `p_used`; the exploratory cross brain–heart edges only exist in the joint fit,
   so `p_used` is never split by signal type — per-signal sub-block orders
   (`signal_row_groups`, one per distinct `signal` value present in `nodes`) are
   still reported, diagnostically only, to expose an order mismatch between signal
   types.
3. **Windowed-ACF-averaged fit** (the estimation default, not a deferred upgrade):
   the design is cut into short overlapping windows
   (`src.design.window_stack`, locked geometry `WIN_LEN_S=10 s`,
   `OVERLAP_FRAC=0.5` → 25 samples/window, ~11 windows per 60 s film at 2.5 Hz),
   each window independently detrended (`detrend_windows`, linear — mean + slope only,
   not variance), then **one** MVAR is fit from the autocovariance **averaged across
   windows** (`src.mvar_diag.fit_mvar_avg_acf` → `mtmvar.ar_coeff`'s existing
   `count_corr`-based trials-axis averaging — the Kamiński sDTF core). Rationale: the
   HRV variable is non-stationary within a 60 s film (drifting RSA center
   frequency/variance), which biases a single global 2-D fit; short windows are
   closer to locally stationary. **Empirical result across all 126 real dyad × film
   cases**: the windowed fit reached "quality_ok" (AR-stable + white residuals) on
   126/126 cases, vs. 0/126 for a single global fit — independently confirmed against
   `statsmodels.VAR` and by refitting near the order cap, which is the empirical case
   for making this the default rather than a later swap-in.
4. **Diagnostics** (`src.mvar_diag`): `ar_root_stability` (companion-matrix
   eigenvalues; every root modulus < 1 ⇒ stable) and `residual_whiteness`
   (per-window one-step-ahead residuals — never predicted across a window boundary —
   ACF averaged across windows, Bartlett-band whiteness fraction), run on both the
   windowed fit and a global single-window comparison fit so the QC gate shows the
   improvement directly.
5. A **fixed-order, fixed-geometry connectivity grid** (`GRID_MODEL_ORDER`,
   `GRID_WIN_LEN_S`/`GRID_OVERLAP_FRAC`, independent of the per-case `p_used`) gives
   one comparable ffDTF/dDTF/GPDC figure per case, and a small window-choice
   sensitivity check (1–2 dyads, `10s/50%` vs. `15s/50%` vs. `10s/0%`, `p_used` held
   fixed) confirms the locked window geometry against alternatives on the primary
   edges.

## 7. Stage 4 — Connectivity estimation (`stage04_ffdtf.py`)

For every case, reads Stage 3's per-case `p_used`/window geometry **verbatim** (never
recomputed) and calls `src.connectivity.Granger_estimator`, which wraps the identical
detrended windowed 3-D stack through the estimator dispatcher
(`mtmvar.dtf_estimator`, `optimal_model_order` always explicit — never `None` on 3-D
input). Three estimator families are implemented (selectable via config `ESTIMATOR`;
default `dDTF`):

- **DTF** (base quantity): `H(f) = A(f)⁻¹` (transfer function from the frequency-domain
  AR matrix); `DTF_ij(f) = |H_ij(f)|² / Σ_k |H_ik(f)|²` — normalized so each row
  (target) sums to 1 across sources at each frequency.
- **ffDTF** (full-frequency DTF, Korzeniewska et al. 2003): normalizes by inflow
  summed over *all* frequencies instead of per-frequency —
  `ffDTF_ij(f) = |H_ij(f)|² / Σ_f Σ_k |H_ik(f)|²` — making magnitudes comparable
  across frequencies.
- **dDTF** (direct DTF, same reference): `dDTF_ij(f) = ffDTF_ij(f) · |κ_ij(f)|`, where
  `κ` is partial coherence (computed from the multivariate spectral matrix's minors)
  — this factor suppresses edges that are only *indirect* (mediated through a third
  variable), which is why **dDTF is the pipeline's production default**.
- **GPDC** (generalized partial directed coherence, Baccalá & Sameshima 2007):
  `GPDC_ij(f) = (|A_ij(f)|/σ_i) / sqrt(Σ_k |A_kj(f)|²/σ_k²)`, normalizing by each
  channel's own noise variance — available as an alternative, noise-heterogeneity
  robust estimator.

An optional Box-Cox transform (`box_cox_lambda`, `-1` = no-op sentinel, distinct from
the mathematically valid `1`) can reshape the connectivity cube's distribution before
downstream statistics; default is no transform. A synthetic **known-truth anchor** (a
2-node AR process with one injected directed edge, run through this exact code path)
is regenerated every run and shown in the gate next to the real cases, so the
estimator is never trusted on real data without an adjacent ground-truth check.

## 8. Stage 5 — Surrogate null and Δ/z (`stage05_surrogate.py`)

Both partners in a real dyad watched the same film, so raw connectivity conflates
genuine interpersonal coupling with a shared-stimulus response. Stage 5 removes that
shared component via **film-matched surrogate dyads**: a surrogate takes one dyad's
child-role node variables and a *different* dyad's caregiver-role node variables
(same film, node-derived via `src.surrogate.assemble_surrogate_design`, which pulls
every `role="child"` variable from one Stage 2 file and every `role="caregiver"`
variable from another and reassembles them in `node_names(nodes)` order — general
over any per-role node count).

1. **Common order for real and surrogate**: every fit in this stage — real dyads
   included — uses one fixed order (`COMMON_MODEL_ORDER`, default 4) and the locked
   Stage 3/4 window geometry, so a group-level dependent variable is never confounded
   by each dyad's own per-case `p_used` varying. Real connectivity is therefore
   **recomputed** at this common order, not read from Stage 4's per-case `.npz`.
2. **Null pool** (`src.surrogate.compute_null`): every ordered mismatched
   `(child_dyad, cg_dyad)` pair for a film (`N·(N−1)`, deterministic, no subsampling
   by default) is stitched, stability-gated (windowed-ACF AR companion eigenvalues,
   excluded if `max_abs_root ≥ SURROGATE_STABILITY_MAX_ROOT`, count reported), and
   estimated through the identical Stage 4 estimator path, band-averaged over the
   coupling band (`COUPLING_BAND_HZ`, default 0.2–1.0 Hz). Two pooling scopes are
   supported: **`film`** (one pool per film, shared across both diagnostic groups —
   the locked, group-agnostic reference null) and **`within_group`** (one pool per
   film × group — a sensitivity analysis; needed because the pooled null, being
   group-invariant, cancels out of every TD-vs-ASD contrast including a film × group
   interaction, and so cannot rule out a group-dependent *stimulus* response on its
   own).
3. **Signed delta/z** (`src.surrogate.delta_and_z`): for every real dyad × film ×
   directed edge, `delta_dtf = real − median(null)`, `z_vs_surrogate =
   delta_dtf / (MAD(null) · 1.4826)` — median/MAD (robust statistics), not mean/SD,
   so a handful of extreme surrogate draws cannot dominate either the center or the
   spread. Both quantities are kept **signed everywhere**, never `abs()`-ed or
   clipped, since the sign of a surviving HRV effect is scientifically meaningful
   (§1).
4. Real dyads that are `real_stable=False` at the common order are **kept and
   flagged**, never silently dropped — whether to exclude them is deferred to
   Stage 6.

Output: `stage05_delta_table.csv` (tidy, one row per dyad × film × edge:
`real_ffdtf, null_median, null_std, delta_dtf, z_vs_surrogate, real_stable, ...`),
its `_within_group`-suffixed sensitivity companion (identical schema), per-film null
`.npz` files, and a QC gate with split-violin null-vs-real plots per edge
(`src.surrogate.plot_null_vs_real_violin`, TD/ASD on opposite halves against the
grey surrogate density).

## 9. Stage 6 — Group model (`stage06_group_model.py`, Bambi/ArviZ)

The original plan specified an R/`brms` group model; `brms`/`cmdstanr` were confirmed
not installed while `bambi`/`arviz` already were, so — per an explicit project-owner
decision recorded in the script — this stage is Python/Bambi throughout, reusing
`src.group_model` for fitting/contrasts/diagnostics and `src.reporting` for the HTML
gate. The statistical design is otherwise the one in the original note:

1. **Per-edge models** (confirmatory default): for each of the 6 emphasis edges
   (`H2_primary/reverse`, `H4_primary/reverse`, 2 exploratory cross brain–heart edges
   — `EMPHASIS_EDGES` derived from `edge_topology`), fit
   `delta_dtf_z ~ film * group (sum-to-zero contrasts) + (1|dyad_id)`, Student-t
   family (robust to outliers), after: dropping `real_stable == False` rows (count
   reported even when zero), standardizing `delta_dtf` **within edge**
   (`src.group_model.standardize_within_edge`) so one shared weakly-informative
   prior set (`Normal(0,1)` on fixed effects, half-Student-t(3,1) on the dyad
   intercept SD) is equally weakly-informative regardless of an edge's raw
   magnitude. Contrasts are reported both in standardized units and back-transformed
   to raw Δ-units (`src.group_model.back_transform`).
2. **Contrasts and inference** (`src.group_model.compute_contrasts`,
   `reference_grid` + `predict_cell_means`): group main effect, the planned film
   contrast `Incredibles vs. (Peppa+Brave)/2` (overall and within each group), and
   the film × group interaction — computed the estimated-marginal-means way (predict
   the six film × group population-level cell means, then take linear combinations
   of those posterior draws), since Bambi has no `emmeans`/`hypothesis()` equivalent.
   Inference is posterior mean, 95% HDI, and directional probabilities `P(>0)`/`P(<0)`
   — no p-values as the primary object.
3. **Asymmetry track** (H2/H4 only, via `src.group_model.asymmetry_specs_from_topology`,
   which derives `(hypothesis, forward_edge, reverse_edge)` pairs directly from
   `edge_topology`'s `_primary`/`_reverse` class labels): `asym = value(forward) −
   value(reverse)` per dyad × film, same formula structure; primary DV is
   `z_vs_surrogate` (scale-free, robust to a child/adult-maturation SNR artefact in
   directionality), `delta_dtf` is a standardized sensitivity model. Intercept > 0
   means caregiver-leading.
4. **One named multiplicity family** (`PRIMARY_FAMILY` in config: the H2 and H4
   *group-effect* contrasts only) gets a mild Benjamini-Hochberg FDR correction
   (`src.group_model.bh_fdr`) on `2·min(P(>0), P(<0))` as an analog p-value,
   reported as a supplementary column, never as the primary inference. The
   exploratory edges receive no correction — they are reported as
   hypothesis-generating.
5. **Convergence gate** (`convergence_row`): max R-hat < `RHAT_MAX` (1.01), min
   bulk/tail ESS > `ESS_MIN` (400), zero divergences, LOO Pareto-k mostly below
   `PARETO_K_MAX` (0.7) — a model failing this is reported as failing, not silently
   retried; per-model diagnostics feed `stage06_diagnostics.csv` and the gate's
   pass/fail badges.
6. **Pooled cross-edge model** (`FIT_POOLED_MODEL`, default on): a `(1|edge)`
   partial-pooling companion across the 6 emphasis edges, a shrinkage cross-check
   alongside (not replacing) the per-edge results.
7. **Cross-edge localization (exploratory, uncorrected)**: when the pooled model
   shows a credible film × group interaction that the primary per-edge H2 model does
   not, three richer hierarchies (`M1`: interaction varies by edge; `M2`: the whole
   film×group surface varies by edge; `M3`: edge as a fixed, un-shrunk factor) are
   compared by LOO (`az.compare`) against the fixed-interaction pooled model to ask
   whether that interaction is genuinely localized to specific edges or better read
   as one shared effect — flagged throughout as hypothesis-generating, never folded
   into the FDR-controlled `PRIMARY_FAMILY`.

Two age/IAF-distance covariate sensitivity toggles (`ADD_AGE_COVARIATE`,
`ADD_IAF_COVARIATE`, reading `iaf_metrics.csv` from the spectral pipeline) are
available but off by default (baseline model is primary).

Output: `stage06_contrasts.csv`, `stage06_diagnostics.csv`,
`stage06_primary_summary.csv` (BH-FDR-adjusted primary family + asymmetry summary),
per-model fitted `idata` (`models/*.nc`), forest/`pp_check`/Pareto-k figures, and
`group_model_gate.html`.

---

## 10. Known limitations and deliberate design choices

- **Order selection stays 2-D** (`mvar_criterion` does not accept a windowed 3-D
  stack); `p_used` is therefore chosen once per case on the global signal and reused
  for the windowed fit, not re-optimized per window.
- **Fixed common order in Stage 5** (`COMMON_MODEL_ORDER`, default 4) intentionally
  ignores each case's own Stage 3 `p_used`, to avoid mixing MVAR orders inside one
  group-level dependent variable — the accepted cost is that Stage 4's per-case
  `.npz` cubes are *not* directly reusable as Stage 5's real-value input; Stage 5
  recomputes.
- **`(1|dyad_id)` only, not `(1|dyad_id)+(1|child_id)+(1|caregiver_id)`**: the
  original plan's three-term formula is not identifiable in this design, since child
  and caregiver ids are 1:1 with dyad (no one appears in more than one dyad) — the
  three grouping factors would partition the rows identically. This correction is
  flagged in the Stage 6 gate for the project owner to ratify.
- **Pooled ("film") null is group-invariant by construction**, so it cancels out of
  every TD-vs-ASD contrast including the interaction — the `within_group` scope
  exists specifically to close that gap, and needs ≥ 2 dyads per (film × group) cell
  or it is asserted infeasible rather than silently skipped.
- **H4's sign is never assumed**: every H4 group-effect report shows both
  `P(ASD>TD)` and `P(ASD<TD)`, per the literature caveat that negative interpersonal
  HRV synchrony can be adaptive.
- **`src.mtmvar.compute_and_plot_mvar` is a known-broken high-level convenience
  wrapper** (bad internal import) — this pipeline never calls it; it always uses the
  lower-level `ar_coeff`/`dtf_estimator`/`mvar_plot` functions directly.
- **Estimator swap-in point**: `src.connectivity.Granger_estimator` is the one stable
  interface every stage from 4 onward calls; a regularized/Bayesian MVAR core could
  replace its internals without changing any caller, but is not currently
  implemented — the windowed-ACF-averaged classical MVAR fit is the production
  estimator, not a placeholder.
- **D1's ROI-channel union coverage check** (Stage 2) evaluates `roi_ok` against the
  union of every ROI node's channels for both roles; an asymmetric topology (e.g. one
  role needing a channel the other role does not) is therefore checked slightly more
  strictly than necessary — a documented simplification, not a bug.
