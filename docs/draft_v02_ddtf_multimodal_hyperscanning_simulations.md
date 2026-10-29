# Directed coupling between people across modalities from amplitude envelopes: a conditioned Granger framework (dDTF) for hyperscanning, validated by simulation

*Working title · DRAFT v0.2 (simulation-based methods paper, target: NeuroImage) · 2026-10-25*

**Authors:** Jarosław Żygierewicz [+ co-authors, TBD]
**Affiliation:** University of Warsaw [TBD]
**Funding:** SYNCC-IN, Horizon Europe, agreement No 101159414.

> Conventions: `[FIG n]` = figure placeholder with a description. `[TBD]` = number or decision to fill. `[REF]` = citation to verify. **`[RESULT]` = a simulation result that has not been produced yet; no number or conclusion in such a place may be written before the run.** Each simulation is marked *existing* (script already in the repository) or *planned*. Open points for the author are at the end.

---

## Abstract

Hyperscanning studies usually measure one modality at a time and describe coupling between people with symmetric measures such as coherence or phase-locking. They therefore cannot say *who leads whom*, or whether *one person's autonomic state predicts another person's cortical state*. We describe and validate a framework that can. EEG rhythm amplitude envelopes and inter-beat-interval (IBI) series, which share a frequency range of about 0.2–1 Hz, enter one low-rate multivariate autoregressive (MVAR) model. Directed influence is quantified by the direct directed transfer function (dDTF), which is frequency-resolved and conditioned on all other channels. Because apparent direction can arise from differences in signal smoothness and from a shared stimulus, each edge is compared with stimulus-matched surrogate dyads. We evaluate the framework entirely on simulated data with known structure: recovery and direction of planted edges, false-positive rate and power of the surrogate test, effect of a common cause, of smoothness, of a shared stimulus (including the case in which groups respond differently to it), sensitivity to estimator, model order and segment length, carrier-frequency mismatch on both the EEG and cardiac sides, and comparison with simpler alternatives. We give practical recommendations and state clearly what the method can and cannot claim: slow, directed, conditioned coupling of engagement, and not fast phase coupling or mechanism. Application to experimental data and individualized frequency-band selection are reported separately.

**Keywords:** hyperscanning; Granger causality; directed transfer function; EEG amplitude envelope; heart rate variability; simulation; surrogate data; multimodal

---

## 1. Introduction

**1.1 What the field measures today.** Hyperscanning has shown that brain activity of interacting people becomes statistically related ("inter-brain synchrony"), and a parallel literature shows the same for heart rhythms ("physiological synchrony"). Most of these measures are *symmetric*: coherence, phase-locking and cross-correlation say that two signals are related, not which one leads. Studies that record several modalities usually analyse each one separately and then compare the results. In mother–child fNIRS–ECG data, for example, neural and cardiac synchrony turned out to be dissociated [Reindl et al., 2022].

**1.2 What is missing.** Where direction has been examined, it is not simple: parent–infant EEG studies often find bidirectional influence [Leong et al., 2017; Marriott Haresign et al., 2022], and cardiac studies report parent-leads or child-leads depending on context [Davis et al., 2018; Capraz et al., 2023]. To our knowledge no study has put *one person's heart and the other person's brain* into a single directed model. That edge is of obvious interest, for instance for co-regulation in caregiver–child dyads, but it cannot be posed with symmetric, single-modality measures.

**1.3 Three obstacles.** (i) *Timescale.* EEG rhythms (~8–12 Hz) and respiration-linked cardiac variability (~0.15–1 Hz) are two orders of magnitude apart and cannot be co-modelled as raw oscillations. (ii) *Frequency mismatch between people.* The EEG rhythms have different centrak frequencies in different people, notably lower in young children than in adults, which biases any phase-based measure. (iii) *Spurious direction.* Apparent direction can arise from properties of the signals (one is smoother than the other) and from common input (i.e., both people watch the same film), without any interaction.

**1.4 What we propose and how we evaluate it.** We combine amplitude envelopes, a joint conditioned MVAR with dDTF, and surrogate-based inference (Fig. 1). A method of this kind should be judged on whether it recovers what is there, does not invent what is not, and fails in understood ways. We therefore evaluate it here only on simulated data, where the truth is known, and we set the simulations up as a pre-specified protocol (Section 5). Experimental data are left to companion papers: the application to caregiver–child dyads, and the selection of individual frequency bands from the EEG spectrum [companion papers, in preparation].

**1.5 Contributions.** (1) A complete, reproducible description of the pipeline. (2) A statement of what quantity it estimates and how to interpret it. (3) A simulation-based assessment of error rates, power and robustness. (4) Practical recommendations on segment length, model order, null distribution and reporting.

> **[FIG 1 — Pipeline overview.]** (A) individual-band envelope (EEG) and resampled IBI → (B) common-rate design matrix (4 channels) → (C) windowed-ACF MVAR → dDTF → (D) real dyads vs. surrogate dyads → Δ, z → (E) group-level readout. EEG stream teal, cardiac stream amber.

---

## 2. What the framework estimates

### 2.1 Directed predictive influence, conditioned

Granger causality asks whether the past of channel *j* improves the prediction of channel *i* beyond what the past of *i* and of the other channels already provides. We use the word "influence" only in this predictive sense. Two properties matter.

*Direction.* Symmetric measures cannot tell parent→child from child→parent. Granger-type measures can, and their frequency-resolved members (DTF, PDC) show at which frequencies the influence acts.

*Conditioning.* When all channels are in one model, each directed edge is the unique contribution of one channel after the past of all others is accounted for. With four channels (child brain, caregiver brain, child heart, caregiver heart), the inter-brain edge is estimated net of shared autonomic dynamics and the heart–heart edges net of the brains. Two separate two-variable models cannot do this, and cross-modal edges exist only in the joint model. This holds only for channels that are in the model: an unmeasured common driver can still produce an edge (Section 8.4).

### 2.2 What the EEG variable represents

The EEG variable is the slow fluctuation of the *power* of a person's rhythm (time scale ~1–2 s), not the oscillation itself. We read it as slow coordination of cortical *engagement*, not millisecond phase coupling. This interpretive commitment applies to every result in the paper. The band is individual: it is centred on each person's own rhythm. In the simulations the centre frequency and band half-width are simulation parameters. For real data they are obtained as described in [companion paper].

### 2.3 What the cardiac variable represents

In the ~0.2–1 Hz range, the interpolated IBI series is dominated by respiratory sinus arrhythmia (RSA), the respiration-paced modulation of heart rate. The raw IBI therefore contains the RSA *oscillation* itself (the carrier), a first-order quantity. The EEG variable is an envelope, a second-order quantity. The two are compatible in frequency but not in order. This has a concrete consequence that we test by simulation (S10): a *linear* MVAR on raw IBI can detect coupling between the IBI oscillation and the EEG envelope (for example breathing-paced modulation of EEG power), but not coupling that exists only between the *amplitudes* of the two. For the latter the cardiac variable would need to be the RSA amplitude, which is slower and needs longer segments (Table 1).

---

## 3. Methods

### 3.1 Signals

**EEG envelope.** For each person we band-pass the ROI signal in the individual band (zero-phase Butterworth) and take the magnitude of the analytic signal:

$$
e(t) \;=\; \left| \, x_{\text{band}}(t) + i\,\mathcal{H}\{x_{\text{band}}\}(t) \right|,
\tag{1}
$$

where $\mathcal{H}$ is the Hilbert transform. *e(t)* is the instantaneous amplitude of the rhythm. It follows slow changes in rhythm power and contains no carrier oscillation. For several channels the envelopes are computed per channel and averaged. The envelope is computed on the whole continuous recording and cut into segments afterwards, so that filter and Hilbert edge effects fall outside the analysed data.

**Why an envelope and not phase.** Phase-based measures compare the phases of two oscillations and assume that they oscillate at the same frequency. If a child's rhythm is at 8 Hz and an adult's at 11 Hz, the phase difference drifts systematically, and phase-locking or phase-based DTF is biased even without coupling. Envelopes compare slow power changes and do not depend on the carrier frequency. The price is that phase information is discarded (Section 8.4).

**Cardiac signal.** The IBI series is interpolated onto a regular grid and only down-sampled; it is not band-pass filtered and no envelope is taken.

**Common rate and filter.** All channels are down-sampled to 2.5 Hz (Nyquist 1.25 Hz) and pass through one identical zero-phase band-pass (0.05–1.0 Hz, 2nd-order Butterworth), so group delay is the same for all variables. Segments of ~60 s give ~150 samples per channel.

> **[FIG 2 — Making the signals compatible.]** (A) Log-frequency axis: EEG rhythm (~8–12 Hz), RSA (~0.15–1 Hz), envelope (~0.2–1 Hz). (B) Raw signal → band-passed signal → envelope. Use the existing `fig_envelope_rationale_*` figure or a simulated equivalent. (C) PSDs of the four design variables on the common grid (simulated dyad; z-scored for display).

**Table 1. Three ways to represent the cardiac signal next to an EEG envelope.**

| | A (default) | A′ | B |
|---|---|---|---|
| Cardiac signal | raw IBI (RSA carrier) | RSA amplitude | RSA amplitude |
| EEG signal | fast envelope | fast envelope | slower, band-matched envelope |
| Matched in | frequency (~0.2–1 Hz) | order (both second-order) | order and frequency (~0.01–0.1 Hz) |
| Can detect | linear IBI-oscillation ↔ envelope coupling | amplitude ↔ amplitude coupling | amplitude ↔ amplitude coupling |
| Sample requirement | ~150 samples/min | similar | ~30 samples/min: needs minutes-long segments |

Option A is the default because it is feasible for ~60 s segments. Simulation S10 compares A and A′ on data with known coupling type.

### 3.2 The joint model

For each dyad and segment we form a vector of *k* variables (default *k* = 4: child ROI envelope, caregiver ROI envelope, child IBI, caregiver IBI), z-scored over time, and fit an MVAR model of order *p*:

$$
\mathbf{x}(t) \;=\; \sum_{r=1}^{p} A(r)\,\mathbf{x}(t-r) + \boldsymbol{\varepsilon}(t),
\tag{2}
$$

where $A(r)$ is a $k\times k$ matrix; $A_{ij}(r)$ expresses how channel *j* at lag *r* contributes to the present value of channel *i*, given all other channels. $\boldsymbol\varepsilon$ is innovation noise with covariance Σ. Nodes and edges of interest are declared in one configuration file, so the same code supports other node sets.

### 3.3 Short, non-stationary segments: windowed autocovariance averaging

A 60 s segment holds ~150 samples, and the cardiac signal is non-stationary within it. We therefore cut the segment into short overlapping windows (10 s = 25 samples, 50% overlap, ≈11 windows), remove the linear trend in each window, treat each window as one *realization*, and estimate the model from the autocovariance averaged across windows [Kamiński et al., 2001]:

$$
\bar R(\tau) \;=\; \frac{1}{W}\sum_{w=1}^{W} R_w(\tau), \qquad \sum_{r=1}^{p} A(r)\,\bar R(s-r) = \bar R(s),\quad s=1,\dots,p,
\tag{3}
$$

where $R_w(\tau)$ is the lag-τ autocovariance matrix within window *w*. The Yule–Walker equations on the right are solved once for $\bar R$. Windows are not stitched together, so no discontinuities are introduced. Z-scoring is done once per segment, before windowing, so that variance drift across windows is preserved. Model quality is checked from the eigenvalues of the companion matrix (all moduli < 1: stable) and from the whiteness of within-window residuals.

### 3.4 Model order

The order is selected by information criteria on the global signal, minimized over *p* = 1…*p*<sub>max</sub>:

$$
\mathrm{AIC}=\ln|\Sigma|+\frac{2pk^2}{n},\quad
\mathrm{HQ}=\ln|\Sigma|+\frac{2\ln(\ln n)\,pk^2}{n},\quad
\mathrm{SC}=\ln|\Sigma|+\frac{\ln(n)\,pk^2}{n},
\tag{4}
$$

with Σ the residual covariance, *k* the number of channels and *n* the number of samples. Schwarz's criterion (SC) is the most conservative and is the default. With ~150 samples and *k*²*p* coefficients, parsimony is essential. For comparisons between dyads, groups or between real and surrogate data, a *common* fixed order is used so that the outcome does not depend on order differences (S8 examines the consequences).

### 3.5 Directed measures

In the frequency domain, Eq. (2) is a linear filter. With $A(f)=I-\sum_r A(r)e^{-i2\pi f r/f_s}$,

$$
H(f) \;=\; A(f)^{-1}, \qquad S(f) = H(f)\,\Sigma\,H(f)^{\mathsf H}.
\tag{5}
$$

$H_{ij}(f)$ describes the transmission from channel *j* to channel *i* at frequency *f*; $S(f)$ is the spectral matrix.

The **directed transfer function (DTF)** [Kamiński & Blinowska, 1991] is the share of the signal at *i* that comes from *j*:

$$
\mathrm{DTF}_{ij}(f) \;=\; \frac{|H_{ij}(f)|^2}{\sum_{m=1}^{k} |H_{im}(f)|^2}.
\tag{6}
$$

It includes indirect paths (*j* → *m* → *i*). The **full-frequency DTF (ffDTF)** normalizes by the inflow summed over all frequencies, making magnitudes comparable across frequencies [Korzeniewska et al., 2003]:

$$
\mathrm{ffDTF}_{ij}(f) \;=\; \frac{|H_{ij}(f)|^2}{\sum_{f'}\sum_{m=1}^{k} |H_{im}(f')|^2}.
\tag{7}
$$

The **direct DTF (dDTF)** multiplies ffDTF by the magnitude of the *partial coherence* $\kappa_{ij}(f)$, which expresses the relation between *i* and *j* after removing the influence of all other channels (computed from minors of $S(f)$):

$$
\mathrm{dDTF}_{ij}(f) \;=\; \mathrm{ffDTF}_{ij}(f)\cdot|\kappa_{ij}(f)|.
\tag{8}
$$

ffDTF says "there is a transmission path from *j* to *i*", partial coherence says "*i* and *j* are directly related once the others are accounted for". The product is large only when both hold, so dDTF suppresses edges that are only indirect. It is the default. Generalized partial directed coherence (GPDC), which normalizes by each channel's noise variance [Baccalá & Sameshima, 2007], is the alternative we compare against.

For each dyad × segment × directed edge, dDTF is averaged over the *coupling band*, the range in which the signals share power (default: [TBD, set from the simulated spectra]), and optionally Box–Cox transformed (λ = 0.25) for distributional reasons. Derived quantities (Section 3.6) are always kept *signed*.

### 3.6 Inference: film-matched surrogate dyads

**The surrogate.** A surrogate dyad combines the child variables from one dyad with the caregiver variables from a *different* dyad that saw the *same stimulus*. Each channel keeps its own smoothness, its own spectrum and its response to the stimulus, while any real interaction between the pair is destroyed. For a stimulus with *N* dyads we use all *N*(*N*−1) ordered mismatched pairs, estimate dDTF on each with the same estimator, order and windows, and discard unstable fits (the number is reported). The real dyad's edge *e* is compared with this null:

$$
\Delta_{e} \;=\; \mathrm{dDTF}^{\text{real}}_{e} - \operatorname{median}\big(\mathrm{dDTF}^{\text{null}}_{e}\big),\qquad
z_{e} \;=\; \frac{\Delta_{e}}{1.4826\cdot\mathrm{MAD}\big(\mathrm{dDTF}^{\text{null}}_{e}\big)} .
\tag{9}
$$

Median and median absolute deviation (MAD; the factor rescales it to a standard deviation under normality) are used so that a few extreme surrogates cannot dominate. Δ keeps the units of the measure, *z* is scale-free.

**Group comparisons need a within-group null.** The pooled null is one number per stimulus and edge. Subtracting it shifts every dyad by the same amount, so it cancels exactly from any between-group contrast. It therefore cannot exclude that two groups respond *differently* to the same stimulus. For group tests the null is built *within group* (surrogates formed only from members of the same group; at least two dyads per stimulus × group cell). Section 6 (S7) tests this argument.

**Two properties of the null that need checking.** (i) The real dyad's members also appear in surrogate pairs, so real value and null are not independent; (ii) *N* varies between studies. Both affect the calibration of the test, and are examined in S3.

**Direction index.** For a pair of opposite edges we use $D = (\mathrm{dDTF}_{j\to i}-\mathrm{dDTF}_{i\to j})/(\mathrm{dDTF}_{j\to i}+\mathrm{dDTF}_{i\to j})$, applied to Δ or to the raw value as stated.

> **[FIG 3 — The model and the null.]** (A) Four nodes, 12 directed edges; inter-brain, inter-heart and cross-modal edges in different colours. (B) Overlapping windows on one segment → averaged autocovariance → one MVAR fit. (C) Surrogate dyads: child from dyad *a*, caregiver from dyad *b*, same stimulus; null distribution per edge; real value; Δ and z.

---

## 4. Simulation framework

We use two levels of simulation (Fig. 4). They test different parts of the pipeline.

**Level 1: analysis-rate generator.** Variables are generated directly at the analysis rate (2.5 Hz) from a VAR with a known coupling tensor $C(r)$: $\mathbf{x}(t)=\sum_r C(r)\mathbf{x}(t-r)+\boldsymbol\eta(t)$, with $\boldsymbol\eta\sim\mathcal N(0,\sigma^2 I)$, $\sigma = 1/\mathrm{SNR}$. Edges are specified as (source, target, lag, gain). Each node has a self-term. *Resonant* nodes, used for IBI-like variables, are AR(2) processes,

$$
x(t) = 2r\cos(2\pi f_0/f_s)\,x(t-1) - r^2 x(t-2) + \eta(t),
\tag{10}
$$

whose spectral peak is at $f_0$ and becomes narrower (the signal "smoother") as the pole radius $r\to1$. *Broadband* nodes (small *r*) mimic envelope-like variables. Level 1 tests the estimator: recovery, error rates, power, conditioning, order, window, estimator choice, smoothness, shared stimulus.

**Level 2: signal-level generator.** Each EEG-like node is $x_i(t) = \exp(\ell_i(t))\,c_i(t) + \text{noise}$, where $\boldsymbol\ell(t)$ is a slow log-amplitude VAR with known coupling (the coupled envelopes), and $c_i(t)$ is an independent unit-amplitude narrow-band carrier with centre frequency $f_i$ (white noise band-passed around $f_i$ and divided by its own Hilbert envelope, so that only its phase is random). Observation noise has standard deviation $\mathrm{std}(x)/\mathrm{SNR}$. The signals pass through the same envelope extraction and down-sampling as real data. Level 2 tests the preprocessing: envelope extraction, and the effect of carrier-frequency differences between persons. For cardiac-like nodes the carrier is an RSA oscillation at a person-specific respiratory frequency with slow drift [planned; see S9].

Unless stated otherwise: 150 samples per segment (60 s at 2.5 Hz), *N* = [TBD, e.g. 25] dyads per stimulus, *p* from SC, windows 10 s/50%, coupling band [TBD], 100–500 replications per cell [TBD]. Each replication simulates a complete study (all dyads, all stimuli), so that error rates refer to the whole procedure including the surrogate step.

> **[FIG 4 — The two generators.]** Level 1: coupling graph with resonant and broadband nodes, example spectra. Level 2: log-amplitude VAR → × carrier → noise → envelope; example traces and PSDs.

---

## 5. Simulation protocol

Each simulation is described by a question, a design, outcome measures and an acceptance criterion. **The criteria are proposed here and should be fixed before the simulations are run.** Results are reported without selection of favourable cells.

**Outcome measures.** *Recovery:* the dDTF of a planted edge vs. its reverse and vs. absent edges. *False-positive rate (FPR):* proportion of replications in which an absent edge is declared present (per-dyad rule: *z* > 1.96; group rule: significant group contrast at α = 0.05). *Power:* the same rates when the edge is present. *Direction accuracy:* proportion of replications in which *D* has the correct sign. *Stability:* proportion of stable, white-residual fits. *Calibration:* distribution of *z* under zero coupling (expected centred near 0).

**Table 2. Simulation overview.**

| ID | Question | Level | Status |
|---|---|---|---|
| S1 | Are planted edges recovered, in the right direction? | 1 | existing |
| S2 | Envelope vs. phase when person centre frequencies differ (two ground truths) | 2 | existing (amplitude truth); phase truth planned |
| S3 | Is the surrogate test calibrated under zero coupling? | 1 | planned |
| S4 | How much power depends on coupling strength, segment length, SNR, number of dyads? | 1 | planned |
| S5 | Does conditioning remove common-cause and indirect edges? | 1 | planned |
| S6 | Does smoothness create a spurious direction, and does the null remove it? | 1 | existing (basic); extension planned |
| S7 | Shared stimulus; groups with different stimulus gain; pooled vs. within-group null | 1 | planned |
| S8 | Sensitivity to estimator, order and window; windowed vs. global fit | 1 | partly existing |
| S9 | Carrier-frequency mismatch on the cardiac side | 2 | planned |
| S10 | Which coupling types are visible with raw IBI (A) vs. RSA amplitude (A′)? | 1–2 | planned |
| S11 | Comparison with simpler alternatives | 1–2 | planned |

### S1 — Recovery and orientation *(existing)*

*Question.* With a known directed structure, does the estimator recover the planted edges, and is the source/target convention correct?
*Design.* (a) A single edge 0→1 and zero elsewhere; we check which axis of the output carries source and target (dominant direction must exceed the reverse by a factor ≥ 3). (b) A four-node VAR mimicking the real design (two ROI and two IBI nodes), with a within-modality edge in each modality and one cross-modal edge at lag 2–3 (gains 0.35–0.5, self-gain 0.3, SNR 5).
*Criterion.* Planted edges have larger dDTF than their reverse and than absent edges.
*Result.* `[RESULT: recovery table and Fig. 6A from the final run]`

### S2 — Envelope vs. phase across a centre-frequency gap *(existing for amplitude truth)*

*Question.* How does a difference in centre frequency between persons affect envelope-based and phase-based dDTF?
*Design.* Child frequency 8 Hz; partner frequency 8 + gap, gap 0–9.5 Hz; 180 s; 100 realizations per gap. Each estimate is expressed relative to its own zero-gap value.
*Ground truth 1 (existing): amplitude coupling* (the coupling acts on the log-amplitude). *Caveat:* here the envelope estimator is aligned with the ground truth by construction, so this simulation shows robustness to the frequency gap and not superiority. *Ground truth 2 (planned): phase coupling* between oscillators with detuning equal to the gap (e.g. weakly coupled phase oscillators or resonators). Both ground truths are analysed with both estimators.
*Reading.* This design makes the estimand explicit: the envelope estimator reports amplitude coordination and is blind to phase coupling; phase-based estimators report phase coupling and are degraded by detuning. The paper's claim is limited to the first.
*Criterion.* Envelope-based dDTF stays within 50% of its zero-gap value across the gap range for amplitude truth.
*Result.* `[RESULT: Fig. 6B–C]`

### S3 — Calibration of the surrogate test *(planned)*

*Question.* With zero true coupling, is the proportion of false positives close to the nominal level?
*Design.* Heterogeneous dyads (spectral peak frequency, smoothness and SNR vary across dyads), no coupling. Vary *N* (10, 25, 50), segment length (60, 120, 300 s) and the correlation structure of the null (real dyad's members included or excluded from the pool). Report FPR (per-dyad and group rule) and the distribution of *z*.
*Criterion.* FPR within [0.03, 0.07] for nominal 0.05; median *z* within ±0.2. *[proposed]*
*Result.* `[RESULT: Fig. 7A]`

### S4 — Power *(planned)*

*Question.* What coupling strength, segment length, SNR and *N* are needed to detect an edge and its direction?
*Design.* A planted edge with gain *g* = 0, 0.05, …, 0.5; segment 60/120/300 s; SNR 2–10; *N* 10–50. Within-modality and cross-modal edges separately.
*Outcome.* Power and direction accuracy as surfaces over (gain, length) for each *N*; the minimal design reaching 80% power.
*Result.* `[RESULT: Fig. 7B–C]`

### S5 — Conditioning *(planned)*

*Question.* Does the joint model remove spurious edges produced by a common cause and by indirect paths?
*Design.* (a) Common cause: an autonomic-like node drives both brain nodes (no direct brain–brain edge); (b) cascade: A→B→C without A→C. Compare pairwise (two-variable) models and the four-variable model, and DTF, ffDTF and dDTF.
*Outcome.* FPR of the spurious edge (A→C, brain→brain), and power for the true edges.
*Criterion.* The joint dDTF has a lower spurious-edge rate than pairwise models and than DTF, without a loss of power > [10%].
*Also tested.* The failure case: the common cause is *omitted* from the model; the spurious edge reappears. This is the quantitative form of the limitation in Section 8.4.
*Result.* `[RESULT: Fig. 8A]`

### S6 — Smoothness *(existing basic; extension planned)*

*Question.* Does a difference in smoothness alone create a spurious direction, and does the surrogate null remove it?
*Design (existing).* Two independent AR(2) resonators at 0.3 Hz that differ only in pole radius (0.50 → 0.97), zero coupling, z-scored channels, windowed fit, 6 batches of 16 dyads per grid point; a second condition with a planted rough→smooth edge (30 dyads). dDTF is not scale-invariant: without z-scoring the sign of the artefact reverses, so direction must be read off the null.
*Extension (planned).* Smoothness contrasts spanning the range plausible for envelope-vs-IBI spectra; cardiac and EEG-like nodes in the same model; FPR and direction accuracy as in S3.
*Existing result (zero coupling).* The smoother channel appears as the source and the raw effect grows with the contrast; Δ stays at ≈ 0 with |z| < ~0.5 across the gradient `[RESULT: confirm in final run]`. With planted coupling, the real distribution separates from the null.
*Result.* `[RESULT: Fig. 8B]`

### S7 — Shared stimulus and group-dependent stimulus gain *(planned)*

*Question.* Does a common stimulus inflate coupling, and which null removes it, when groups respond differently?
*Design.* A common input *u*(*t*) (same for all dyads in a stimulus) drives all four nodes with person-specific gain; true interaction is zero. (a) Equal gain in two groups: pooled null should remove the effect. (b) Group B has a different gain: group contrast of Δ with the pooled null; with the within-group null. Vary the gain difference.
*Outcome.* False-positive rate of the group contrast under (a) and (b) for the raw value, the pooled-null Δ and the within-group Δ.
*Criterion.* Raw value: inflated; pooled-null Δ: calibrated under (a), inflated under (b); within-group Δ: calibrated under both. *[hypothesis to be tested, not an expectation to be confirmed]*
*Group readout.* A robust two-group contrast on Δ or *z* (permutation test of the difference in medians); the Bayesian mixed model used in applications is described in the Supplement.
*Result.* `[RESULT: Fig. 8C]`

### S8 — Estimator, order and window *(partly existing)*

*Question.* How much do the choices of estimator (DTF, ffDTF, dDTF, GPDC), order (SC, AIC, fixed *p* = 2–6) and window (10 s/50%, 15 s/50%, 10 s/0%, or none) change recovery, direction accuracy, stability and calibration?
*Design.* S1 and S3 designs crossed with the choices above, with sample-to-parameter ratio reported. Include a global (non-windowed) fit and the effect of non-stationary drift in the resonant nodes (a slow change in $f_0$ and in variance).
*Criterion.* The default configuration is within [10%] of the best configuration on every measure.
*Result.* `[RESULT: Fig. 9]`

### S9 — Carrier-frequency mismatch on the cardiac side *(planned)*

*Question.* The argument for envelopes on the EEG side (persons' carriers differ) also applies in principle to raw IBI, because children and adults breathe at different rates. How large is the effect?
*Design.* Level 2. Cardiac nodes are RSA oscillations with person-specific respiratory frequencies (e.g. 0.25–0.5 Hz) and slow drift, linked by (a) no coupling, (b) a planted IBI→IBI coupling through frequency pulling or amplitude drive. Vary the frequency gap and the width of the coupling band.
*Outcome.* FPR and power of the IBI–IBI edge as functions of the gap and band width; effect on the surrogate null.
*Result.* `[RESULT: Fig. 10A]`

### S10 — Which coupling types are visible with raw IBI (option A) vs. RSA amplitude (A′)? *(planned)*

*Question.* What does a cross-modal edge mean under each operationalization?
*Design.* Level 2. Truths: (a) *linear coupling* — breathing-paced modulation of EEG power by the IBI oscillation; (b) *amplitude-only coupling* — RSA amplitude drives EEG envelope amplitude; (c) none. Each is analysed with raw IBI (A) and RSA amplitude (A′, from the high-frequency band-passed IBI envelope).
*Outcome.* Power and FPR for each truth × option. *[Prediction to test: A detects (a) but not (b); A′ detects (b) but not (a).]*
*Result.* `[RESULT: Fig. 10B; Table 1 to be updated accordingly]`

### S11 — Comparison with simpler alternatives *(planned)*

*Question.* What does the joint dDTF add compared with what a typical hyperscanning analysis would do?
*Competitors.* (i) Pairwise Granger on envelopes; (ii) lagged cross-correlation of envelopes; (iii) magnitude-squared coherence of envelopes; (iv) phase-based measures on raw signals (PLV, phase-based DTF).
*Design.* Run S3 (zero coupling, with a common stimulus), S5 (common cause) and S4 (power) for all methods; each with the surrogate null where applicable.
*Outcome.* FPR, power, direction accuracy and ability to separate direct from indirect coupling.
*Result.* `[RESULT: Fig. 11]`

> **[FIG 5 — Summary of simulations in Table 2 as a graphical overview, optional.]**
> **[FIG 6 — S1 and S2.]** (A) Planted vs. recovered edge strengths; (B) amplitude truth: relative recovered coupling vs. frequency gap for envelope and phase estimators; (C) phase truth, same.
> **[FIG 7 — S3 and S4.]** (A) FPR and *z* distribution; (B–C) power surfaces.
> **[FIG 8 — S5, S6, S7.]** (A) conditioning; (B) smoothness gradient and Δ; (C) pooled vs. within-group null with group-dependent gain.
> **[FIG 9 — S8.]** Heat map: recovery/direction/stability/calibration by estimator × order × window.
> **[FIG 10 — S9 and S10.]** (A) cardiac frequency gap; (B) visible coupling types for A vs. A′.
> **[FIG 11 — S11.]** FPR/power for competing methods.

---

## 6. Results

*To be written after the simulations are run. Each subsection follows S1–S11. Report all conditions, including those in which the method fails.*

---

## 7. Practical recommendations

*To be derived from S3, S4, S8; a proposed layout for a box:*

| Decision | Recommendation | Source |
|---|---|---|
| Minimum segment length | [TBD s] for [TBD] power at gain [TBD] | S4 |
| Minimum number of dyads per stimulus (pooled null) | [TBD] | S3 |
| Minimum per stimulus × group cell (within-group null) | [TBD] (at least 2 for feasibility) | S3, S7 |
| Model order | SC, or fixed *p* = [TBD] if fewer than [TBD] samples per parameter | S8 |
| Window | 10 s/50% unless [TBD] | S8 |
| Estimator | dDTF; GPDC if noise variance differs strongly between channels | S8 |
| Null | pooled for within-group effects; within-group for group contrasts | S7 |
| Reporting | stable-fit rate; Δ and *z*; signed values; null scope | S3 |
| Interpretation | amplitude coordination, not phase coupling; cross-modal edge type depends on option A/A′ | S2, S10 |

---

## 8. Discussion

### 8.1 What the framework gives

It turns descriptive questions ("are the signals related?") into directional and conditional ones: whether the caregiver's cortical engagement predicts the child's more than the reverse, net of heart dynamics; whether autonomic state predicts cortical engagement within a person; and whether one person's autonomic state predicts another person's cortical engagement. Direction is a quantity to be tested, not assumed; the literature does not favour one direction. Frequency resolution allows asking at which time scale the coupling acts (a fast "engagement" time scale with option A, or a slow "control" time scale with option B).

### 8.2 Extensibility

Nodes and edges are declared in a configuration file. Further ROIs, signals such as pupil size or electrodermal activity, or other roles can be added, provided each signal is brought to a common rate and band by a transformation whose meaning is stated. Haemodynamic fNIRS signals are already slow, so no envelope step is needed and commensurability with IBI is more direct. Nothing in the framework is specific to caregiver–child dyads.

### 8.3 What the simulations show about the method's behaviour

`[To be written after results: where it is calibrated, where it fails (smoothness without null; group-dependent stimulus gain with a pooled null; omitted common cause; cardiac carrier mismatch), and the sample sizes needed.]`

### 8.4 Limitations

- *Slow engagement, not fast coupling.* Envelopes carry no phase information (S2). Phase-based analyses answer a different question and are complementary.
- *Predictive influence, not mechanism.* Edges are Granger-predictive. Scalp electrodes are coarse proxies of cortical regions.
- *Omitted drivers.* Conditioning acts only on channels in the model (S5). **Breathing** is the most obvious omitted driver of both IBI series (and possibly of EEG power). A respiration channel is not available in our data; recording it and adding it as a node is the most valuable improvement of the design, and the framework supports it without change.
- *First- vs. second-order variables.* With raw IBI, cross-modal edges reflect linear coupling around breathing frequencies and not coupling of amplitudes (S10).
- *Cardiac carriers differ between persons.* The mismatch argument that motivates envelopes on the EEG side applies in principle to raw IBI (S9).
- *Short segments.* ~150 samples constrain model order; dyad-level estimates are noisy and informative mainly in aggregate (S4).
- *Linear, stationary within window.* The windowed fit reduces but does not remove sensitivity to non-stationarity (S8). Order selection is performed on the global signal.
- *Simulation is not data.* All results hold for the generators used. Real signals contain artefacts (movement, muscle activity), non-Gaussian features and a more complex spectral structure. The generators were chosen to isolate specific failure modes, not to reproduce real data, and the companion paper tests the pipeline on experimental data.
- *Fixed analytic choices.* Band-averaging range and Box–Cox λ are choices; their sensitivity is part of S8.

### 8.5 Outlook

Applications to experimental data (caregiver–child dyads in typically developing and autism groups; the pre-specified predictions and the within-group null) are reported in [companion paper]. The selection of individual frequency bands from the EEG spectrum, on which the envelope depends, is described in [companion paper].

---

## 9. Conclusions

Envelopes make EEG and cardiac signals commensurate in frequency, and a joint conditioned dDTF turns multimodal hyperscanning from "which signals are related?" into "who predicts whom, through which channel, and beyond the stimulus?". The evidence for this rests on simulations with known structure; they show, on pre-specified criteria, where the method is calibrated and where it fails, and they define the data requirements for its use.

---

## Data and code availability

All simulation code, configurations and random seeds will be deposited in a public repository with a DOI `[TBD]`. The pipeline is configuration-driven (`pipeline_config.json`). No experimental data are used.

## Acknowledgements

The work was conducted as part of the SYNCC-IN project funded by the European Union under the Horizon Europe programme (agreement No 101159414). Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the granting authority.

## References (to be verified and completed)

- Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a new concept in neural structure determination. *Biological Cybernetics*, 84, 463–474. `[REF]`
- Baccalá, L. A., & Sameshima, K. (2007). Generalized partial directed coherence. *15th International Conference on Digital Signal Processing*. `[REF]`
- Capraz, Y. Z., Konrad, K., & Reindl, V. (2023). Concurrent and lagged physiological synchrony during mother–child interaction and their relationship to positive affect in 8- to 10-year-old children. *Scientific Reports*, 13, 17744.
- Davis, M., West, K., Bilms, J., Morelen, D., & Suveg, C. (2018). A systematic review of parent–child synchrony: It is more than skin deep. *Developmental Psychobiology*, 60(6), 674–691.
- Feldman, R., Magori-Cohen, R., Galili, G., Singer, M., & Louzoun, Y. (2011). Mother and infant coordinate heart rhythms through episodes of interaction synchrony. *Infant Behavior and Development*, 34(4), 569–577.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37, 424–438. `[REF]`
- Kamiński, M., & Blinowska, K. J. (1991). A new method of the description of the information flow in the brain structures. *Biological Cybernetics*, 65, 203–210. `[REF]`
- Kamiński, M., Ding, M., Truccolo, W. A., & Bressler, S. L. (2001). Evaluating causal relations in neural systems: Granger causality, directed transfer function and statistical assessment of significance. *Biological Cybernetics*, 85, 145–157. `[REF]`
- Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003). Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. *Journal of Neuroscience Methods*, 125, 195–207.
- Leong, V., Byrne, E., Clackson, K., Georgieva, S., Lam, S., & Wass, S. (2017). Speaker gaze increases information coupling between infant and adult brains. *PNAS*, 114(50), 13290–13295.
- Marriott Haresign, I., Phillips, E., Whitehorn, M., Noreika, V., Jones, E. J. H., Leong, V., & Wass, S. V. (2022). Measuring the temporal dynamics of inter-personal neural entrainment in continuous child–adult EEG hyperscanning data. *Developmental Cognitive Neuroscience*, 54, 101093.
- Reindl, V., Gerloff, C., Scharke, W., & Konrad, K. (2022). Multimodal hyperscanning reveals that synchrony of body and mind are distinct in mother–child dyads. *NeuroImage*, 251, 118982.
- `[REF]` Valenza et al. (within-person brain–heart directed coupling); h-DCM inter-brain studies; surrogate-data reviews (e.g. Lancaster et al., 2018, *Physics Reports*); simulation-study reporting guidance (e.g. Morris, White & Crowther, 2019, *Statistics in Medicine*).
- `[REF]` Companion papers: individual frequency bands; application to caregiver–child dyads.

---

## Author notes – delete before submission

1. **Most of the simulation protocol does not exist yet.** In the repository: S1 (Stage 0 Part A), S2 amplitude truth (Stage 0 Part B), S6 basic (Stage 0b), part of S8 (window sensitivity on 1–2 dyads; windowed vs. global fit on real data only). Everything else needs new code. Needed building blocks: an RSA-type cardiac generator with person-specific drifting respiratory frequency (S9, S10), a phase-coupled oscillator generator (S2), a common-stimulus generator with group-dependent gain (S7), a replication harness that simulates a whole study including the surrogate step (S3, S4, S7), and baseline competitors (S11). I can write these when you want.
2. **Fairness of S2.** In the existing generator the coupling acts on log-amplitude, so the envelope estimator is matched to the truth by construction. A reviewer will see this. I framed the existing result as robustness to detuning and added a phase-coupling truth. The statement in the earlier draft that the phase-based estimate "collapses" should not be repeated without this second ground truth.
3. **Amplitude-only coupling is invisible to a linear MVAR on raw IBI.** If the carrier has random phase, the cross-spectrum between a slow envelope and a carrier oscillation is zero, so the raw-IBI cross-modal edge can reflect only linear coupling around breathing frequencies (e.g. breathing-linked modulation of EEG power or of artefacts). This sharpens the earlier "partly respiratory-timing coupling" and is the content of S10. It is stated as a prediction until the simulation is run.
4. **Acceptance criteria** (FPR band, ±0.2 on median *z*, 10% margins, 3× orientation ratio) are my proposals; fix them before running, and ideally register the protocol (e.g. OSF).
5. **Surrogate pool dependence.** The real dyad's members are part of the surrogate pairs. This could slightly affect calibration; S3 includes an "excluded from pool" arm.
6. **Group readout.** I used a simple permutation contrast for group effects in simulations, because fitting the Bayesian mixed model on thousands of replications is expensive. The Bayesian model moves to the Supplement or the companion paper. Decide whether you want one Bayesian demonstration in the main text.
7. **Coupling band.** Documentation says 0.2–1.0 Hz, `pipeline_config.json` says 0.15–0.5 Hz. In this paper it is `[TBD]` and set from the simulated spectra; make sure it matches the companion application paper.
8. **Real-data content removed.** The H2/H4 labels, TD/ASD results, the Incredibles example, ROI/electrode P7 and the 126/126 stability statistic are no longer in this paper. If you want one real-data panel as a feasibility illustration, I would put it in the Supplement, clearly labelled.
9. **Respiration** is mentioned only as a limitation and an improvement (Section 8.4), as agreed.
10. **NeuroImage formalities** (to check on the journal's site): data/code availability with DOI, ethics statement (not needed if no human data are used, but state it), CRediT, competing interests, AI-use declaration.
