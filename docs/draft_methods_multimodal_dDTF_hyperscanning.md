# Directed brain–heart coupling between people from amplitude envelopes: individualized frequency bands and conditioned Granger analysis (dDTF) for multimodal hyperscanning

*Working title · DRAFT v0.1 · 2026-10-25*

**Authors:** Jarosław Żygierewicz [+ SYNCC-IN co-authors, TBD]
**Affiliation:** University of Warsaw [TBD]
**Funding:** SYNCC-IN, Horizon Europe, agreement No 101159414.

> Conventions in this draft: `[FIG n]` marks a figure placeholder with a description of what it should show. `[TBD]` marks a number or decision to be filled in. `[REF]` marks a citation to be checked. A list of open points for the author is at the end (*Author notes – delete before submission*).

---

## Abstract

Hyperscanning studies usually measure one modality at a time and describe coupling between people with symmetric measures such as coherence or phase-locking. Two questions are therefore hard to ask: *who leads whom*, and *whether one person's autonomic state predicts the other person's cortical state*. We describe a method that addresses both. (1) Each participant's EEG rhythm is located individually, by separating periodic peaks from the aperiodic 1/f background (specparam) and grouping peaks across a region of interest into slow and fast rhythms. (2) The amplitude envelope of this individual rhythm is used instead of the oscillation itself. The envelope fluctuates at about 0.2–1 Hz, the same range as the respiration-paced component of the inter-beat-interval (IBI) series, so EEG and cardiac signals can enter one low-rate multivariate autoregressive (MVAR) model. (3) Directed influence is quantified with the direct directed transfer function (dDTF), which is frequency-resolved and conditioned on all other channels. (4) Because direction estimates can be biased by differences in signal smoothness and by a shared stimulus, every edge is judged against film-matched surrogate dyads. We validate the estimator on synthetic data with known coupling, and we show the approach on caregiver–child dyads watching animated films. We state what the method can claim (slow, directed, conditioned coupling of engagement) and what it cannot (fast phase coupling, mechanism).

**Keywords:** hyperscanning; Granger causality; directed transfer function; EEG amplitude envelope; heart rate variability; individual alpha frequency; specparam; caregiver–child

---

## 1. Introduction

**1.1 What the field measures today.** Hyperscanning has established that brain activity of interacting people becomes statistically related ("inter-brain synchrony"), and a parallel literature shows the same for heart rhythms ("physiological synchrony"). Most of these measures are *symmetric*: coherence, phase-locking value and cross-correlation say that two signals are related, not which one leads. Studies that record several modalities typically analyse each one separately and then correlate the results; in mother–child fNIRS–ECG data, for example, neural and cardiac synchrony turned out to be dissociated [Reindl et al., 2022].

**1.2 What is missing.** Where direction has been examined, the picture is not simple. Parent–infant EEG studies often find bidirectional influence in the theta/alpha range [Leong et al., 2017; Marriott Haresign et al., 2022], and cardiac studies report parent-leads or child-leads depending on context [Davis et al., 2018; Capraz et al., 2023]. We found no study that places *one person's heart and the other person's brain* in a single directed model. That edge — autonomic state of person A predicting cortical engagement of person B — is of obvious interest for developmental and clinical work (e.g. co-regulation in caregiver–child dyads), and it cannot be posed with symmetric, single-modality measures.

**1.3 Why this is hard.** Two technical obstacles stand in the way. First, EEG rhythms (about 8–12 Hz) and respiration-linked cardiac variability (about 0.15–1 Hz) differ by roughly two orders of magnitude in frequency, so they cannot be modelled together as raw oscillations. Second, the EEG rhythm itself sits at different frequencies in different people — lower and more variable in young children than in adults — which biases any measure that depends on phase. A third, statistical obstacle applies to every directed estimate: apparent direction can arise from properties of the signals (e.g. one is smoother than the other) and from common input (both people watch the same film), without any interaction between them.

**1.4 What we propose.** We combine four steps into one pipeline (Fig. 1): individualized frequency bands, amplitude envelopes, a joint conditioned MVAR with dDTF, and surrogate-based inference. Our aim in this paper is to describe the method in enough detail to be reproduced, to justify each design choice, and to say plainly what each choice costs. The analysis is exploratory in application; the method itself is anchored in synthetic ground truth.

> **[FIG 1 — Pipeline overview.]** Flow diagram: (A) cleaned EEG → PSD → specparam → individual band; (B) band-pass + Hilbert → envelope; IBI → resample; (C) common-rate design matrix (4 channels); (D) windowed-ACF MVAR → dDTF; (E) real dyads vs. surrogate dyads → Δ, z; (F) group model. Colour-code EEG (teal) and cardiac (amber) streams.

---

## 2. Conceptual framework

### 2.1 Why a directed, conditioned measure

Granger causality asks whether the past of channel *j* improves prediction of channel *i* beyond what the past of *i* itself (and of the other channels) already provides. Two properties matter here.

*Direction.* Symmetric measures cannot distinguish parent→child from child→parent. Granger-type measures can, and the frequency-resolved members of the family (DTF, PDC) say at which frequencies the influence acts.

*Conditioning.* When all channels are modelled together, each directed edge is the *unique* predictive contribution of one channel after accounting for the past of all the others. With four channels (child brain, caregiver brain, child heart, caregiver heart), the inter-brain edge is estimated while controlling for shared autonomic dynamics, and the heart–heart edges while controlling for the brains. Two separate two-variable models cannot do this, and a cross-modal edge exists only in the joint model.

We use "influence" in the Granger sense — *predictive* precedence within the measured set. It does not identify a physical mechanism, and unmeasured common drivers (breathing is the obvious one, see §7.4) can still produce an edge.

### 2.2 What the signal represents

The central interpretive commitment of the method is that **the EEG variable is the slow fluctuation of the power of an individual rhythm (~1–2 s scale), not the fast oscillation itself.** We read it as slow coordination of cortical *engagement* (e.g. alpha-band power changes related to attention), not as millisecond phase coupling. All conclusions in this paper are limited to that level.

---

## 3. Individualized frequency bands

### 3.1 Rationale

A fixed definition such as "alpha = 8–13 Hz" is not appropriate when the cohort is 3–6 years old: dominant rhythms sit lower and vary more between children, and they differ between children and their adult caregivers (roughly 8–10 Hz vs. 10–12 Hz). A fixed filter would either miss a child's rhythm or admit broadband and aperiodic power instead of a rhythm. Because the next step takes the *envelope* of the filtered signal, the filter must be centred on the person's own rhythm so that the envelope tracks one construct — the power of that person's dominant oscillation — in everyone.

Two features of our approach follow from this: (i) rhythms are detected *relative to the aperiodic background* (so age-related differences in the 1/f slope are not mistaken for oscillatory differences), and (ii) the band is derived *per participant and per region*, from peaks that are consistent across neighbouring channels.

### 3.2 Step 1: Spectra

For each participant and film we trim the cleaned EEG to the film window and compute a multitaper power spectral density (PSD; 1–30 Hz; half-bandwidth smoothing 2 Hz) at each of 19 channels (10–20 montage). PSDs are averaged across the three films for each participant, giving one spectrum per channel. Averaging increases the signal-to-noise ratio of peak detection; whether it is justified is checked afterwards (§3.7).

### 3.3 Step 2: Separating rhythm from background (specparam)

We fit each PSD with the specparam model [Donoghue et al., 2020], which describes log-power as an aperiodic component plus a sum of Gaussian peaks:

$$
\log_{10} P(f) \;=\; \underbrace{b - \chi \log_{10} f}_{\text{aperiodic (1/f)}} \;+\; \underbrace{\sum_{n=1}^{N} a_n \exp\!\left(-\frac{(f-c_n)^2}{2 w_n^2}\right)}_{\text{periodic peaks}}
\tag{1}
$$

Here *b* is the offset and χ the exponent of the 1/f background, and each peak *n* has a centre frequency *c<sub>n</sub>*, a height *a<sub>n</sub>* above the background (the "power" of the peak) and a width *w<sub>n</sub>*. Only the peak parameters (*c<sub>n</sub>*, *a<sub>n</sub>*, bandwidth) are used downstream. Settings are given in Table 1.

**Table 1. specparam settings.**

| Setting | Channel-level fit | ROI-averaged refit |
|---|---|---|
| Fit range | 3–15 Hz | 3–14 Hz |
| Peak width limits | 1–3 Hz | 1–3 Hz |
| Max. number of peaks | 4 | 4 |
| Min. peak height (log power above fit) | 0.05 | 0.10 |
| Aperiodic mode | fixed (no knee) | fixed (no knee) |

The stricter threshold in the ROI-averaged refit reflects the higher signal-to-noise ratio after averaging: a lower threshold would admit more spurious peaks.

> **[FIG 2 — Spectral parameterization of one participant.]** Panel A: PSD of one child and one caregiver at one channel with the aperiodic fit and Gaussian peaks. Panel B: scalp map of peak presence/frequency across the 19 channels.

### 3.4 Step 3: Where and at what frequency are peaks found?

Before defining any band we inventory the peaks: for each channel and frequency bin (3–5, 5–7, 7–9, 9–11, 11–13 Hz), the fraction of participants with a detected peak (*prevalence*), by role and diagnostic group. A bimodality check over sensorimotor, parietal, lateral-temporal and temporo-parietal regions asks whether a region shows two distinct rhythms across the cohort. This motivates a two-band model (slow and fast).

### 3.5 Step 4: Grouping peaks into one slow and one fast rhythm per region

Regions of interest (ROIs) are theory-driven scalp neighbourhoods (e.g. parietal = P3, Pz, P4; temporo-parietal = P7, P8, used as a proxy of the temporo-parietal junction). For each participant and ROI we pool the peaks detected at the ROI's channels within 3–13 Hz and assign them to **at most two** rhythms by a seeded, power-weighted *k*-means with *k* = 2:

1. **Seeds.** One seed is taken from the slow range (3–7.5 Hz) and one from the fast range (7.5–13 Hz). In each range the seed is the peak whose frequency is *closest to the range midpoint* (ties: higher power, then lower frequency) — not the strongest peak. This is deliberate: a strong peak at the 7.5 Hz border is plausibly part of the fast rhythm, and a "strongest-peak" seed would pull it into the slow cluster.
2. **Assignment and update.** Each peak joins the nearest centre (in Hz). Each centre is then recomputed as the power-weighted mean of its members,

$$
\bar c_k \;=\; \frac{\sum_{n\in C_k} a_n\, c_n}{\sum_{n\in C_k} a_n},
\tag{2}
$$

so louder (more reliable) peaks count more. Steps repeat until nothing changes (max. 50 iterations).

3. **Single-rhythm rule.** If only one range contains a candidate, if the partition collapses, or if the two centres end up closer than 1.5 Hz, the participant is given a *single* rhythm, labelled slow or fast according to the 7.5 Hz boundary. The 1.5 Hz minimum gap reflects the resolution of the spectral model: peaks that close together cannot be told apart from one broad peak, given peak widths of 1–3 Hz.
4. **Band edges.** The band is $[\bar c_k - h_k,\; \bar c_k + h_k]$, with the half-width $h_k$ large enough to contain every merged peak together with its own width:

$$
h_k \;=\; \max_{n\in C_k}\Big( |c_n-\bar c_k| + \tfrac{1}{2}\,\mathrm{BW}_n \Big),
\tag{3}
$$

where BW<sub>n</sub> is the peak bandwidth reported by specparam. The band thus adapts to the width of each person's rhythm instead of using a fixed ± multiple of the centre frequency.

This procedure replaced an earlier approach (greedy clustering with 1 Hz tolerance, then keeping the two strongest peaks). Greedy clustering could split one true oscillation into two adjacent peaks, and the "two strongest" rule would then discard one as if it were noise.

The output is one row per participant and ROI with the slow and fast centre and bandwidth, and a label (`two_rhythms`, `single_slow`, `single_fast`, `no_peaks`). We also derive individual alpha frequency (IAF; parietal ROI, with sensorimotor as fallback) and the child–caregiver distance in IAF, which can be used as a covariate.

> **[FIG 3 — Peak grouping and individual bands.]** Panel A: peaks from one ROI plotted as frequency vs. power, coloured by cluster, with the resulting band window. Panel B: distribution of fast-rhythm centre frequency in children and caregivers (shows the group difference that motivates individualization). Panel C: examples of the `two_rhythms` vs `single_fast` cases.

### 3.6 Step 5–6: Is the ROI a trustworthy unit?

Averaging channels is only useful if the rhythm is really present in the ROI. We therefore require, separately for the slow (3–7 Hz) and fast (7–14 Hz) windows, that the mean peak-detection prevalence across the ROI's channels is at least 50% in each subgroup (role × group). Next, we re-fit specparam on the ROI-averaged PSD and test whether peaks seen at single channels *survive* averaging, within 1 Hz. A peak present at only one channel, or whose phase relations differ between channels, can vanish on averaging; ROI–participant combinations with survival below 60% are flagged as a caveat. Single-channel ROIs (e.g. P7 alone) need no averaging step.

### 3.7 Step 7: Is one band per participant justified across films?

Bands are estimated on the film-averaged spectrum. To check that this is acceptable, we re-fit specparam separately for each film and compare the centre frequency with the film-averaged one, separately for the slow and the fast band (a stable fast rhythm does not imply a stable slow one). A shift larger than 2 Hz is flagged as unstable (Bland–Altman-style agreement plots and a stability table).

### 3.8 What this step does and does not guarantee

The thresholds (peak height, 50% prevalence, 1.5 Hz gap, 1 Hz tolerance, 60% survival, 2 Hz stability) are fixed pragmatic choices recorded in the analysis report, not derived from the data. The two-band model is a simplification: a participant with three genuine rhythms is still reduced to two. ROI definitions are scalp neighbourhoods and do not claim anatomical source specificity. Participants for whom no rhythm is found in the ROI have no band, and the dyad–film cell cannot be analysed `[TBD: confirm handling]`.

---

## 4. Making EEG and cardiac signals commensurate

### 4.1 The timescale problem

An MVAR model relates channels through their past values at a *single* sampling rate. A 10 Hz oscillation and a 0.3 Hz oscillation cannot be sensibly co-modelled: at a rate adequate for the first, the second changes almost nothing from sample to sample; at a rate adequate for the second, the first is aliased.

> **[FIG 4 — Two orders of magnitude apart.]** Log-frequency axis showing the EEG rhythm range (~8–12 Hz), the RSA range (~0.15–1 Hz) and the envelope range (~0.2–1 Hz), with example PSDs from one dyad.

### 4.2 EEG: the amplitude envelope of the individual rhythm

For each participant we band-pass the ROI signal in the individual band (zero-phase Butterworth, order 4) and take the magnitude of the analytic signal:

$$
e(t) \;=\; \left| \, x_{\text{band}}(t) + i\,\mathcal{H}\{x_{\text{band}}\}(t) \right|,
\tag{4}
$$

where $\mathcal{H}$ is the Hilbert transform. *e(t)* is the instantaneous amplitude of the rhythm; it follows the slow rise and fall of rhythm power and contains no carrier oscillation. For multi-channel ROIs we compute the envelope at each channel and then average. The envelope is computed on the whole continuous recording and cut into film segments *afterwards*, so that filter and Hilbert edge effects fall in the discarded margins and not in the analysed data.

**Why envelope and not phase.** Phase-based measures compare the phase of two oscillations and assume that they oscillate at the same frequency. When the child's rhythm is at 9 Hz and the adult's at 11 Hz, the phase difference drifts systematically, and measures such as phase-locking or phase-based DTF are biased even when nothing couples the two. Envelopes compare slow *power* changes and are not affected by the carrier-frequency difference. In simulation (§8.2) the envelope-based estimate stays within 50% of its zero-gap value as the centre-frequency gap grows from 0 to 10 Hz, whereas the phase-based estimate collapses.

**What we give up.** The envelope represents slow coordination of engagement, not fast phase coupling (§2.2). It is a complementary measure, not a replacement for phase-based hyperscanning analyses.

> **[FIG 5 — From signal to envelope.]** Three traces for one participant: raw ROI signal, individual-band filtered signal, envelope. Use the existing figure `fig_envelope_rationale_W_005_caregiver_Peppa`.

### 4.3 Cardiac: the inter-beat-interval series

The IBI series is interpolated onto a regular grid and *down-sampled only* — no band-pass and no envelope. In the 0.2–1 Hz range the raw IBI series is dominated by respiratory sinus arrhythmia (RSA), i.e. the respiration-paced modulation of heart rate by the vagus. This is the same frequency range in which the EEG envelope fluctuates, so the two modalities can share one model (Fig. 6).

### 4.4 A deliberate and explicit trade-off

There are three ways to put "vagal control" into the model (Table 2). The raw IBI carries the RSA oscillation itself (the *carrier*; first-order). The construct that is conceptually parallel to the EEG envelope is the *amplitude* of RSA (RSA depth, or high-frequency power) — a second-order quantity, but about ten times slower than its carrier. Using RSA depth next to a ~0.2–1 Hz EEG envelope would reopen the timescale gap that the envelope was meant to close. We therefore chose the raw IBI (option A): it matches the EEG envelope in *frequency*, but not in *order*. This is an estimation-driven choice and it changes what a cross-modal edge means (§7.2).

**Table 2. Three ways to represent the cardiac signal.**

| | A (current) | A′ | B |
|---|---|---|---|
| Cardiac signal | raw IBI (RSA carrier) | RSA depth | RSA depth |
| EEG signal | fast envelope | fast envelope | band-matched slow envelope |
| What is matched | frequency band (~0.2–1 Hz) | order (both second-order) | order and band (~0.01–0.1 Hz) |
| Meaning of cross-modal edge | partly respiratory-timing coupling | control → engagement | control → engagement (cleanest) |
| Direction-bias risk | moderate | high (very smooth) | high |
| Feasible for ~60 s clips? | yes | possible | no (~30 samples) — needs minutes-long segments |

A useful diagnostic is A vs. A′: keep the EEG side fixed and swap only the cardiac channel. If a cross-modal edge survives both, it is not merely coupling through the respiratory carrier. Option B is reserved for longer recordings (free conversation, minutes-long segments).

### 4.5 Common rate and common filter

All channels are down-sampled to 2.5 Hz (Nyquist 1.25 Hz) and pass through one identical zero-phase band-pass (0.05–1.0 Hz, 2nd-order Butterworth), so that group delay is identical for every channel. After this step every variable occupies a comparable, non-aliased band (Fig. 6). Segments are ~60 s, i.e. ~150 samples per channel.

> **[FIG 6 — The two modalities in one band.]** PSDs of the four design variables (child/caregiver EEG envelope and IBI) on the common 2.5 Hz grid for one dyad (z-scored for display only).

---

## 5. The joint model and its estimation

### 5.1 The model

For each dyad and film, we form the vector of *k* = 4 variables (child ROI envelope, caregiver ROI envelope, child IBI, caregiver IBI), z-scored over time. We fit a multivariate autoregressive (MVAR) model of order *p*:

$$
\mathbf{x}(t) \;=\; \sum_{r=1}^{p} A(r)\,\mathbf{x}(t-r) + \boldsymbol{\varepsilon}(t),
\tag{5}
$$

where $A(r)$ is the $k\times k$ coefficient matrix at lag *r*; the element $A_{ij}(r)$ says how much the value of channel *j* at lag *r* contributes to the present value of channel *i*, given all other channels. $\boldsymbol\varepsilon$ is the innovation noise with covariance Σ. The model is *config-driven*: the number and type of nodes and the directed edges of interest are declared once, so the same code supports different node sets (more ROIs, other signals) without modification.

> **[FIG 7 — The four-node model.]** Four nodes (two cortical, two cardiac; child left, caregiver right) and the 12 directed edges, with inter-brain (B–B) and inter-heart (H–H) edges highlighted and the cross-modal edges in a different colour.

### 5.2 Short, non-stationary segments: windowed autocorrelation averaging

A ~60 s segment holds only ~150 samples, and the cardiac signal is non-stationary within it (RSA frequency and amplitude drift). Fitting one model to the whole segment violates stationarity. Instead we cut the segment into short overlapping windows (10 s = 25 samples, 50% overlap, ≈11 windows), remove the linear trend in each window, treat each window as one *realization* of the same process, and estimate the model from the **autocovariance averaged across windows** [Kamiński et al., 2001]:

$$
\bar R(\tau) \;=\; \frac{1}{W}\sum_{w=1}^{W} R_w(\tau), \qquad \sum_{r=1}^{p} A(r)\,\bar R(s-r) = \bar R(s),\quad s=1,\dots,p,
\tag{6}
$$

where $R_w(\tau)$ is the lag-τ autocovariance matrix within window *w*. The right-hand set of equations (Yule–Walker equations) is solved once, for the averaged $\bar R$. Windows are never stitched together, and no window crosses a film boundary, so no discontinuities enter the data. Z-scoring is done once per film, *before* windowing, so that variance drift across windows is preserved in the average and not removed.

In our data this was decisive: the windowed fit gave stable models with white residuals in 126 of 126 dyad×film cases, versus 0 of 126 for a single fit on the whole segment `[TBD: re-check count against final dataset]`. Stability was checked from the eigenvalues of the companion matrix (all moduli < 1) and whiteness from the residual autocorrelation within windows.

### 5.3 Model order

The order *p* is chosen by information criteria computed on the global signal, minimized over *p* = 1…15:

$$
\mathrm{AIC}=\ln|\Sigma|+\frac{2pk^2}{n},\quad
\mathrm{HQ}=\ln|\Sigma|+\frac{2\ln(\ln n)\,pk^2}{n},\quad
\mathrm{SC}=\ln|\Sigma|+\frac{\ln(n)\,pk^2}{n},
\tag{7}
$$

with Σ the residual covariance, *k* the number of channels and *n* the number of samples. Schwarz's criterion (SC), the most conservative, is the default. With only ~150 samples and 4×4×*p* coefficients, parsimony is essential: we monitor the ratio of samples to parameters. Order is chosen once for the joint model (cross-modal edges exist only there); per-modality sub-block orders are reported for diagnostics only. For group comparison, real and surrogate data are re-fitted at a *common* order (p = 4) so that the dependent variable is not confounded by order differences between dyads.

> **[FIG 8 — Windowed-ACF estimation.]** One film segment with overlapping windows; the window-wise autocovariances averaged; one MVAR fit. Add a small panel contrasting a global fit (unstable roots) with the windowed fit (stable).

---

## 6. Directed measures: from DTF to dDTF

### 6.1 The transfer function

Equation (5) is a linear filter. In the frequency domain, convolution becomes multiplication: with $A(f)=I-\sum_r A(r)e^{-i2\pi f r/f_s}$, the transfer matrix is

$$
H(f) \;=\; A(f)^{-1}, \qquad S(f) = H(f)\,\Sigma\,H(f)^{\mathsf H}.
\tag{8}
$$

$H_{ij}(f)$ describes how a signal at frequency *f* is transmitted from channel *j* to channel *i*; $S(f)$ is the spectral matrix of the channels.

### 6.2 Three normalizations

The **directed transfer function** is the squared transfer from *j* to *i*, normalized by all inflows to *i* at that frequency [Kamiński & Blinowska, 1991]:

$$
\mathrm{DTF}_{ij}(f) \;=\; \frac{|H_{ij}(f)|^2}{\sum_{m=1}^{k} |H_{im}(f)|^2}.
\tag{9}
$$

It is the fraction of the signal at *i* (at frequency *f*) that comes from *j*. DTF includes indirect paths (*j* → *m* → *i*).

The **full-frequency DTF (ffDTF)** normalizes by the inflow summed over all frequencies, which makes magnitudes comparable across frequencies [Korzeniewska et al., 2003]:

$$
\mathrm{ffDTF}_{ij}(f) \;=\; \frac{|H_{ij}(f)|^2}{\sum_{f'}\sum_{m=1}^{k} |H_{im}(f')|^2}.
\tag{10}
$$

The **direct DTF (dDTF)** multiplies ffDTF by the magnitude of the *partial coherence* $\kappa_{ij}(f)$, which measures the relation between *i* and *j* after statistically removing the influence of all other channels (computed from minors of $S(f)$):

$$
\mathrm{dDTF}_{ij}(f) \;=\; \mathrm{ffDTF}_{ij}(f)\cdot|\kappa_{ij}(f)|.
\tag{11}
$$

ffDTF says "there is a transmission path from *j* to *i*"; partial coherence says "*i* and *j* are directly related once the others are accounted for". Their product is large only when both hold, so **dDTF suppresses edges that are only indirect.** This is why it is our default estimator, and why it is suited to a design in which cross-modal and within-modality edges coexist and could otherwise be confused through a third channel. Generalized partial directed coherence (GPDC) is available as an alternative that normalizes by each channel's noise variance [Baccalá & Sameshima, 2007].

### 6.3 From a spectrum to one number per edge

For each dyad × film × directed edge we average dDTF over the *coupling band* (the range where the envelope and the IBI series share power; `[TBD: 0.15–0.5 Hz in current configuration]`) and, for distributional reasons, apply a Box-Cox transform (λ = 0.25) before statistics. We always keep the **sign** of derived quantities (§7) — never absolute values — since the sign of a cardiac effect may be informative.

---

## 7. Inference: separating coupling from artefacts

A directed estimate is not trustworthy on its own. Two sources of spurious edges are specific to our setting, and both are handled by the same device, a surrogate null.

### 7.1 Two threats to validity

**Smoothness.** DTF-family estimators tend to label the smoother, more self-predictable channel as the *sender*. In our design the cardiac signal is smoother than the EEG envelope, so a naive "heart → brain" edge may partly reflect smoothness. We tested this on synthetic data with *zero* coupling: two independent resonators at the same frequency (0.3 Hz) that differ only in how narrow-band (smooth) they are. With z-scored channels (as in our pipeline) the smoother channel appears as the source, and the spurious edge grows with the smoothness contrast. The effect even changes sign without z-scoring, because dDTF is not scale-invariant. Since the direction depends on a pre-processing choice, direction must be read off a null distribution rather than reasoned about a priori.

**Shared stimulus.** Both people watch the same film, so common sensory drive raises estimated coupling even when the two do not interact.

### 7.2 Film-matched surrogate dyads

A surrogate dyad takes the child's variables from one dyad and the caregiver's variables from a *different* dyad, **for the same film**. Each channel keeps its own smoothness and its response to the film, but any real interaction between the pair is destroyed. For a film with *N* dyads, we use all *N(N−1)* ordered mismatched pairs, estimate the same dDTF on each (same estimator, order and windows), and discard unstable fits (reported). The real dyad's edge is then compared with this null:

$$
\Delta_{e} \;=\; \mathrm{dDTF}^{\text{real}}_{e} - \operatorname{median}\big(\mathrm{dDTF}^{\text{null}}_{e}\big),\qquad
z_{e} \;=\; \frac{\Delta_{e}}{1.4826\cdot\mathrm{MAD}\big(\mathrm{dDTF}^{\text{null}}_{e}\big)} .
\tag{12}
$$

Median and median absolute deviation (MAD; the factor 1.4826 rescales it to a standard deviation under normality) are used so that a few extreme surrogates cannot dominate the centre or the spread. Δ keeps the units of the measure; *z* is scale-free and is our primary outcome when comparing edges with different signal-to-noise properties, such as child vs. adult.

A connection "counts" only if the real dyad shows more than random pairs of people who watched the same film. In the smoothness simulation above, Δ stays at about zero across the whole smoothness gradient, and a genuine injected edge remains positive (§8.3).

> **[FIG 9 — Surrogate dyads.]** Schematic: child from dyad *a*, caregiver from dyad *b*, same film; the null distribution of band-averaged dDTF per edge; the real value as a vertical line; Δ and z indicated.

### 7.3 A subtle point for group comparisons

A pooled null (one per film, shared by all groups) is a single number per film and edge. Subtracting it from every dyad's value shifts all dyads in a film by the same amount, so **it cancels exactly from any between-group contrast**. It therefore cannot rule out that two groups respond *differently* to the same film. For group tests we additionally build the null *within group* (surrogates formed only from members of the same group, which requires at least two dyads per film × group cell). The pooled null remains the reference for within-group effects.

### 7.4 What an edge can and cannot mean

- *Heart–heart edges* (e.g. caregiver IBI → child IBI) with raw IBI reflect, in part, coordination of respiration-paced rhythms. Respiration itself is not recorded as a node in the current design, and a shared breathing pattern could drive both IBI series. Adding a respiration channel is the natural extension (§9).
- *Cross-modal edges* (e.g. caregiver IBI → child cortical envelope) are, with raw IBI, partly respiratory-timing coupling; with RSA depth they would read as control → engagement (Table 2). They are exploratory.
- A related caution: children and adults breathe at different rates, so the cardiac carriers of the two persons also differ in frequency. The argument that motivates envelopes on the EEG side (carrier frequency mismatch) therefore also applies, in principle, to raw IBI. Wide coupling band, surrogate nulls and the A vs. A′ diagnostic mitigate this; a dedicated synthetic test is planned `[TBD]`.

---

## 8. Validation on synthetic ground truth

All checks run the real estimator code on signals with known directed structure, and are re-run whenever the estimator changes.

### 8.1 Orientation and recovery

We inject a single directed edge and verify empirically which axis of the output is source and which is target (so that this convention is never assumed). In a four-node model that mimics the real design — child and caregiver ROI and IBI nodes, plus one novel cross-modal edge — injected edges are recovered with larger dDTF than their reverse by a prespecified margin.

### 8.2 Envelope vs. phase when centre frequencies differ

We simulate child–caregiver pairs with one coupled oscillator pair, and widen the difference in centre frequency from 0 to 9.5 Hz (100 realizations per gap, 3-min recordings). Each estimate is expressed relative to its own zero-gap value. The envelope-based dDTF remains within 50% of baseline across the whole range; the phase-based dDTF drops by more than 50% `[TBD: add numbers and CI from final run]`.

> **[FIG 10 — Envelope vs. phase.]** Recovered coupling (relative to zero gap) vs. centre-frequency gap (Hz); mean ± 95% CI over realizations; two curves.

### 8.3 Smoothness gradient and its cure

Using the same machinery as §7 on two-channel data with zero coupling, we sweep the smoothness contrast (pole radius of the smooth channel from 0.50 to 0.97; 6 batches × 16 dyads per point). Raw dDTF in the smooth→rough direction rises with the contrast, but so does the surrogate null, and Δ stays at ≈0 with |z| below about 0.5 across the gradient. When a genuine rough→smooth coupling is injected (30 dyads), the real distribution separates from the null (Δ > 0).

> **[FIG 11 — The artefact and the cure.]** Panel A: real dDTF, null median and Δ across the smoothness gradient (zero true coupling). Panel B: null vs. real histogram for ground truth = 0 and ground truth > 0.

---

## 9. Potential of dDTF for multimodal hyperscanning

**9.1 Questions that become askable.** The method turns descriptive questions ("are the two signals related?") into directional and conditional ones:

1. *Within a modality:* does the caregiver's cortical engagement predict the child's more than the reverse, once heart dynamics are accounted for?
2. *Across modalities, within a person* (brain–heart): does autonomic state predict cortical engagement, and when?
3. *Across modalities, between persons:* does one person's autonomic state predict the other's cortical engagement? This is the edge that, to our knowledge, has not been modelled directly.
4. *Asymmetry:* is the forward edge larger than the reverse (caregiver-leading vs. child-leading)? Direction is a result to be tested, not an assumption — the literature does not favour one direction.

**9.2 Why conditioning matters in practice.** Suppose both people's rhythm power rises during an exciting scene because both become aroused. A pairwise model shows a brain→brain edge. In the joint model, the cardiac channels carry part of that shared arousal, so the inter-brain edge is estimated net of it. The logic only works for confounders that are *in the model*, which motivates the next point.

**9.3 Extensibility.** Because nodes and edges are declared in a configuration file, the same pipeline can include further ROIs per person, respiration, pupil size or electrodermal activity, as long as each signal is brought to a common rate and band by an appropriate transformation. For modalities that are already slow, such as haemodynamic fNIRS signals, no envelope step is needed, and the commensurability with IBI is more direct.

**9.4 Timescale-resolved questions.** Because dDTF is frequency-resolved, one can ask at which timescale interpersonal autonomic-to-cortical coupling is carried: a fast "engagement" timescale (option A) or a slow "control" timescale (option B, minutes-long segments). The comparison A vs. B is itself a new empirical question.

**9.5 Group inference with uncertainty.** Δ or *z* per dyad × film × edge feed a Bayesian mixed-effects model (Student-*t* likelihood, film × group fixed effects with sum-to-zero coding, random intercept for dyad; weakly informative priors; contrasts as posterior means with 95% HDI and directional probabilities, no *p*-values as primary outputs). Edges are standardized within edge so one prior set is equally weakly informative for all. A named primary family receives mild Benjamini–Hochberg correction; the remaining edges are reported as hypothesis-generating. Convergence is gated (R̂ < 1.01, ESS > 400, no divergences, Pareto-*k* mostly < 0.7).

**9.6 Beyond caregiver–child dyads.** Nothing in the method is specific to the caregiver–child design: the same pipeline applies to adult dyads, to free conversation, and to other groups, provided the within-group null and sample-size gating are respected.

---

## 10. Illustrative application

`[Authors to decide whether to include; present as a demonstration of the pipeline, not as an inferential result.]`

We show the pipeline on caregiver–child dyads (typically developing and autism groups) watching three animated films of different valence (Peppa, Incredibles, Brave; treated as unordered conditions, not as a scale). Nodes were the temporo-parietal ROI (electrode P7, fast individual band) and raw IBI of each person (Fig. 7). Six edges were examined: caregiver→child and child→caregiver for the brain and heart (primary), and the two cross-modal edges (exploratory).

As a proof of concept for one film (Incredibles; order p = 4), the real dyads' dDTF is plotted against the film-matched null (Fig. 12). In this film, typically developing dyads show above-null caregiver→child autonomic coupling that autism dyads do not, including a cross-modal caregiver-IBI → child-cortical edge, whereas cortico-cortical edges show no excess in either group. The pattern is film-specific (it reverses or vanishes in the other two films), which we read as a stimulus-evoked, not trait-like, difference. **We treat this as an exploratory hypothesis.** Confirmation requires the predictions in Box 1 to be tested in held-out data, with within-group surrogates.

> **Box 1. Predictions to test in new data.** (i) group × edge-family interaction: TD > ASD on edges involving IBI, ≈ 0 on brain–brain edges; (ii) group × stimulus interaction: the TD–ASD difference tracks an independent stimulus property (arousal/engagement) and is not uniform across films; (iii) directional asymmetry: caregiver→child autonomic–cortical flow exceeds the reverse.

> **[FIG 12 — Real vs. null for one film.]** Split-violin per edge: grey = film-matched surrogate null; blue/orange = real dyads by group; y-axis ΔdDTF (Box-Cox, λ = 0.25 scale). Caption must state: one film, proof of concept, not inference; ROI = P7.

---

## 11. Scope and limitations

**The method claims:** slow coupling of engagement, quantified; directed and conditioned estimates; recovery of planted structure and suppression of known artefacts on synthetic data.

**It does not claim:**

- *Fast neural coupling.* The envelope carries no phase information.
- *Mechanism or anatomical source.* Granger influence is predictive; scalp electrodes are coarse proxies for regions (P7 as a TPJ proxy).
- *Completeness of the model.* Unmeasured drivers (respiration, motion, gaze) can produce edges.
- *Equal footing of modalities.* EEG is a second-order (envelope) signal and IBI a first-order one; cross-modal edges must be read with that in mind (Table 2).
- *Robust estimation from very short data.* ~150 samples per segment constrains model order; estimates are noisy at the dyad level and informative mainly in aggregate.
- *Order selection per window.* Order is selected on the global signal and then used in the windowed fit.
- *Fixed thresholds.* Several decisions in the band-assignment stage are pragmatic constants, not data-derived.

Because the cross-modal edge is novel, the standard of evidence is higher: within-group surrogates for any group test, and out-of-sample replication before any confirmatory claim.

---

## 12. Conclusions

Envelopes of individual rhythms make cortical and cardiac signals commensurate in frequency, and a joint conditioned dDTF turns multimodal hyperscanning from "which signals are related?" into "who predicts whom, through which channel, and beyond the stimulus?". The conclusions are credible only because each step is paired with a check: individual bands with ROI viability and stability checks, envelopes with a simulation of the frequency-gap bias, directional estimates with surrogates, and the whole pipeline with synthetic ground truth.

---

## Data and code availability

Code: `hyperscanning-signal-analysis` repository `[TBD: public URL/DOI]`. Configuration: `pipeline_config.json`. Data: `[TBD]`.

## Acknowledgements

The work was conducted as part of the SYNCC-IN project funded by the European Union under the Horizon Europe programme (agreement No 101159414). Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the granting authority.

## References (to be verified and completed)

- Baccalá, L. A., & Sameshima, K. (2001). Partial directed coherence: a new concept in neural structure determination. *Biological Cybernetics*, 84, 463–474. `[REF]`
- Baccalá, L. A., & Sameshima, K. (2007). Generalized partial directed coherence. *15th International Conference on Digital Signal Processing*. `[REF]`
- Capraz, Y. Z., Konrad, K., & Reindl, V. (2023). Concurrent and lagged physiological synchrony during mother–child interaction and their relationship to positive affect in 8- to 10-year-old children. *Scientific Reports*, 13, 17744.
- Davis, M., West, K., Bilms, J., Morelen, D., & Suveg, C. (2018). A systematic review of parent–child synchrony: It is more than skin deep. *Developmental Psychobiology*, 60(6), 674–691.
- Donoghue, T., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655–1665. `[REF]`
- Feldman, R., Magori-Cohen, R., Galili, G., Singer, M., & Louzoun, Y. (2011). Mother and infant coordinate heart rhythms through episodes of interaction synchrony. *Infant Behavior and Development*, 34(4), 569–577.
- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37, 424–438. `[REF]`
- Kamiński, M., & Blinowska, K. J. (1991). A new method of the description of the information flow in the brain structures. *Biological Cybernetics*, 65, 203–210. `[REF]`
- Kamiński, M., Ding, M., Truccolo, W. A., & Bressler, S. L. (2001). Evaluating causal relations in neural systems: Granger causality, directed transfer function and statistical assessment of significance. *Biological Cybernetics*, 85, 145–157. `[REF]`
- Kang, E., et al. (2023). Electroencephalographic interbrain synchronization in children with disabilities, their parents, and neurologic music therapists. *European Journal of Neuroscience*. doi:10.1111/ejn.16036
- Korzeniewska, A., Mańczak, M., Kamiński, M., Blinowska, K. J., & Kasicki, S. (2003). Determination of information flow direction among brain structures by a modified directed transfer function (dDTF) method. *Journal of Neuroscience Methods*, 125, 195–207.
- Leong, V., Byrne, E., Clackson, K., Georgieva, S., Lam, S., & Wass, S. (2017). Speaker gaze increases information coupling between infant and adult brains. *PNAS*, 114(50), 13290–13295.
- Marriott Haresign, I., Phillips, E., Whitehorn, M., Noreika, V., Jones, E. J. H., Leong, V., & Wass, S. V. (2022). Measuring the temporal dynamics of inter-personal neural entrainment in continuous child–adult EEG hyperscanning data. *Developmental Cognitive Neuroscience*, 54, 101093.
- Reindl, V., Gerloff, C., Scharke, W., & Konrad, K. (2022). Multimodal hyperscanning reveals that synchrony of body and mind are distinct in mother–child dyads. *NeuroImage*, 251, 118982.
- Wass, S. V., et al. (2018). Parental neural responsivity to infants' visual attention: How mature brains influence immature brains during social interaction. *PLOS Biology*, 16(12), e2006328.
- `[REF]` Valenza et al. (within-person brain–heart directed coupling); Esposito et al. (film valence, fNIRS); h-DCM inter-brain studies — add exact citations from the talk.
- `[REF]` Bambi / ArviZ software citations; Box–Cox (1964); Thomson (1982) multitaper.

---

## Author notes – delete before submission

Points I found while reading the code and docs that need your decision or a check before this goes further:

1. **Coupling band mismatch.** `docs/methodology_multimodal_ddtf_pipeline.md` states the default coupling band 0.2–1.0 Hz; `pipeline_config.json` has `COUPLING_BAND_HZ = [0.15, 0.5]`. I wrote the band as a `[TBD]`. Please confirm which one was used for the reported results.
2. **Dyad counts.** The "126/126 cases" statement implies 42 dyads × 3 films, while the config lists 51 `included_dyads`. Update the numbers to the final dataset.
3. **Bandwidth convention in Eq. 3.** The config notes `bw_half = fast_bw/2`. I assumed BW is the two-sided width reported by specparam; please verify against `src/bands.py`.
4. **Missing rhythm.** `band_lookup` returns `(None, None)` when a participant has no fast peak in the ROI. I could not confirm in the stage-2 code whether the dyad is dropped or handled otherwise — §3.8 needs the actual rule and the number of excluded dyad × film cells.
5. **Respiration and the raw-IBI carrier (§7.4).** The envelope argument rests on carrier-frequency mismatch between child and adult. Raw IBI is itself a carrier whose frequency (breathing rate) differs by age. I flagged this as a limitation; a synthetic test with a frequency gap on the IBI side would strengthen the paper, and recording respiration would address the main confound.
6. **Stage-0 numbers.** The text says the phase-based estimate "collapses" and the envelope stays within 50% of baseline; please add effect sizes/CIs from the final run. The smoothness-gate HTML was produced with *p* = 2 while the script docstring says locked settings are *p* = 4 — check which is reported.
7. **Illustrative application (§10).** This is the most claim-heavy part. It is based on slides 16 and 22 (one film, one electrode). I suggest keeping it short and strictly labelled exploratory, or moving it to a separate empirical paper.
8. **Order selection** is global (2-D) and reused for windowed fits; I listed this as a limitation. If you can show that results are insensitive to *p* (the code has a window-choice sensitivity check, but not an order-sensitivity one), the argument would be stronger.
9. **Figures.** Only the envelope rationale figure (`figures/protocol_paper/`) and the Stage-0/0b gate figures exist already; the others need to be made.
