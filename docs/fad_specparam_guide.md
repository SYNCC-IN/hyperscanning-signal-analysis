# FAD Decomposition & specparam Guide

**Version:** 1.0  
**Last updated:** 2026-04-10  
**Author:** Jarosław Żygierewicz

---

## Overview

This guide describes two complementary methods for extracting oscillatory components from EEG power spectra:

| Method | Module / Package | Input | Approach |
|--------|-----------------|-------|----------|
| **FAD** (Frequency-Amplitude-Damping) | `src/mtmvar.py` | raw time series | AR model + partial fraction decomposition |
| **specparam** (formerly FOOOF) | `specparam` (pip) | pre-computed power spectrum | Gaussian peak fitting on 1/f-adjusted spectrum |

Both approaches are demonstrated in `scripts/fad_demo_w030_fz.ipynb` using the W_030 caregiver Brave EEG signal (Fz / Pz channel).

---

## FAD Decomposition

### Background

FAD decomposes the impulse response of a univariate AR model into exponentially damped oscillators:

$$h(t) = \sum_j B_j \cdot e^{-\beta_j t} \cos(\omega_j t + \varphi_j)$$

Each component is characterised by:

| Symbol | Key in output dict | Unit | Description |
|--------|--------------------|------|-------------|
| $f_j$ | `freq_hz` | Hz | Resonance frequency |
| $B_j = 2\|C_j\|$ | `B` | signal units | Amplitude |
| $\beta_j$ | `beta` | s⁻¹ | Damping coefficient (positive = decay) |
| $\Delta f_j = \beta_j / (2\pi)$ | `bandwidth_hz` | Hz | Half-power bandwidth proxy |
| $\varphi_j$ | `phi` | rad | Initial phase |

The approach follows the partial-fraction expansion of the AR transfer function:

$$H(z) = \sum_j \frac{C_j \cdot z}{z - z_j}, \quad \alpha_j = \frac{\ln z_j}{\Delta t}$$

Reference: Blinowska & Żygierewicz, *Practical Biomedical Signal Analysis*, 2nd ed.

### API

```python
from src.mtmvar import fad_decomposition, fad_components_table

fad = fad_decomposition(
    signal,                # 1-D array, shape (N,)
    fs=fs,                 # sampling frequency [Hz]
    model_order=None,      # None → automatic selection via crit_type
    max_model_order=12,    # upper bound for automatic search
    crit_type='AIC',       # 'AIC' | 'HQ' | 'SC'
    plot=True,             # show pole plot + AR spectrum
    pair_conjugates=True,  # report one oscillator per conjugate pair
)

# Compact DataFrame (one row per oscillatory component)
df = fad_components_table(fad, output='dataframe', decimals=4)
```

### Output structure

`fad_decomposition` returns a dict. The most useful sub-structures are:

```
fad['model_order']          # int — selected AR order p
fad['paired_components']    # dict — one oscillator per conjugate pair
    ['freq_hz']             # ndarray — frequencies in Hz (sorted)
    ['B']                   # ndarray — amplitudes
    ['beta']                # ndarray — damping coefficients [s⁻¹]
    ['bandwidth_hz']        # ndarray — bandwidths [Hz]
    ['phi']                 # ndarray — phases [rad]
    ['poles']               # ndarray — complex poles z_j
fad['noise_variance']       # float — residual AR noise variance
fad['ar_coeffs']            # ndarray (p,) — fitted AR coefficients
```

### Model order selection

Automatic order selection evaluates AIC (or HQ / SC) for orders 1 … `max_model_order` and picks the minimum. A plot of the criterion vs. order is shown when `plot=True`.

Typical values for EEG (250–1024 Hz, bandpass 1–40 Hz): p = 6–16.

---

## specparam (formerly FOOOF)

### Background

`specparam` models the power spectrum as a sum of an aperiodic (1/f) component and one or more Gaussian peaks:

$$\log P(f) = L(f) + \sum_k G_k(f)$$

where $L(f)$ is a Lorentzian/knee model for the aperiodic background and $G_k$ are Gaussian peaks representing oscillatory components.

Package: **specparam 2.x** (`pip install specparam`)  
Documentation: https://specparam-tools.github.io/

### Quick start

```python
from specparam import SpectralModel
from scipy.signal import welch

# Compute PSD
freqs, psd = welch(x, fs=fs, nperseg=4*fs)

# Fit specparam model
sm = SpectralModel(
    peak_width_limits=[1.0, 8.0],  # [Hz] min/max peak width
    max_n_peaks=6,
    min_peak_height=0.1,
    aperiodic_mode='fixed',        # 'fixed' | 'knee'
    verbose=False,
)
sm.fit(freqs, psd, freq_range=[1, 45])
sm.report()                        # print + plot

# Access results
print(sm.aperiodic_params_)        # [offset, exponent]  (or [offset, knee, exponent])
print(sm.peak_params_)             # [[CF, PW, BW], ...]
```

### Key output parameters

| Attribute | Description |
|-----------|-------------|
| `aperiodic_params_` | `[offset, exponent]` (fixed) or `[offset, knee, exponent]` (knee) |
| `peak_params_` | `[[CF (Hz), PW (log power), BW (Hz)], …]` per peak |
| `r_squared_` | goodness of fit ($R^2$) |
| `error_` | mean absolute error on log power |

---

## FAD vs specparam: comparison

| Property | FAD | specparam |
|----------|-----|-----------|
| Input | Raw time series | Pre-computed PSD |
| Aperiodic removal | Implicit (AR captures 1/f trend) | Explicit Lorentzian/knee fit |
| Frequency resolution | Continuous (pole-based) | Limited by FFT resolution |
| Damping / bandwidth | Direct output ($\beta$, $\Delta f$) | Peak width BW (Gaussian σ) |
| Phase information | Yes ($\varphi_j$) | No |
| Model order choice | Required (automatic via AIC) | `max_n_peaks`, `peak_width_limits` |
| Multiple channels | Extend to MVAR for connectivity | Run per-channel independently |
| Package dependency | `scipy` only (via `src/mtmvar.py`) | `specparam` |

### Practical guidance

- **Use FAD** when you need phase estimates, damping constants, or when the data segment is short (AR is data-efficient).
- **Use specparam** when you want to cleanly separate aperiodic slope from peaks, compare peak center frequencies across conditions, or integrate with MNE/BIDS pipelines.
- The two methods can be run in parallel: FAD frequencies and specparam CF values for the same channel should agree within ±0.5 Hz for well-resolved peaks.

---

## Demo notebook

`scripts/fad_demo_w030_fz.ipynb` — end-to-end example:

1. Load W_030 EEG (caregiver, Brave) from NetCDF.
2. Run `fad_decomposition` on the Pz channel.
3. Display FAD component table.
4. Fit `SpectralModel` (specparam) on the Welch PSD of the same signal.
5. Compare component frequencies side by side.

---

## References

- Franaszczuk, P. J., Bergey, G. K., & Kamiński, M. J. (1994). Analysis of mesial temporal seizure onset and propagation using the directed transfer function method. *Electroencephalography and Clinical Neurophysiology*, 91(6), 413–427.
- Blinowska, K. J., & Żygierewicz, J. (2011). *Practical Biomedical Signal Analysis Using MATLAB®*. CRC Press.
- Donoghue, T., Haller, M., Peterson, E. J., Varma, P., Sebastian, P., Gao, R., … Voytek, B. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655–1665. https://doi.org/10.1038/s41593-020-00744-x
