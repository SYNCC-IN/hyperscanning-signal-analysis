"""Level-2 cardiac (RSA-type) generator and the three cardiac representations A, A', B (simulations S9, S10, S11).

A cardiac node is an RSA-type oscillation ``exp(l(t)) cos(phi(t))``: a carrier at the respiratory frequency
(person-specific, with slow drift) times a slow amplitude. Child and caregiver can be linked by
  - 'pulling'   : directed phase coupling (frequency pulling) from caregiver to child,
  - 'amplitude' : the caregiver's log-amplitude drives the child's log-amplitude,
  - 'none'.
For S10 the child's EEG envelope can additionally be driven by the caregiver's cardiac signal:
  - 'linear'    : the caregiver's RSA oscillation itself modulates the log EEG power (breathing-paced modulation),
  - 'amp_to_eeg': the caregiver's RSA amplitude modulates the log EEG power,
  - 'none'.
All series are generated at `FS_HI` and brought to the 2.5 Hz analysis rate by `represent`.
"""

import numpy as np

from src.envelopes import bandpass_filter, downsample, hrv_hf_envelope
from src.sim_generators import add_observation_noise, ar2_process, lagged, ou_process, simulate_phase_oscillators

FS_HI = 10.0
FS_ANALYSIS = 2.5
EDGE_TRIM_S = 20.0                       # simulated and discarded at both ends (filter and Hilbert edge effects)
RSA_HF_BAND = (0.15, 0.6)                # band-pass used to extract the RSA amplitude [TBD: age-adjusted in real data]


def standardize(x):
    """Zero mean, unit standard deviation."""
    return (x - x.mean()) / x.std()


def simulate_cardiac_dyad(rng, f_child, f_cg, duration_s, link="none", strength=0.0, link_lag_s=1.0,
                          phase_noise_sd=0.5, drift_sd_hz=0.02, amp_sd=0.4, amp_tau_s=15.0, obs_snr=5.0):
    """One dyad's raw cardiac oscillations (child, caregiver) and the clean caregiver amplitude process.

    Parameters
    ----------
    rng : numpy.random.Generator
    f_child, f_cg : float
        Respiratory (carrier) frequencies in Hz.
    duration_s : float
        Analysed duration; `EDGE_TRIM_S` is added at both ends.
    link : {'none', 'pulling', 'amplitude'}
    strength : float
        'pulling': phase-coupling strength in Hz (K = 2 pi strength rad/s). 'amplitude': gain of the caregiver's
        standardized log-amplitude on the child's log-amplitude.
    link_lag_s : float
        Delay of the amplitude link.

    Returns
    -------
    cardiac : np.ndarray, shape (2, n)   rows: child, caregiver (with observation noise)
    clean_cg : np.ndarray, shape (n,)    noise-free caregiver oscillation (for the EEG link)
    log_amp_cg : np.ndarray, shape (n,)  caregiver log-amplitude
    """
    n = int((duration_s + 2 * EDGE_TRIM_S) * FS_HI)
    k = np.zeros((2, 2))
    if link == "pulling":
        k[0, 1] = 2 * np.pi * strength
    _, _, phase = simulate_phase_oscillators(rng, [f_child, f_cg], k, FS_HI, n, phase_noise_sd, drift_sd_hz=drift_sd_hz)
    log_amp = np.stack([ou_process(rng, n, FS_HI, amp_sd, amp_tau_s) for _ in range(2)])
    if link == "amplitude":
        log_amp[0] = log_amp[0] + strength * lagged(standardize(log_amp[1]), int(round(link_lag_s * FS_HI))) * amp_sd
    clean = np.exp(log_amp) * np.cos(phase)
    return add_observation_noise(clean, obs_snr, rng), clean[1], log_amp[1]


def simulate_eeg_envelope(rng, n, source_oscillation, source_log_amp, fast_radius, fast_f0_hz, link="none", strength=0.0,
                          link_lag_s=0.4, fast_sd=0.5, slow_sd=0.5, slow_tau_s=15.0):
    """Child's EEG envelope at `FS_HI`: exp of a log-amplitude process, optionally driven by the caregiver's cardiac signal.

    The log-amplitude is a fast resonant AR(2) component (pole radius `fast_radius` and frequency `fast_f0_hz` at
    `FS_HI`, standard deviation `fast_sd`) plus a slow Ornstein-Uhlenbeck component (`slow_sd`, `slow_tau_s`).
    """
    log_env = ar2_process(rng, n, FS_HI, fast_sd, fast_radius, fast_f0_hz) + ou_process(rng, n, FS_HI, slow_sd, slow_tau_s)
    lag = int(round(link_lag_s * FS_HI))
    if link == "linear":
        log_env = log_env + strength * lagged(standardize(source_oscillation), lag)
    elif link == "amp_to_eeg":
        log_env = log_env + strength * lagged(standardize(source_log_amp), lag)
    return np.exp(log_env)


def _trim(x, fs):
    """Remove the simulated edge margin."""
    m = int(EDGE_TRIM_S * fs)
    return x[m:-m]


def represent(signal_hi, option, role):
    """Bring one raw high-rate series to the 2.5 Hz analysis rate under option 'A', 'Aprime' or 'B'.

    Parameters
    ----------
    signal_hi : np.ndarray, shape (n,)
    option : {'A', 'Aprime', 'B'}
    role : {'ibi', 'eeg'}
        'ibi': cardiac series (raw IBI oscillation under A, RSA amplitude under A' and B);
        'eeg': EEG envelope (shared 0.05-1 Hz band under A and A', band-matched 0.01-0.15 Hz under B).
    """
    if role == "ibi" and option in ("Aprime", "B"):
        series, fs = hrv_hf_envelope(signal_hi, FS_HI, RSA_HF_BAND[0], RSA_HF_BAND[1], 4, FS_ANALYSIS)
    else:
        series, fs = downsample(signal_hi, FS_HI, FS_ANALYSIS)
    low, high = (0.01, 0.15) if option == "B" else (0.05, 1.0) if option == "A" else (0.01, 1.0)
    if option == "B" or option == "Aprime":
        series = bandpass_filter(series, fs, low, high, 2)
    else:
        series = bandpass_filter(series, fs, low, high, 4)
    return _trim(series, fs)
