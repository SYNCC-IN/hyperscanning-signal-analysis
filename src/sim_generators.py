"""Generators for the simulation studies of the multimodal dDTF framework.

Level 1 generators work directly at the analysis rate: a VAR process with a
known coupling tensor (resonant AR(2) self-terms plus planted directed edges),
an optional common stimulus, and optional measurement noise. Level 2 helpers
produce signal-level data: phase-coupled oscillators with drifting frequency
and slowly varying amplitude (EEG-like carriers, respiration-paced cardiac
oscillations). All random draws come from a `numpy.random.Generator` passed in
by the caller, so every simulation is reproducible from its seed.
"""

import numpy as np
from scipy.signal import lfilter

try:
    from .synthetic_mvar import self_ar2_coeffs
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.synthetic_mvar import self_ar2_coeffs


def build_coupling(pole_radius, f0_hz, fs, edges, n_lags):
    """Coupling tensor: one AR(2) resonator per node plus planted directed edges.

    Parameters
    ----------
    pole_radius, f0_hz : sequence of float, length k
        Pole modulus (self-persistence, closer to 1 = smoother channel) and
        resonance frequency of each node's AR(2) self-term.
    fs : float
        Sampling frequency (Hz).
    edges : list of tuple
        ``(source, target, lag, gain)``; ``lag >= 1`` samples.
    n_lags : int
        Number of lags in the tensor (at least 2 and at least the largest edge lag).

    Returns
    -------
    np.ndarray, shape (k, k, n_lags)
        ``coupling[target, source, lag - 1]``, the convention of `generate_var_process`.
    """
    k = len(pole_radius)
    coupling = np.zeros((k, k, n_lags))
    for node in range(k):
        a1, a2 = self_ar2_coeffs(pole_radius[node], f0_hz[node], fs)
        coupling[node, node, 0] = a1
        coupling[node, node, 1] = a2
    for source, target, lag, gain in edges:
        coupling[target, source, lag - 1] += gain
    return coupling


def simulate_var(coupling, n_samples, rng, noise_std=1.0, drive=None,
                 coupling_end=None, noise_scale=None, burn_in=200):
    """Simulate a VAR process, optionally with an external drive and slow drifts.

    Implements ``x[:, t] = sum_lag C(t)[:, :, lag-1] @ x[:, t-lag] + drive[:, t] +
    noise_scale[t] * noise[:, t]``. If `coupling_end` is given, the coefficient
    tensor changes linearly from `coupling` (start of the recording) to
    `coupling_end` (end), which produces a non-stationary process (drifting
    resonance frequency, changing coupling).

    Parameters
    ----------
    coupling : np.ndarray, shape (k, k, n_lags)
        Coefficient tensor at the start of the recording.
    n_samples : int
        Samples returned (after `burn_in`).
    rng : numpy.random.Generator
    noise_std : float, optional
        Innovation standard deviation.
    drive : np.ndarray, shape (k, n_samples), optional
        External input added to the process after the burn-in.
    coupling_end : np.ndarray, shape (k, k, n_lags), optional
        Coefficient tensor at the end of the recording.
    noise_scale : np.ndarray, shape (n_samples,), optional
        Time-varying multiplier of the innovation standard deviation (after burn-in).
    burn_in : int, optional
        Initial samples discarded.

    Returns
    -------
    np.ndarray, shape (k, n_samples)
    """
    k, _, n_lags = coupling.shape
    n_total = n_samples + burn_in
    noise = rng.normal(0.0, noise_std, size=(k, n_total))
    if noise_scale is not None:
        noise[:, burn_in:] *= noise_scale[None, :]
    x = np.zeros((k, n_total + n_lags))
    for t in range(n_total):
        if coupling_end is not None and t >= burn_in:
            weight = (t - burn_in) / max(n_samples - 1, 1)
            current = (1.0 - weight) * coupling + weight * coupling_end
        else:
            current = coupling
        acc = noise[:, t].copy()
        for lag in range(1, n_lags + 1):
            acc += current[:, :, lag - 1] @ x[:, n_lags + t - lag]
        if drive is not None and t >= burn_in:
            acc += drive[:, t - burn_in]
        x[:, n_lags + t] = acc
    assert np.all(np.isfinite(x)), "simulated VAR process diverged - coupling spec is unstable"
    return x[:, n_lags + burn_in:]


def sample_node_params(rng, n_dyads, ranges):
    """Draw one parameter value per dyad and node from uniform ranges.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_dyads : int
    ranges : np.ndarray, shape (k, 2)
        ``(low, high)`` per node.

    Returns
    -------
    np.ndarray, shape (n_dyads, k)
    """
    ranges = np.asarray(ranges, dtype=float)
    return rng.uniform(ranges[:, 0], ranges[:, 1], size=(n_dyads, ranges.shape[0]))


def add_observation_noise(signal, snr, rng):
    """Add white measurement noise so that std(signal) / std(noise) = snr, per row.

    Parameters
    ----------
    signal : np.ndarray, shape (k, n_samples)
    snr : float or None
        Signal-to-noise ratio (standard-deviation ratio); None leaves the signal unchanged.
    rng : numpy.random.Generator

    Returns
    -------
    np.ndarray, shape (k, n_samples)
    """
    if snr is None:
        return signal
    noise_std = signal.std(axis=1, keepdims=True) / snr
    return signal + rng.normal(size=signal.shape) * noise_std


def make_stimulus(rng, n_samples, fs, pole_radius, f0_hz):
    """A common stimulus-driven input: broadband AR(2) noise with unit standard deviation.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_samples : int
    fs : float
    pole_radius, f0_hz : float
        AR(2) parameters of the stimulus time course.

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    coupling = build_coupling([pole_radius], [f0_hz], fs, [], 2)
    u = simulate_var(coupling, n_samples, rng)[0]
    return u / u.std()


def sample_stimulus_gain(rng, n_dyads, base_gain, sd_log, dyad_multiplier=None):
    """Person-specific gain of the stimulus on each node of each dyad.

    ``gain[d, i] = base_gain[i] * exp(sd_log * N(0, 1)) * dyad_multiplier[d]``.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_dyads : int
    base_gain : sequence of float, length k
    sd_log : float
        Standard deviation of the log-normal person-to-person variability.
    dyad_multiplier : np.ndarray, shape (n_dyads,), optional
        Extra multiplier per dyad (e.g. a group difference in stimulus gain).

    Returns
    -------
    np.ndarray, shape (n_dyads, k)
    """
    base_gain = np.asarray(base_gain, dtype=float)
    gain = base_gain[None, :] * np.exp(sd_log * rng.normal(size=(n_dyads, base_gain.size)))
    if dyad_multiplier is not None:
        gain = gain * np.asarray(dyad_multiplier, dtype=float)[:, None]
    return gain


def simulate_dyads(rng, n_dyads, n_samples, fs, pole_radius_ranges, f0_ranges, edges, n_lags,
                   stimulus=None, stimulus_gain=None, obs_snr=None, drifting_nodes=None,
                   drift_f0_factor=1.0, variance_modulation=0.0):
    """Simulate `n_dyads` independent multi-node processes with a known directed structure.

    Every dyad draws its own pole radii and resonance frequencies (person-to-person
    heterogeneity in smoothness and spectrum) and shares the planted `edges`.
    Optionally a common `stimulus` drives all dyads with person-specific gains,
    measurement noise is added, and the nodes in `drifting_nodes` become
    non-stationary (resonance frequency drifting by `drift_f0_factor` over the
    recording and sinusoidal modulation of the innovation variance).

    Parameters
    ----------
    rng : numpy.random.Generator
    n_dyads, n_samples : int
    fs : float
    pole_radius_ranges, f0_ranges : np.ndarray, shape (k, 2)
        Uniform ranges per node.
    edges : list of tuple
        ``(source, target, lag, gain)``, shared by all dyads.
    n_lags : int
    stimulus : np.ndarray, shape (n_samples,), optional
        Common input time course (same for all dyads).
    stimulus_gain : np.ndarray, shape (n_dyads, k), optional
        Gain of the stimulus on each node of each dyad.
    obs_snr : float, optional
        Measurement signal-to-noise ratio (see `add_observation_noise`).
    drifting_nodes : sequence of int, optional
        Nodes made non-stationary.
    drift_f0_factor : float, optional
        Final / initial resonance frequency of drifting nodes.
    variance_modulation : float, optional
        Relative amplitude (0 to 1) of the sinusoidal innovation-variance modulation
        (one cycle over the recording), applied to the whole process.

    Returns
    -------
    np.ndarray, shape (n_dyads, k, n_samples)
        Raw (not z-scored) signals.
    """
    radii = sample_node_params(rng, n_dyads, pole_radius_ranges)
    f0s = sample_node_params(rng, n_dyads, f0_ranges)
    out = []
    for dyad in range(n_dyads):
        coupling = build_coupling(radii[dyad], f0s[dyad], fs, edges, n_lags)
        coupling_end = None
        if drifting_nodes is not None:
            f0_end = f0s[dyad].copy()
            f0_end[list(drifting_nodes)] *= drift_f0_factor
            coupling_end = build_coupling(radii[dyad], f0_end, fs, edges, n_lags)
        noise_scale = None
        if variance_modulation > 0.0:
            phase = rng.uniform(0, 2 * np.pi)
            noise_scale = 1.0 + variance_modulation * np.sin(2 * np.pi * np.arange(n_samples) / n_samples + phase)
        drive = None
        if stimulus is not None:
            drive = stimulus_gain[dyad][:, None] * stimulus[None, :]
        x = simulate_var(coupling, n_samples, rng, drive=drive, coupling_end=coupling_end, noise_scale=noise_scale)
        out.append(add_observation_noise(x, obs_snr, rng))
    return np.stack(out, axis=0)


def zscore_designs(raw):
    """Z-score every row of every dyad over time (ddof=0), as in the real design matrix.

    Parameters
    ----------
    raw : np.ndarray, shape (n_dyads, k, n_samples)

    Returns
    -------
    np.ndarray, same shape
    """
    return (raw - raw.mean(axis=2, keepdims=True)) / raw.std(axis=2, keepdims=True)


def ou_process(rng, n_samples, fs, sd, tau_s):
    """Stationary Ornstein-Uhlenbeck-type (AR(1)) process with given standard deviation and time constant.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_samples : int
    fs : float
    sd : float
        Stationary standard deviation.
    tau_s : float
        Correlation time (s).

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    decay = np.exp(-1.0 / (fs * tau_s))
    n_burn = int(np.ceil(5 * tau_s * fs))
    eps = rng.normal(size=n_samples + n_burn) * sd * np.sqrt(1.0 - decay ** 2)
    return lfilter([1.0], [1.0, -decay], eps)[n_burn:]


def simulate_phase_oscillators(rng, freq_hz, coupling_k, fs, n_samples, phase_noise_sd,
                               drift_sd_hz=0.0, drift_tau_s=20.0, amp_sd=0.0, amp_tau_s=10.0,
                               burn_in_s=20.0):
    """Noisy phase oscillators with directed phase coupling, drifting frequency and slow amplitude.

    Euler-Maruyama integration of
    ``d phi_i = (2 pi f_i(t) + sum_j K_ij sin(phi_j - phi_i)) dt + phase_noise_sd dW_i``
    with ``f_i(t) = freq_hz[i] + drift_i(t)``; the signal is ``exp(l_i(t)) cos(phi_i)``, where
    ``l_i`` is a slow log-amplitude process. ``K_ij`` (rad/s) is the influence of oscillator j on i;
    a pair locks if ``K >= 2 pi |f_i - f_j|`` (Adler condition).

    Parameters
    ----------
    rng : numpy.random.Generator
    freq_hz : sequence of float, length k
        Mean frequency of each oscillator.
    coupling_k : np.ndarray, shape (k, k)
        Phase-coupling strengths (rad/s), ``coupling_k[i, j]`` = influence of j on i.
    fs : float
        Integration / output sampling frequency (Hz).
    n_samples : int
    phase_noise_sd : float
        Phase-diffusion strength (rad / sqrt(s)).
    drift_sd_hz, drift_tau_s : float, optional
        Standard deviation (Hz) and correlation time (s) of the frequency drift.
    amp_sd, amp_tau_s : float, optional
        Standard deviation and correlation time of the log-amplitude process.
    burn_in_s : float, optional
        Seconds discarded at the start.

    Returns
    -------
    signal : np.ndarray, shape (k, n_samples)
    log_amp : np.ndarray, shape (k, n_samples)
    phase : np.ndarray, shape (k, n_samples)
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    k = freq_hz.size
    dt = 1.0 / fs
    n_burn = int(burn_in_s * fs)
    n_total = n_samples + n_burn
    if drift_sd_hz > 0:
        drift = np.stack([ou_process(rng, n_total, fs, drift_sd_hz, drift_tau_s) for _ in range(k)])
    else:
        drift = np.zeros((k, n_total))
    if amp_sd > 0:
        log_amp = np.stack([ou_process(rng, n_total, fs, amp_sd, amp_tau_s) for _ in range(k)])
    else:
        log_amp = np.zeros((k, n_total))
    inst_freq = freq_hz[:, None] + drift
    noise = rng.normal(size=(k, n_total)) * phase_noise_sd * np.sqrt(dt)
    phase = np.zeros((k, n_total))
    phase[:, 0] = rng.uniform(0, 2 * np.pi, k)
    for t in range(n_total - 1):
        pull = (coupling_k * np.sin(phase[None, :, t] - phase[:, None, t])).sum(axis=1)
        phase[:, t + 1] = phase[:, t] + dt * (2 * np.pi * inst_freq[:, t] + pull) + noise[:, t]
    signal = np.exp(log_amp) * np.cos(phase)
    return signal[:, n_burn:], log_amp[:, n_burn:], phase[:, n_burn:]


def lagged(x, lag):
    """Shift a 1-D series later in time by `lag` samples (zero-padded at the start).

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    lag : int
        Delay in samples (>= 1).

    Returns
    -------
    np.ndarray, shape (n,)
    """
    return np.concatenate([np.zeros(lag), x[:-lag]])
