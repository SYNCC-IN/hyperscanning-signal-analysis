"""Signal/HRV processing helpers for the SECoRe / H10 pipeline.

Holds IBI correction/interpolation and cross-correlation lag estimation.
Plotting lives in ``signal_plots``; file IO in ``secore_helpers``.
"""
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.interpolate import CubicSpline


def fix_and_interpolate_ibi(
    ibi_cum_s,
    stage,
    fs_out=8,
    samp_rate=1024,
    window_size=30,
):
    """
    Correct ectopic beats, interpolate IBI to a uniform grid, and compute RMSSD.

    Returns
    -------
    t_interp, ibi_interp, stage_interp, nn_ms, t_nn, rmssd_interp
    """
    cum_samp = ibi_cum_s * samp_rate
    _, nn_pos = nk.signal_fixpeaks(
        cum_samp,
        sampling_rate=samp_rate,
        iterative=True,
        method="Kubios",
        alpha=4,
        window_width=61,
        medfilt_order=11,
    )

    nn_ms = np.diff(nn_pos) / samp_rate * 1000.0
    t_nn = np.cumsum(nn_ms) / 1000.0
    cs = CubicSpline(t_nn, nn_ms)
    t_interp = np.arange(0, ibi_cum_s[-1], 1 / fs_out)

    rmssd_interp = np.full_like(t_interp, np.nan, dtype=float)
    for i, t in enumerate(t_interp):
        window_start = t - window_size
        mask = (t_nn > window_start) & (t_nn <= t)
        peaks_in_window = t_nn[mask]

        if len(peaks_in_window) >= 3:
            ibi_in_window = np.diff(peaks_in_window) * 1000.0
            rmssd_interp[i] = np.sqrt(np.mean(np.diff(ibi_in_window) ** 2))

    rmssd_series = pd.Series(rmssd_interp)
    rmssd_interp = (
        rmssd_series.interpolate(method="linear", limit_direction="both")
        .fillna(0.0)
        .values
    )

    stage_interp = np.round(np.interp(t_interp, ibi_cum_s, stage)).astype(int)
    return t_interp, cs(t_interp), stage_interp, nn_ms, t_nn, rmssd_interp


def compute_signal_lag(signal1, signal2, fs, plot=False, label1="", label2=""):
    """Return the integer-sample lag that maximizes the cross-correlation."""
    b, a = signal.butter(2, 0.01 / (fs / 2), btype="high")
    s1 = signal.filtfilt(b, a, signal1.flatten())
    s2 = signal.filtfilt(b, a, signal2.flatten())
    xc = signal.correlate(s1, s2, mode="full")
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    max_lag_samples = np.max(lags) - int(2000 * fs)
    search_mask = lags >= max_lag_samples
    best_idx = np.argmax(xc[search_mask])
    lag = lags[search_mask][best_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lags, xc)
        ax.set_title(f"Cross-correlation - {label1} vs {label2}")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return lag
