import os

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


def plot_h10_ecg_alignment(
    t_h10,
    ibi_cg_i,
    ibi_ch_i,
    ibi_cg_ecg,
    ibi_ch_ecg,
    lag_cg,
    lag_ch,
    fs_ibi,
    dyad_id,
    save_dir=None,
):
    """Plot full-range and overlap-zoomed H10-vs-ECG alignment diagnostics."""
    lag_cg_s = lag_cg / fs_ibi
    lag_ch_s = lag_ch / fs_ibi

    start_t = min(lag_ch, lag_cg)
    t_ecg = np.arange(start_t, start_t + len(ibi_ch_ecg)) / fs_ibi

    h10_start, h10_end = float(t_h10[0]), float(t_h10[-1])
    ecg_start, ecg_end = float(t_ecg[0]), float(t_ecg[-1])
    overlap_start = max(h10_start, ecg_start)
    overlap_end = min(h10_end, ecg_end)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), dpi=100)
    axes[0].plot(t_h10, ibi_cg_i, label="H10 CG (aligned)")
    axes[0].plot(t_ecg, ibi_cg_ecg, label="ECG CG", alpha=0.8)
    axes[1].plot(t_h10, ibi_ch_i, label="H10 CH (aligned)")
    axes[1].plot(t_ecg, ibi_ch_ecg, label="ECG CH", alpha=0.8)
    axes[0].set_ylabel("IBI [ms]")
    axes[1].set_ylabel("IBI [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"CG alignment (lag={lag_cg} samples, {lag_cg_s:+.2f} s)")
    axes[1].set_title(f"CH alignment (lag={lag_ch} samples, {lag_ch_s:+.2f} s)")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Alignment check: full time range", y=1.02)
    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{dyad_id}_alignment_fullrange.png"),
            bbox_inches="tight",
            dpi=100,
        )
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), dpi=100)
    axes[0].plot(t_h10, ibi_cg_i, label="H10 CG (aligned)")
    axes[0].plot(t_ecg, ibi_cg_ecg, label="ECG CG", alpha=0.8)
    axes[1].plot(t_h10, ibi_ch_i, label="H10 CH (aligned)")
    axes[1].plot(t_ecg, ibi_ch_ecg, label="ECG CH", alpha=0.8)

    if overlap_end > overlap_start:
        axes[0].set_xlim(overlap_start, overlap_end)
        axes[1].set_xlim(overlap_start, overlap_end)
        axes[0].autoscale_view(scalex=False, scaley=True)
        axes[1].autoscale_view(scalex=False, scaley=True)

    axes[0].set_ylabel("IBI [ms]")
    axes[1].set_ylabel("IBI [ms]")
    axes[1].set_xlabel("Time [s]")
    axes[0].set_title(f"CG alignment (lag={lag_cg} samples, {lag_cg_s:+.2f} s)")
    axes[1].set_title(f"CH alignment (lag={lag_ch} samples, {lag_ch_s:+.2f} s)")
    axes[0].legend()
    axes[1].legend()

    if overlap_end > overlap_start:
        fig.suptitle(
            f"Alignment check: zoomed to overlap [{overlap_start:.2f}, {overlap_end:.2f}] s",
            y=1.02,
        )
    else:
        fig.suptitle(
            "Alignment check: no overlap between displayed H10 and ECG ranges",
            y=1.02,
        )

    plt.tight_layout()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, f"{dyad_id}_alignment_zoomed.png"),
            bbox_inches="tight",
            dpi=100,
        )
    plt.show()
