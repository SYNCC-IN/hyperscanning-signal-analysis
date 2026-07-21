import re
from copy import deepcopy
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert, resample_poly, welch

from .passive_io_helpers import FS_ATTR_KEYS, infer_fs


def channel_row(ch_name):
    c = ch_name.upper()
    if c.startswith("F"):
        return 0
    if c.startswith("C"):
        return 1
    if c.startswith("P"):
        return 2
    return None


def channel_col(ch_name):
    c = ch_name.upper()
    if "Z" in c:
        return 1
    m = re.search(r"(\d+)$", c)
    if m:
        n = int(m.group(1))
        return 0 if n % 2 == 1 else 2
    if c.endswith("L"):
        return 0
    if c.endswith("R"):
        return 2
    return 1


def channel_sort_key(ch_name):
    c = ch_name.upper()
    col = channel_col(c)
    m = re.search(r"(\d+)$", c)
    idx = int(m.group(1)) if m else 999
    return (col, idx, c)


def node_to_roi(node_name, roi_prefix_map=None):
    """Map an electrode node name to a region of interest by its leading letter."""
    if roi_prefix_map is None:
        roi_prefix_map = {"F": "frontal", "C": "central", "P": "parietal"}
    node = str(node_name).strip()
    if len(node) == 0:
        return "other"
    prefix = node[0].upper()
    return roi_prefix_map.get(prefix, "other")


def build_psd_channel_grid_layout(analysis_channels):
    row_channels = {0: [], 1: [], 2: []}

    for ch in analysis_channels:
        r = channel_row(ch)
        if r is None:
            continue
        row_channels[r].append(ch)

    for r in row_channels:
        row_channels[r] = sorted(row_channels[r], key=channel_sort_key)

    rows = [r for r in (0, 1, 2) if row_channels[r]]
    max_cols = max((len(row_channels[r]) for r in rows), default=0)
    return rows, row_channels, max_cols


def extract_band_envelopes(
    da,
    min_f,
    max_f,
    channels,
    downsample_fs_hz=None,
    truncate_to_event_window=False,
    filter_order=4,
    debug=False,
    psd_resolution_hz=0.05,
    welch_overlap=0.5,
):
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray")

    channel_dim = "channel"
    time_dim = "time"
    if channel_dim not in da.dims or time_dim not in da.dims:
        raise ValueError("da must have dims 'channel' and 'time'")

    if min_f <= 0 or max_f <= min_f:
        raise ValueError("Require 0 < min_f < max_f")
    if not channels:
        raise ValueError("channels list is empty")

    available_channels = da.coords[channel_dim].values
    missing = [ch for ch in channels if ch not in available_channels]
    if missing:
        raise ValueError(f"Channels not found in da: {missing}")

    fs = infer_fs(da.coords[time_dim].values, da.attrs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "Sampling frequency not found in attrs (expected one of: "
            f"{list(FS_ATTR_KEYS)}) and could not be derived from the time axis."
        )

    da_sel = da.sel({channel_dim: channels})
    original_dims = tuple(da_sel.dims)

    da_proc = da_sel.transpose(channel_dim, time_dim)
    x = np.asarray(da_proc.values, dtype=float)

    time_vals = np.asarray(da_proc.coords[time_dim].values, dtype=float)
    if time_vals.size < 3:
        raise ValueError("Need at least 3 time points")

    nyq = fs / 2.0
    if max_f >= nyq:
        raise ValueError(f"max_f ({max_f:.3f} Hz) must be < Nyquist ({nyq:.3f} Hz)")

    wn = [min_f / nyq, max_f / nyq]
    b, a = butter(filter_order, wn, btype="bandpass")

    env = np.full_like(x, np.nan, dtype=float)
    for ci in range(x.shape[0]):
        sig = x[ci, :]
        finite = np.isfinite(sig)
        if not np.any(finite):
            continue

        sig_work = sig.copy()
        if not np.all(finite):
            idx = np.arange(sig.size)
            sig_work[~finite] = np.interp(idx[~finite], idx[finite], sig[finite])

        bp = filtfilt(b, a, sig_work, axis=-1)
        amp = np.abs(hilbert(bp, axis=-1))
        env[ci, :] = amp

    fs_out = fs
    t_out = time_vals

    if downsample_fs_hz is not None:
        downsample_fs_hz = float(downsample_fs_hz)
        if not np.isfinite(downsample_fs_hz) or downsample_fs_hz <= 0:
            raise ValueError("downsample_fs_hz must be a positive finite number")
        if downsample_fs_hz > fs:
            raise ValueError("downsample_fs_hz must be <= input sampling frequency")

        if downsample_fs_hz < fs:
            ratio = Fraction(downsample_fs_hz / fs).limit_denominator(1000)
            env = resample_poly(env, up=ratio.numerator, down=ratio.denominator, axis=1)
            fs_out = fs * ratio.numerator / ratio.denominator
            t0 = float(time_vals[0])
            t_out = t0 + np.arange(env.shape[1], dtype=float) / fs_out

    if truncate_to_event_window:
        ev_dur = da.attrs.get("event_duration", da.attrs.get("event_duration_s", np.nan))
        try:
            ev_dur = float(ev_dur)
        except Exception:
            ev_dur = np.nan
        if not np.isfinite(ev_dur) or ev_dur <= 0:
            raise ValueError(
                "truncate_to_event_window=True requires positive event duration in attrs: "
                "event_duration or event_duration_s"
            )

        keep = (t_out >= 0.0) & (t_out <= ev_dur)
        if not np.any(keep):
            raise ValueError(
                f"No samples remain after truncation to [0, {ev_dur}] s. "
                "Check time coordinates and event metadata."
            )
        env = env[:, keep]
        t_out = t_out[keep]

    env = stats.zscore(env, axis=1, nan_policy="omit")

    env_proc = xr.DataArray(
        env,
        dims=(channel_dim, time_dim),
        coords={channel_dim: da_proc.coords[channel_dim].values, time_dim: t_out},
        attrs={
            "band_min_f_hz": float(min_f),
            "band_max_f_hz": float(max_f),
            "fs_hz": float(fs_out),
            "source_fs_hz": float(fs),
            "method": "bandpass_butter_hilbert_amplitude_zscore",
        },
    )

    if downsample_fs_hz is not None:
        env_proc.attrs["requested_downsample_fs_hz"] = float(downsample_fs_hz)
    env_proc.attrs["truncate_to_event_window"] = int(bool(truncate_to_event_window))

    if truncate_to_event_window:
        env_proc.attrs["event_window_start_s"] = 0.0
        env_proc.attrs["event_window_end_s"] = float(t_out[-1])

    if "child_info" in da.attrs:
        env_proc.attrs["child_info"] = deepcopy(da.attrs["child_info"])

    env_da = env_proc.transpose(*original_dims)

    if debug:
        n_ch = env_proc.sizes[channel_dim]
        fig, axes = plt.subplots(n_ch, 2, figsize=(12, max(2.5 * n_ch, 4.0)), squeeze=False)

        nperseg = int(round(fs_out / psd_resolution_hz))
        nperseg = max(8, min(nperseg, env_proc.sizes[time_dim]))
        noverlap = int(round(welch_overlap * nperseg))
        noverlap = min(max(0, noverlap), nperseg - 1)

        for i, ch in enumerate(env_proc.coords[channel_dim].values):
            y = np.asarray(env_proc.sel({channel_dim: ch}).values, dtype=float)
            t = np.asarray(env_proc.coords[time_dim].values, dtype=float)

            ax_t = axes[i, 0]
            ax_t.plot(t, y, lw=1.0)
            ax_t.set_title(f"{ch} envelope (z-scored)")
            ax_t.set_xlabel("Time [s]")
            ax_t.set_ylabel("Amplitude")
            ax_t.grid(alpha=0.3)

            ax_p = axes[i, 1]
            good = np.isfinite(y)
            if np.sum(good) >= 8:
                f, pxx = welch(y[good], fs=fs_out, nperseg=nperseg, noverlap=noverlap)
                ax_p.plot(f, pxx, lw=1.2)
            else:
                ax_p.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax_p.transAxes)

            ax_p.set_title(f"{ch} envelope PSD (Welch, df~{psd_resolution_hz:.2f} Hz)")
            ax_p.set_xlabel("Frequency [Hz]")
            ax_p.set_ylabel("PSD")
            ax_p.set_xlim((0, fs_out / 2))
            ax_p.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return env_da


def select_common_channels(ch_child, ch_caregiver, preferred=None):
    c1 = set(map(str, ch_child))
    c2 = set(map(str, ch_caregiver))
    common = c1 & c2
    if preferred is None:
        return sorted(common)
    ordered = [ch for ch in preferred if ch in common]
    return ordered
