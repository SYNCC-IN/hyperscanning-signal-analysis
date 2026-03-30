import json
import os
import re
from datetime import datetime

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal as signal
import xarray as xr
from scipy.interpolate import CubicSpline

from src import dataloader

__all__ = [
    "load_h10_ibi",
    "fix_and_interpolate_ibi",
    "compute_signal_lag",
    "build_h10_ibi_rmssd_xarray",
    "build_h10_ibi_rmssd_xarray_auto",
]


def load_h10_ibi(path: str):
    """Load H10 CSV file and return (stage, computer_timestamps_s, ibi_ms)."""
    data = np.genfromtxt(path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]


def fix_and_interpolate_ibi(
    ibi_ms,
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
        cum_samp, sampling_rate=samp_rate, iterative=True, method="Kubios", show=False
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


def compute_signal_lag(signal1, signal2, plot=False, label1="", label2=""):
    """Return the integer-sample lag that maximizes normalized cross-correlation."""
    s1 = signal1.flatten() - np.mean(signal1)
    s2 = signal2.flatten() - np.mean(signal2)
    xc = signal.correlate(s1, s2, mode="full")
    lags = signal.correlation_lags(s1.size, s2.size, mode="full")
    lag = lags[np.argmax(xc)]

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(lags, xc)
        ax.set_title(f"Cross-correlation - {label1} vs {label2}")
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    return lag


def build_h10_ibi_rmssd_xarray(
    dyad_nr,
    date,
    time_of_recording,
    dev_ch,
    dev_cg,
    data_base_path="../data",
    fs_ibi=8,
    window_size_rmssd_s=30,
    decimate_factor_loader=8,
    decimate_factor_align=16,
    selected_time=(0, 220),
    lowcut=1.0,
    highcut=40.0,
    eeg_filter_type="iir",
    plot=False,
):
    """
    Build aligned H10 IBI/RMSSD xarray (with integer events channel) for one dyad.

    Returns
    -------
    xr.DataArray with dims: (time, channel)
    channels: IBI_CH, IBI_CG, RMSSD_CH, RMSSD_CG, events
    """
    dyad_id = f"W_{dyad_nr}"

    eeg_dir = os.path.join(data_base_path, dyad_id, "eeg")
    path_ch = os.path.join(
        eeg_dir, f"{dyad_id}_{date}_{time_of_recording}_{dev_ch}_IBI.csv"
    )
    path_cg = os.path.join(
        eeg_dir, f"{dyad_id}_{date}_{time_of_recording}_{dev_cg}_IBI.csv"
    )

    stage_ch, _, ibi_ch = load_h10_ibi(path_ch)
    stage_cg, _, ibi_cg = load_h10_ibi(path_cg)

    t_ch_cum_s = np.cumsum(ibi_ch) / 1000.0
    t_cg_cum_s = np.cumsum(ibi_cg) / 1000.0

    _, ibi_ch_i, stage_ch_i, _, _, rmssd_ch_i = fix_and_interpolate_ibi(
        ibi_ch, t_ch_cum_s, stage_ch, fs_out=fs_ibi, window_size=window_size_rmssd_s
    )
    _, ibi_cg_i, stage_cg_i, _, _, rmssd_cg_i = fix_and_interpolate_ibi(
        ibi_cg, t_cg_cum_s, stage_cg, fs_out=fs_ibi, window_size=window_size_rmssd_s
    )

    n = min(len(ibi_ch_i), len(ibi_cg_i))
    ibi_ch_i = ibi_ch_i[:n]
    ibi_cg_i = ibi_cg_i[:n]
    rmssd_ch_i = rmssd_ch_i[:n]
    rmssd_cg_i = rmssd_cg_i[:n]
    stage_ch_i = stage_ch_i[:n]
    stage_cg_i = stage_cg_i[:n]

    mmd = dataloader.create_multimodal_data(
        data_base_path=data_base_path,
        dyad_id=dyad_id,
        load_eeg=True,
        load_et=True,
        lowcut=lowcut,
        highcut=highcut,
        eeg_filter_type=eeg_filter_type,
        interpolate_et_during_blinks_threshold=0.3,
        median_filter_size=64,
        low_pass_et_order=351,
        et_pos_cutoff=128,
        et_pupil_cutoff=4,
        pupil_model_confidence=0.9,
        decimate_factor=decimate_factor_loader,
        plot_flag=False,
    )
    mmd = mmd._decimate_signals(q=decimate_factor_align)

    _, _, ibi_ch_ecg = mmd.get_signals(
        mode="IBI", member="ch", selected_channels=[""], selected_times=list(selected_time)
    )
    _, _, ibi_cg_ecg = mmd.get_signals(
        mode="IBI", member="cg", selected_channels=[""], selected_times=list(selected_time)
    )

    lag_cg = compute_signal_lag(ibi_cg_i, ibi_cg_ecg, plot=plot, label1="H10_cg", label2="ECG_cg")
    lag_ch = compute_signal_lag(ibi_ch_i, ibi_ch_ecg, plot=plot, label1="H10_ch", label2="ECG_ch")

    lag_diff = lag_ch - lag_cg
    if lag_diff > 0:
        ibi_ch_i = ibi_ch_i[lag_diff:]
        ibi_cg_i = ibi_cg_i[:-lag_diff]
        rmssd_ch_i = rmssd_ch_i[lag_diff:]
        rmssd_cg_i = rmssd_cg_i[:-lag_diff]
        stage_ch_i = stage_ch_i[lag_diff:]
        stage_cg_i = stage_cg_i[:-lag_diff]
    elif lag_diff < 0:
        ibi_cg_i = ibi_cg_i[-lag_diff:]
        ibi_ch_i = ibi_ch_i[:lag_diff]
        rmssd_cg_i = rmssd_cg_i[-lag_diff:]
        rmssd_ch_i = rmssd_ch_i[:lag_diff]
        stage_cg_i = stage_cg_i[-lag_diff:]
        stage_ch_i = stage_ch_i[:lag_diff]

    t_h10 = np.arange(len(ibi_cg_i)) / fs_ibi

    if plot:
        import matplotlib.pyplot as plt

        lag_cg_s = lag_cg / fs_ibi
        lag_ch_s = lag_ch / fs_ibi

        start_t = min(lag_ch, lag_cg)
        t_ecg = np.arange(start_t, start_t + len(ibi_ch_ecg)) / fs_ibi

        h10_start, h10_end = float(t_h10[0]), float(t_h10[-1])
        ecg_start, ecg_end = float(t_ecg[0]), float(t_ecg[-1])
        overlap_start = max(h10_start, ecg_start)
        overlap_end = min(h10_end, ecg_end)

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
        axes[0].plot(t_h10, ibi_cg_i, label="H10 CG (aligned)")
        axes[0].plot(t_ecg, ibi_cg_ecg, label="ECG CG", alpha=0.8)
        axes[1].plot(t_h10, ibi_ch_i, label="H10 CH (aligned)")
        axes[1].plot(t_ecg, ibi_ch_ecg, label="ECG CH", alpha=0.8)

        if overlap_end > overlap_start:
            axes[0].set_xlim(overlap_start, overlap_end)

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
        plt.show()
        plt.close(fig)

    timings_path = os.path.join(eeg_dir, f"{dyad_id}_1_25fps.txt")
    with open(timings_path) as f:
        lines = f.readlines()

    df_timings = pd.DataFrame(
        [ln.strip().split("\t") for ln in lines[4:8] if len(ln.strip().split("\t")) >= 7],
        columns=[
            "Label",
            "Start_HH_MM_SS",
            "Start_Sec",
            "End_HH_MM_SS",
            "End_Sec",
            "Duration_HH_MM_SS",
            "Duration_Sec",
        ],
    )
    df_timings[["Start_Sec", "End_Sec", "Duration_Sec"]] = (
        df_timings[["Start_Sec", "End_Sec", "Duration_Sec"]].astype(float)
    )
    required_labels = {"T1", "T2", "T3", "T4"}
    found_labels = set(df_timings["Label"].values)
    missing = required_labels - found_labels
    if missing:
        raise ValueError(
            f"Timing file {timings_path} is missing labels: {sorted(missing)}"
        )
    t1_start = df_timings.loc[df_timings["Label"] == "T1", "Start_Sec"].iat[0]
    df_timings["Start_Sec"] -= t1_start
    df_timings["End_Sec"] -= t1_start

    def _t(label):
        return df_timings.loc[df_timings["Label"] == label, "Start_Sec"].iat[0]

    moments = pd.DataFrame(
        [
            {"moment": "puzzle", "start": _t("T2") + 1.0 * 60, "end": _t("T2") + 2.5 * 60},
            {"moment": "cleaning", "start": _t("T3") - 1.5 * 60, "end": _t("T3")},
            {"moment": "wrong present", "start": _t("T3"), "end": _t("T3") + 1.5 * 60},
            {"moment": "surprise", "start": _t("T4"), "end": _t("T4") + 1.5 * 60},
        ]
    )

    v = min(np.min(np.where(stage_cg_i == 2)), np.min(np.where(stage_ch_i == 2)))
    t_h10 = t_h10 - t_h10[v]

    event_code_map = {
        "baseline": 0,
        "puzzle": 1,
        "cleaning": 2,
        "wrong present": 3,
        "surprise": 4,
    }
    events_signal = np.zeros_like(t_h10, dtype=int)
    event_windows_s = {}

    for _, row in moments.iterrows():
        name = row["moment"]
        code = event_code_map.get(name, 0)
        start = float(row["start"])
        end = float(row["end"])
        event_windows_s[name] = {"start_s": start, "end_s": end, "code": int(code)}
        mask = (t_h10 >= start) & (t_h10 <= end)
        events_signal[mask] = code

    channels = ["IBI_CH", "IBI_CG", "RMSSD_CH", "RMSSD_CG", "events"]
    data_array = np.vstack([ibi_ch_i, ibi_cg_i, rmssd_ch_i, rmssd_cg_i, events_signal]).T

    out = xr.DataArray(
        data=data_array,
        coords={"time": t_h10, "channel": channels},
        dims=["time", "channel"],
        name="H10_IBI_RMSSD_events",
        attrs={
            "sampling_frequency_Hz": fs_ibi,
            "dyad_id": dyad_id,
            "window_size_RMSSD_s": window_size_rmssd_s,
            "device_CH": dev_ch,
            "device_CG": dev_cg,
            "recording_date": date,
            "recording_time": time_of_recording,
            "units_IBI": "ms",
            "units_RMSSD": "ms",
            "units_events": "integer_code",
            "event_code_map_json": json.dumps(event_code_map, ensure_ascii=False),
            "event_windows_s_json": json.dumps(event_windows_s, ensure_ascii=False),
            "description": "Aligned H10 IBI, sliding-window RMSSD, and integer-coded events channel",
        },
    )

    return out


def _autodetect_latest_h10_recording(dyad_nr, data_base_path="../data"):
    """Return (date, time_of_recording, device_ids) for the latest dyad H10 IBI recording pair."""
    dyad_id = f"W_{dyad_nr}"
    eeg_dir = os.path.join(data_base_path, dyad_id, "eeg")
    if not os.path.isdir(eeg_dir):
        raise FileNotFoundError(f"EEG folder not found: {eeg_dir}")

    pattern = re.compile(
        rf"^{dyad_id}_(\d{{2}}_\d{{2}}_\d{{4}})_(\d{{2}}_\d{{2}})_([A-Za-z0-9]+)_IBI\.csv$"
    )
    groups = {}

    for fn in os.listdir(eeg_dir):
        m = pattern.match(fn)
        if not m:
            continue
        date, rec_time, dev = m.groups()
        key = (date, rec_time)
        groups.setdefault(key, set()).add(dev)

    valid_groups = [(k, sorted(v)) for k, v in groups.items() if len(v) >= 2]
    if not valid_groups:
        raise FileNotFoundError(f"No paired H10 IBI files found for {dyad_id} in {eeg_dir}")

    valid_groups.sort(key=lambda kv: datetime.strptime(f"{kv[0][0]}_{kv[0][1]}", "%d_%m_%Y_%H_%M"))
    (date, rec_time), devices = valid_groups[-1]
    return date, rec_time, devices


def build_h10_ibi_rmssd_xarray_auto(
    dyad_nr,
    data_base_path="../data",
    fs_ibi=8,
    window_size_rmssd_s=30,
    decimate_factor_loader=8,
    decimate_factor_align=16,
    selected_time=(0, 220),
    lowcut=1.0,
    highcut=40.0,
    eeg_filter_type="iir",
    plot=False,
    preferred_dev_ch=None,
    preferred_dev_cg=None,
):
    """
    Ultra-short wrapper: only dyad_nr is required.
    Auto-detects latest recording date/time and CH/CG device IDs, then builds the xarray.
    """
    date, rec_time, devices = _autodetect_latest_h10_recording(
        dyad_nr, data_base_path=data_base_path
    )

    dev_ch = preferred_dev_ch if preferred_dev_ch in devices else None
    dev_cg = preferred_dev_cg if preferred_dev_cg in devices else None

    remaining = [d for d in devices if d not in {dev_ch, dev_cg}]
    if dev_ch is None:
        dev_ch = remaining.pop(0)
    if dev_cg is None:
        if remaining:
            dev_cg = remaining.pop(0)
        else:
            dev_cg = [d for d in devices if d != dev_ch][0]

    print(
        f"Auto-detected latest recording for W_{dyad_nr}: "
        f"{date} {rec_time}, CH={dev_ch}, CG={dev_cg}"
    )

    return build_h10_ibi_rmssd_xarray(
        dyad_nr=dyad_nr,
        date=date,
        time_of_recording=rec_time,
        dev_ch=dev_ch,
        dev_cg=dev_cg,
        data_base_path=data_base_path,
        fs_ibi=fs_ibi,
        window_size_rmssd_s=window_size_rmssd_s,
        decimate_factor_loader=decimate_factor_loader,
        decimate_factor_align=decimate_factor_align,
        selected_time=selected_time,
        lowcut=lowcut,
        highcut=highcut,
        eeg_filter_type=eeg_filter_type,
        plot=plot,
    )
