import json

import numpy as np
import pandas as pd
import xarray as xr

from . import dataloader
from .secore_helpers import (
    align_h10_pairs_by_lag,
    autodetect_latest_h10_recording,
    load_h10_ibi,
    resolve_h10_ibi_pair_paths,
)
from .secore_signal_helpers import (
    compute_signal_lag,
    fix_and_interpolate_ibi,
)
from .signal_plots import plot_h10_ecg_alignment

__all__ = [
    "load_h10_ibi",
    "fix_and_interpolate_ibi",
    "compute_signal_lag",
    "build_h10_ibi_rmssd_xarray",
]


def build_h10_ibi_rmssd_xarray(
    dyad_nr,
    date,
    time_of_recording,
    dev_ch,
    dev_cg,
    video_timings, # dictionary with time to allign with the momentsbulit based on the timings_secore_hrv.csv in the interaction, in seconds (relative to the start of the recording)
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
    save_dir=None,
):
    """
    Build aligned H10 IBI/RMSSD xarray (with integer events channel) for one dyad.

    Returns
    -------
    xr.DataArray with dims: (time, channel)
    channels: IBI_CH, IBI_CG, RMSSD_CH, RMSSD_CG, events
    """
    dyad_id = f"W_{str(dyad_nr).zfill(3)}"

    path_ch, path_cg = resolve_h10_ibi_pair_paths(
        data_base_path=data_base_path,
        dyad_id=dyad_id,
        date=date,
        time_of_recording=time_of_recording,
        dev_ch=dev_ch,
        dev_cg=dev_cg,
    )

    stage_ch, _, ibi_ch = load_h10_ibi(path_ch)
    stage_cg, _, ibi_cg = load_h10_ibi(path_cg)

    t_ch_cum_s = np.cumsum(ibi_ch) / 1000.0
    t_cg_cum_s = np.cumsum(ibi_cg) / 1000.0

    _, ibi_ch_i, stage_ch_i, _, _, rmssd_ch_i = fix_and_interpolate_ibi(
        t_ch_cum_s, stage_ch, fs_out=fs_ibi, window_size=window_size_rmssd_s
    )
    _, ibi_cg_i, stage_cg_i, _, _, rmssd_cg_i = fix_and_interpolate_ibi(
        t_cg_cum_s, stage_cg, fs_out=fs_ibi, window_size=window_size_rmssd_s
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
        load_et=False,
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

    lag_cg = compute_signal_lag(ibi_cg_i, ibi_cg_ecg, fs=fs_ibi, plot=plot, label1="H10_cg", label2="ECG_cg")
    lag_ch = compute_signal_lag(ibi_ch_i, ibi_ch_ecg, fs=fs_ibi, plot=plot, label1="H10_ch", label2="ECG_ch")
    print(f"Computed lags (in seconds) to ECG: CG={lag_cg/fs_ibi}, CH={lag_ch/fs_ibi}")

    lag_diff = lag_ch - lag_cg
    (
        ibi_ch_i,
        ibi_cg_i,
        rmssd_ch_i,
        rmssd_cg_i,
        stage_ch_i,
        stage_cg_i,
    ) = align_h10_pairs_by_lag(
        ibi_ch_i=ibi_ch_i,
        ibi_cg_i=ibi_cg_i,
        rmssd_ch_i=rmssd_ch_i,
        rmssd_cg_i=rmssd_cg_i,
        stage_ch_i=stage_ch_i,
        stage_cg_i=stage_cg_i,
        lag_diff=lag_diff,
    )

    t_h10 = np.arange(len(ibi_cg_i)) / fs_ibi

    if plot:
        plot_h10_ecg_alignment(
            t_h10=t_h10,
            ibi_cg_i=ibi_cg_i,
            ibi_ch_i=ibi_ch_i,
            ibi_cg_ecg=ibi_cg_ecg,
            ibi_ch_ecg=ibi_ch_ecg,
            lag_cg=lag_cg,
            lag_ch=lag_ch,
            fs_ibi=fs_ibi,
            dyad_id=dyad_id,
            save_dir=save_dir,
        )
    # Load timing annotations and define event windows based on T1–T4, which mark key moments in the interaction.
    # timings_path = os.path.join(eeg_dir, f"{dyad_id}_1_25fps.txt")
    # with open(timings_path) as f:
    #     lines = f.readlines()

    # # Use regex to find timing rows (T1–T4) regardless of how many camera
    # # header lines precede them.  Truncate to 7 columns to drop optional
    # # annotator comments placed in column 8+.
    # _timing_re = re.compile(r"^T\d\t")
    # timing_rows = [
    #     ln.strip().split("\t")[:7]
    #     for ln in lines
    #     if _timing_re.match(ln) and len(ln.strip().split("\t")) >= 7
    # ]

    # df_timings = pd.DataFrame(
    #     timing_rows,
    #     columns=[
    #         "Label",
    #         "Start_HH_MM_SS",
    #         "Start_Sec",
    #         "End_HH_MM_SS",
    #         "End_Sec",
    #         "Duration_HH_MM_SS",
    #         "Duration_Sec",
    #     ],
    # )
    # df_timings[["Start_Sec", "End_Sec", "Duration_Sec"]] = (
    #     df_timings[["Start_Sec", "End_Sec", "Duration_Sec"]].astype(float)
    # )
    # required_labels = {"T1", "T2", "T3", "T4"}
    # found_labels = set(df_timings["Label"].values)
    # missing = required_labels - found_labels
    # if missing:
    #     raise ValueError(
    #         f"Timing file {timings_path} is missing labels: {sorted(missing)}"
    #    )
    #     t1_start = df_timings.loc[df_timings["Label"] == "T1", "Start_Sec"].iat[0]

    
    # df_timings["Start_Sec"] -= t1_start
    # df_timings["End_Sec"] -= t1_start

    # def _t(label):
    #     return df_timings.loc[df_timings["Label"] == label, "Start_Sec"].iat[0]

    moments = pd.DataFrame(
        [
            {"moment": "puzzle", "start": video_timings["T2"] + 1.0 * 60, "end": video_timings["T2"] + 2.5 * 60},
            {"moment": "cleaning", "start": video_timings["T3"] - 1.5 * 60, "end": video_timings["T3"]},
            {"moment": "wrong present", "start": video_timings["T3"], "end": video_timings["T3"] + 1.5 * 60},
            {"moment": "surprise", "start": video_timings["T4"], "end": video_timings["T4"] + 1.5 * 60},
        ]
    )

    idx_stage_cg = np.where(stage_cg_i == 2)[0]
    idx_stage_ch = np.where(stage_ch_i == 2)[0]
    if idx_stage_cg.size == 0 or idx_stage_ch.size == 0:
        raise ValueError(
            "Cannot align to stage 2: no samples with stage == 2 found in one or both channels."
        )
    v = int(min(idx_stage_cg.min(), idx_stage_ch.min()))
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

