import os
import re
from datetime import datetime

import numpy as np


def _resolve_eeg_directory(data_base_path, dyad_id):
    """Resolve dyad EEG directory supporting both EEG and eeg names."""
    candidates = [
        os.path.join(data_base_path, dyad_id, "EEG"),
        os.path.join(data_base_path, dyad_id, "eeg"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"EEG folder not found for {dyad_id}. Checked: {candidates}."
    )


def load_h10_ibi(path: str):
    """Load H10 CSV file and return (stage, computer_timestamps_s, ibi_ms)."""
    data = np.genfromtxt(path, delimiter=",")
    return data[:, 0], data[:, 1], data[:, 2]


def resolve_h10_ibi_pair_paths(
    data_base_path,
    dyad_id,
    date,
    time_of_recording,
    dev_ch,
    dev_cg,
):
    """Resolve child/caregiver H10 IBI paths with timestamped and simple fallbacks."""
    eeg_dir = _resolve_eeg_directory(data_base_path, dyad_id)
    ch_candidates = []
    cg_candidates = []

    if date and time_of_recording:
        ch_candidates.append(
            os.path.join(
                eeg_dir,
                f"{dyad_id}_{date}_{time_of_recording}_{dev_ch}_IBI.csv",
            )
        )
        cg_candidates.append(
            os.path.join(
                eeg_dir,
                f"{dyad_id}_{date}_{time_of_recording}_{dev_cg}_IBI.csv",
            )
        )

    ch_candidates.append(os.path.join(eeg_dir, f"{dyad_id}_{dev_ch}_IBI.csv"))
    cg_candidates.append(os.path.join(eeg_dir, f"{dyad_id}_{dev_cg}_IBI.csv"))

    path_ch = next((p for p in ch_candidates if os.path.exists(p)), None)
    path_cg = next((p for p in cg_candidates if os.path.exists(p)), None)

    if path_ch is None or path_cg is None:
        raise FileNotFoundError(
            f"Could not locate paired H10 IBI files for {dyad_id}. "
            f"Checked CH={ch_candidates}, CG={cg_candidates}."
        )

    return path_ch, path_cg


def autodetect_latest_h10_recording(dyad_nr, data_base_path="../data"):
    """Return (date, time_of_recording, device_ids) for latest dyad H10 IBI pair."""
    dyad_id = f"W_{str(dyad_nr).zfill(3)}"
    eeg_dir = _resolve_eeg_directory(data_base_path, dyad_id)

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
        simple_pattern = re.compile(rf"^{dyad_id}_([A-Za-z0-9]+)_IBI\.csv$")
        simple_devices = []
        for fn in os.listdir(eeg_dir):
            m = simple_pattern.match(fn)
            if m:
                full_path = os.path.join(eeg_dir, fn)
                if os.path.getsize(full_path) > 0:
                    simple_devices.append(m.group(1))
        simple_devices = sorted(set(simple_devices))
        if len(simple_devices) >= 2:
            return None, None, simple_devices
        raise FileNotFoundError(f"No paired H10 IBI files found for {dyad_id} in {eeg_dir}")

    valid_groups.sort(
        key=lambda kv: datetime.strptime(f"{kv[0][0]}_{kv[0][1]}", "%d_%m_%Y_%H_%M")
    )
    (date, rec_time), devices = valid_groups[-1]
    return date, rec_time, devices


def align_h10_pairs_by_lag(
    ibi_ch_i,
    ibi_cg_i,
    rmssd_ch_i,
    rmssd_cg_i,
    stage_ch_i,
    stage_cg_i,
    lag_diff,
):
    """Align child/caregiver vectors by relative lag difference."""
    if lag_diff > 0:
        return (
            ibi_ch_i[lag_diff:],
            ibi_cg_i[:-lag_diff],
            rmssd_ch_i[lag_diff:],
            rmssd_cg_i[:-lag_diff],
            stage_ch_i[lag_diff:],
            stage_cg_i[:-lag_diff],
        )

    if lag_diff < 0:
        return (
            ibi_ch_i[:lag_diff],
            ibi_cg_i[-lag_diff:],
            rmssd_ch_i[:lag_diff],
            rmssd_cg_i[-lag_diff:],
            stage_ch_i[:lag_diff],
            stage_cg_i[-lag_diff:],
        )

    return ibi_ch_i, ibi_cg_i, rmssd_ch_i, rmssd_cg_i, stage_ch_i, stage_cg_i
