import re
from pathlib import Path

import numpy as np


CLEAN_RE = re.compile(r"^(W_\d+)_EEG_(ch|cg)_(.+)_cleaned$")


def infer_fs(time_s, attrs):
    fs_attr = attrs.get("sampling_freq", np.nan)
    if np.isfinite(fs_attr) and fs_attr > 0:
        return float(fs_attr)
    if len(time_s) < 2:
        return np.nan
    dt = np.median(np.diff(time_s))
    return float(1.0 / dt) if dt > 0 else np.nan


def get_event_mask(time_s, attrs):
    event_start = float(attrs.get("event_start", 0.0))
    event_duration = float(attrs.get("event_duration", attrs.get("event_duration_s", np.nan)))

    if np.isfinite(event_duration) and event_duration > 0:
        return (time_s >= event_start) & (time_s <= event_start + event_duration)

    if np.any(time_s >= 0):
        return time_s >= 0
    return np.ones_like(time_s, dtype=bool)


def discover_cleaned_role_files(cleaned_signals_folder, target_events, valid_dyads=None, clean_re=None):
    rx = CLEAN_RE if clean_re is None else clean_re
    target_set = set(map(str, target_events))
    valid_set = None if valid_dyads is None else set(map(str, valid_dyads))

    out = []
    for p in sorted(Path(cleaned_signals_folder).rglob("*_cleaned.nc")):
        m = rx.match(p.stem)
        if m is None:
            continue

        dyad_id, role, event = m.group(1), m.group(2), m.group(3)
        if event not in target_set:
            continue
        if valid_set is not None and dyad_id not in valid_set:
            continue

        out.append((p, dyad_id, role, event))

    return out
