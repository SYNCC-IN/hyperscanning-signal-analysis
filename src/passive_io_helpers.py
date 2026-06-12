import re
from pathlib import Path

import numpy as np


# Attribute keys that may carry the sampling frequency, in priority order.
FS_ATTR_KEYS = ("sampling_freq", "fs", "sfreq", "fs_hz")





def infer_fs(time_s, attrs, keys=FS_ATTR_KEYS):
    """Infer sampling frequency from attrs (trying ``keys``) then the time axis.

    Returns the first positive finite value found among ``keys`` in ``attrs``;
    otherwise derives it from the median spacing of ``time_s``; otherwise NaN.
    """
    for key in keys:
        val = attrs.get(key, np.nan)
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            return v

    time_s = np.asarray(time_s, dtype=float)
    if time_s.size < 2:
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

def _role_file_regex(signal_type, cleaned=False):
    """Compile the exported role-file stem pattern for a given signal type.

    Matches ``W_<id>_<SIGNAL>_(ch|cg)_<event>`` (with a trailing ``_cleaned``
    when ``cleaned`` is True), capturing ``(dyad, role, event)``. ``signal_type``
    is e.g. ``"EEG"``, ``"IBI"`` or ``"RMSSD"``.
    """
    suffix = r"_cleaned" if cleaned else ""
    return re.compile(rf"^(W_\d+)_{re.escape(str(signal_type))}_(ch|cg)_(.+){suffix}$")

def discover_role_files(
    folder,
    target_events,
    signal_type="EEG",
    glob_pattern="*.nc",
    valid_dyads=None,
    cleaned=False,
):
    """Walk ``folder`` and return ``(path, dyad_id, role, event)`` role files.

    The stem pattern is built from ``signal_type`` (e.g. ``"EEG"``, ``"IBI"``,
    ``"RMSSD"``); pass ``cleaned=True`` to match ``*_cleaned`` outputs. Files
    whose stem does not match, whose event is not in ``target_events``, or whose
    dyad is not in ``valid_dyads`` (when provided) are skipped.
    """
    role_re = _role_file_regex(signal_type, cleaned=cleaned)
    target_set = set(map(str, target_events))
    valid_set = None if valid_dyads is None else set(map(str, valid_dyads))

    out = []
    for p in sorted(Path(folder).rglob(glob_pattern)):
        m = role_re.match(p.stem)
        if m is None:
            continue

        dyad_id, role, event = m.group(1), m.group(2), m.group(3)
        if event not in target_set:
            continue
        if valid_set is not None and dyad_id not in valid_set:
            continue

        out.append((p, dyad_id, role, event))

    return out


def discover_cleaned_role_files(cleaned_signals_folder, target_events, signal_type="EEG", valid_dyads=None):
    """Discover cleaned ``*_cleaned.nc`` role files (see :func:`discover_role_files`).
    This is for EEG files cleand by ICA"""
    return discover_role_files(
        cleaned_signals_folder,
        target_events,
        signal_type=signal_type,
        glob_pattern="*_cleaned.nc",
        valid_dyads=valid_dyads,
        cleaned=True,
    )


def build_role_lookup(role_files):
    """Group ``(path, dyad, role, event)`` tuples into ``{(dyad, event): {role: path}}``."""
    lookup = {}
    for p, dyad_id, role, event in role_files:
        lookup.setdefault((dyad_id, event), {})[role] = p
    return lookup


def pairs_from_lookup(lookup):
    """Return sorted ``(dyad, event, ch_path, cg_path)`` for keys having both roles."""
    out = []
    for (dyad_id, event), roles in sorted(lookup.items()):
        if "ch" in roles and "cg" in roles:
            out.append((dyad_id, event, roles["ch"], roles["cg"]))
    return out
