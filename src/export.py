"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import os
import json
import numbers
import warnings
from dataclasses import asdict, is_dataclass
from typing import Optional, TYPE_CHECKING

import joblib
import xarray as xr

from . import dataloader
from .data_structures import MultimodalData

if TYPE_CHECKING:
    import mne
    import numpy as np
    import pandas as pd


def _sanitize_netcdf_attr_value(value):
    if value is None:
        return ""

    if isinstance(value, (str, bytes, bool, int, float, numbers.Number)):
        return value

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple)):
        has_nested = any(isinstance(v, (dict, list, tuple)) for v in value)
        if has_nested:
            return json.dumps(value, ensure_ascii=False, default=str)
        return ["" if v is None else v for v in value]

    if hasattr(value, "tolist"):
        converted = value.tolist()
        return _sanitize_netcdf_attr_value(converted)

    return str(value)


def _sanitize_netcdf_attrs_inplace(attrs: dict) -> None:
    for key in list(attrs.keys()):
        attrs[key] = _sanitize_netcdf_attr_value(attrs[key])


def _try_decode_json_attr_value(value):
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    return decoded


def _decode_json_attrs_inplace(attrs: dict) -> None:
    for key in list(attrs.keys()):
        attrs[key] = _try_decode_json_attr_value(attrs[key])


def _dataclass_or_dict(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    return None


def _build_export_metadata(multimodal_data, selected_modality):
    metadata = {
        "notes": multimodal_data.notes,
        "child_info": _dataclass_or_dict(multimodal_data.child_info),
    }
    if selected_modality == 'EEG':
        metadata["eeg"] = {
            "filtration": _dataclass_or_dict(multimodal_data.eeg_filtration),
            "references": multimodal_data.references,
        }
    return metadata

def export_to_xarray(multimodal_data, selected_event, selected_channels, selected_modality, member, time_margin, verbose=True, logger: Optional[object] = None):
    '''Export selected signals from a MultimodalData instance to an xarray DataArray.
    Args:
        multimodal_data: The MultimodalData instance containing the data.
        selected_event: The name of the event to select (e.g., 'Incredibles').
        selected_channels: List of channel names to include in the export (e.g., ['Fp1', 'Fp2'] for EEG).
        selected_modality: The modality to export (e.g., 'EEG', 'ECG', 'ET', 'IBI', or 'diode').
        member: The member to select ('ch' or 'cg').
        time_margin: Margin in seconds to include before and after the event.
        verbose: If True, emit export progress messages.
        logger: Optional logger-like object with .info(str). If provided and verbose=True,
            messages are sent to logger.info instead of print.
    Returns:
        An xarray DataArray containing the selected signals for the specified event and modality, with time reset to 0 at the start of the event and metadata included as attributes.   
        The DataArray will have dimensions 'time' and 'channel', and coordinates corresponding to the time points and channel names. Metadata attributes will include information about the dyad, member, sampling frequency, event details, and any additional notes or child information from the MultimodalData instance.
    '''
    if selected_event not in multimodal_data.events:
        raise ValueError(f"Event '{selected_event}' not found. Available events: {list(multimodal_data.events.keys())}")

    ev = multimodal_data.events[selected_event]
    event_start = ev['start']
    event_end = ev['start'] + ev['duration']

    # find time range covering selected event with margin on both sides
    recording_start = multimodal_data.data['time'].min()
    recording_end = multimodal_data.data['time'].max()

    selected_time = [
        max(recording_start, event_start - time_margin),
        min(recording_end, event_end + time_margin),
    ]

    if verbose:
        msg_1 = f"Event '{selected_event}' starts at {event_start:.2f}s and ends at {event_end:.2f}s"
        msg_2 = f"Selected time range with ±{time_margin}s margin: {selected_time[0]:.2f}s to {selected_time[1]:.2f}s"
        if logger is not None:
            logger.info(msg_1)
            logger.info(msg_2)
        else:
            print(msg_1)
            print(msg_2)

    signals = multimodal_data.get_signals(
        mode=selected_modality,
        member=member,
        selected_channels=selected_channels,
        selected_times=selected_time
    )
    if signals is None:
        raise ValueError(
            f"No signals available for modality='{selected_modality}', member='{member}', "
            f"channels={selected_channels}, event='{selected_event}'."
        )
    time, channels, data = signals

    # convert the retrieved data to xarray DataArray, resetting time to 0 at event start
    time = time - event_start
    # strip channel names to remove EEG_{member}_ prefix if modality is EEG
    if selected_modality == 'EEG':
        channels = [ch.replace(f'EEG_{member}_', '') for ch in channels]
    elif selected_modality == 'ET':
        channels = [ch.replace(f'ET_{member}_', '') for ch in channels]
    elif selected_modality == 'IBI':
        channels = [ch.replace(f'IBI_{member}', 'IBI') for ch in channels]
    elif selected_modality == 'ECG':
        channels = [ch.replace(f'ECG_{member}', 'ECG') for ch in channels]
    elif selected_modality == 'diode':
        channels = ['diode']

    channel_names = [str(ch) for ch in channels]

    data_xr = xr.DataArray(
        data,
        coords=[time, channels],
        dims=['time', 'channel'],
        name='signals'
    )

    metadata = _build_export_metadata(multimodal_data, selected_modality)

    data_xr.attrs.update({
        'dyad_id': multimodal_data.id,
        'who': member,
        'sampling_freq': float(multimodal_data.fs),
        'event_name': selected_event,
        'event_start': 0.0,
        'event_duration': float(event_end - event_start),
        'time_margin_s': float(time_margin),
        'channel_names_csv': ','.join(channel_names),
        'channel_names_json': json.dumps(channel_names, ensure_ascii=True),
        'metadata_json': json.dumps(metadata, ensure_ascii=False, default=str),
    })

    _sanitize_netcdf_attrs_inplace(data_xr.attrs)
    return data_xr


def write_dyad_to_uniwaw_imported(dyad_id_list=None, load_eeg=True, load_et=True, load_meta=True, lowcut=1.0, highcut=40.0, eeg_filter_type='fir',decimate_factor=8, plot_flag=False, time_margin=10, input_data_path="../data", export_path="../data/UNIWAW_imported", verbose=False, logger: Optional[object] = None):
    '''Export signals from a specified dyad to xarray DataArrays and save them as NetCDF files in a structured directory format compatible with UNIWAW_imported.
    Args:
        dyad_id_list: List of the IDs of the dyads to export (e.g., ['W_003']). If None, a ValueError is raised'
        load_eeg: Whether to load EEG data for the dyad.
        load_et: Whether to load eye-tracking data for the dyad.
        load_meta: Whether to load metadata for the dyad.
        lowcut: The low cut frequency for EEG filtering.
        highcut: The high cut frequency for EEG filtering.
        eeg_filter_type: The type of EEG filter to use ('fir' or 'iir').
        decimate_factor: The factor by which to decimate the EEG data.
        plot_flag: Whether to plot the data during processing.
        time_margin: The time margin to include around events.
        input_data_path: The path to the input data directory.
        export_path: The path to the export directory.
        verbose: If True, emit progress messages during export.
        logger: Optional logger-like object with .info(str). If provided and verbose=True,
            messages are sent to logger.info instead of print.
        '''
    def _log(message: str) -> None:
        if not verbose:
            return
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    if dyad_id_list is None:
        raise ValueError("dyad_id_list must be provided")
    if isinstance(dyad_id_list, str):
        dyad_id_list = [dyad_id_list]
    if not isinstance(dyad_id_list, list) or len(dyad_id_list) == 0:
        raise ValueError("dyad_id_list must be a non-empty list of dyad IDs to export (e.g., ['W_003']).")
    members = {'ch': 'child', 'cg': 'caregiver'}
    selected_channels = {
        'EEG': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5', 'P3', 'Pz',
                'P4', 'T6', 'O1', 'O2'],
        'ET': ['x', 'y', 'pupil', 'blinks'],
        'ECG': ['ECG'],
        'IBI': ['IBI']}
    for dyad_id in dyad_id_list:
        _log(f"Loading dyad '{dyad_id}' from '{input_data_path}'")
        multimodal_data = dataloader.create_multimodal_data(data_base_path = input_data_path,
                                                    dyad_id = dyad_id,
                                                    load_eeg=load_eeg,
                                                    load_et=load_et,
                                                    load_meta=load_meta,
                                                    lowcut=lowcut,
                                                    highcut=highcut,
                                                    eeg_filter_type=eeg_filter_type,
                                                    interpolate_et_during_blinks_threshold=0.3,
                                                    median_filter_size=64,
                                                    low_pass_et_order=351,
                                                    et_pos_cutoff=128,
                                                    et_pupil_cutoff=4,
                                                    pupil_model_confidence=0.9,
                                                    decimate_factor=decimate_factor,
                                                    plot_flag=plot_flag)
        _log(f"Loaded dyad '{multimodal_data.id}'. Export root: '{export_path}'")
        for modality in multimodal_data.modalities:
            path_modality = os.path.join(export_path, modality,str(multimodal_data.id))
            if not os.path.exists(path_modality):
                os.makedirs(path_modality)
            for who, member in members.items():
                path_member = os.path.join(path_modality, member)
                if not os.path.exists(path_member):
                    os.makedirs(path_member)
                for event in multimodal_data.events.keys():
                    _log(f"Exporting modality='{modality}', member='{who}', event='{event}'")
                    data_xr = export_to_xarray(multimodal_data=multimodal_data,
                                                selected_event=event,
                                                selected_channels=selected_channels.get(modality),
                                                selected_modality=modality,
                                                member=who,
                                                time_margin=time_margin,
                                                verbose=False,
                                                logger=logger)
                    file_path = os.path.join(path_member, f'{multimodal_data.id}_{modality}_{who}_{event}.nc')
                    data_xr.to_netcdf(file_path, engine='netcdf4', format='NETCDF4_CLASSIC')
                    _log(f"Saved: {file_path}")

        _log(f"Finished export for dyad '{multimodal_data.id}'")





def load_xarray_from_netcdf(filename: str, decode_json_attrs: bool = True) -> xr.DataArray:
    """Load DataArray from a NetCDF file with optional JSON attribute decoding.

    Args:
        filename: Path to the NetCDF file.
        decode_json_attrs: If True, decode JSON-serialized attribute strings
            (typically dict/list values serialized during export).

    Returns:
        xarray.DataArray: Loaded DataArray.
    """
    data_xr = xr.load_dataarray(filename)
    if decode_json_attrs:
        _decode_json_attrs_inplace(data_xr.attrs)
    return data_xr


def get_export_metadata(data_xr: xr.DataArray) -> dict:
    """Get structured export metadata from a DataArray attrs payload.

    Args:
        data_xr: Exported DataArray that may contain ``metadata_json`` attr.

    Returns:
        dict: Parsed metadata dictionary, or empty dict if unavailable/invalid.
    """
    raw_metadata = data_xr.attrs.get("metadata_json")
    decoded_metadata = _try_decode_json_attr_value(raw_metadata)
    if isinstance(decoded_metadata, dict):
        return decoded_metadata
    return {}


def _infer_sfreq_from_time_coord(time_coord: "np.ndarray") -> float:
    import numpy as np

    if time_coord.size < 2:
        raise ValueError("Cannot infer sampling frequency: time coordinate has fewer than 2 samples.")

    dt = np.diff(time_coord.astype(float))
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling frequency: time deltas are invalid.")

    return float(1.0 / np.median(dt))


def load_eeg_ncdf_as_mne_raw(
    ncdf_path: str,
    montage: Optional[str] = "standard_1020",
    scale_to_volts: float = 1e-6,
    data_xr: Optional[xr.DataArray] = None,
) -> "mne.io.RawArray":
    """Load an exported EEG NetCDF file and convert it to MNE RawArray.

    Args:
        ncdf_path: Path to exported EEG NetCDF file.
        montage: MNE montage name. If None, montage is not set.
        scale_to_volts: Multiplicative scale to convert values to volts.
        data_xr: Pre-loaded DataArray. If provided, the file is not loaded again.

    Returns:
        mne.io.RawArray: Continuous EEG signal in MNE format.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG quality analysis.") from exc

    import numpy as np

    if data_xr is None:
        data_xr = load_xarray_from_netcdf(ncdf_path)

    if not isinstance(data_xr, xr.DataArray):
        raise TypeError(f"Expected xarray.DataArray in '{ncdf_path}', got {type(data_xr)}")

    if "time" not in data_xr.dims or "channel" not in data_xr.dims:
        raise ValueError(
            f"Expected DataArray with 'time' and 'channel' dims, got: {data_xr.dims}"
        )

    data_t = data_xr.transpose("channel", "time")
    ch_names = [str(ch) for ch in data_t.coords["channel"].values]
    data_values = np.asarray(data_t.values, dtype=float) * float(scale_to_volts)

    sfreq_attr = data_xr.attrs.get("sampling_freq")
    if sfreq_attr is None or (isinstance(sfreq_attr, str) and not sfreq_attr.strip()):
        sfreq = _infer_sfreq_from_time_coord(np.asarray(data_xr.coords["time"].values))
    else:
        sfreq = float(sfreq_attr)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(data_values, info, verbose=False)

    if montage:
        try:
            raw.set_montage(montage, on_missing="ignore")
        except ValueError as exc:
            warnings.warn(
                f"Could not set montage '{montage}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return raw


def plot_eeg_with_rejected_segments(
    raw: "mne.io.BaseRaw",
    rejected_windows: Optional["pd.DataFrame"] = None,
    max_channels: int = 19,
    spacing: float = 8.0,
    figsize: tuple[float, float] = (16.0, 9.0),
    time_offset: float = 0.0,
    event_duration: Optional[float] = None,
    time_margin_s: Optional[float] = None,
):
    """Plot stacked EEG traces and highlight rejected windows.

    Args:
        raw: MNE Raw object with EEG channels.
        rejected_windows: DataFrame with columns ``start_s`` and ``end_s`` in NCDF time coords.
        max_channels: Maximum number of EEG channels to display.
        spacing: Vertical distance between channel traces.
        figsize: Matplotlib figure size.
        time_offset: First sample time from the NCDF time coordinate (typically negative,
            equal to -time_margin_s). Used to shift MNE's 0-based time axis to match the
            original NCDF time axis where 0 = event start.
        event_duration: Duration of the event in seconds; used to shade the post-event margin.
        time_margin_s: Margin length in seconds; enables light-gray shading of pre/post margins.

    Returns:
        tuple: (figure, axis)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    picks = raw.copy().pick("eeg")
    data = picks.get_data()
    # Shift MNE's 0-based time axis to match the NCDF time coordinate.
    times = picks.times + time_offset
    ch_names = list(picks.ch_names)

    if data.size == 0:
        raise ValueError("No EEG channels available to plot.")

    n_channels = min(max_channels, data.shape[0])
    data = data[:n_channels]
    ch_names = ch_names[:n_channels]

    stds = np.std(data, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    normalized = data / stds

    fig, ax = plt.subplots(figsize=figsize)
    offsets = np.arange(n_channels) * spacing

    # Light-gray shading for margin regions (drawn first, behind traces).
    if time_margin_s is not None and time_margin_s > 0:
        t_start = float(times[0])
        t_end = float(times[-1])
        if t_start < 0.0:
            ax.axvspan(t_start, 0.0, color="#cccccc", alpha=0.45, zorder=0, label="margin")
        if event_duration is not None and np.isfinite(event_duration) and t_end > event_duration:
            ax.axvspan(event_duration, t_end, color="#cccccc", alpha=0.45, zorder=0)

    for idx in range(n_channels):
        ax.plot(times, normalized[idx] + offsets[idx], linewidth=0.6, color="#1f4f8b", zorder=1)

    if rejected_windows is not None and not rejected_windows.empty:
        for _, row in rejected_windows.iterrows():
            ax.axvspan(float(row["start_s"]), float(row["end_s"]), color="#d62728", alpha=0.18, zorder=2)

    # Dashed vertical lines at event boundaries.
    if event_duration is not None and np.isfinite(event_duration):
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(event_duration, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_yticks(offsets)
    ax.set_yticklabels(ch_names)
    ax.set_xlabel("Time [s]  (0 = event start)")
    ax.set_ylabel("EEG channel")
    ax.set_title("EEG traces with AutoReject suggested rejection windows")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig, ax


def run_eeg_autoreject_quality_report(
    ncdf_path: str,
    epoch_duration_s: float = 2.0,
    n_interpolate: tuple[int, ...] = (1, 2, 4),
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    montage: Optional[str] = "standard_1020",
    scale_to_volts: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Create AutoReject quality report for EEG exported to NetCDF.

    Steps:
        1. Load NetCDF and convert to MNE Raw.
        2. Split signal into fixed-length epochs.
        3. Fit AutoReject and collect reject log.
        4. Build tabular summaries and visualization.

    Returns:
        dict with keys:
            - raw
            - epochs
            - autoreject
            - reject_log
            - epoch_summary
            - channel_summary
            - global_summary
            - figure
            - axis
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required for EEG quality analysis.") from exc

    try:
        from autoreject import AutoReject  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "autoreject is required for quality reporting. Install it with: pip install autoreject"
        ) from exc

    import numpy as np
    import pandas as pd

    # Load NCDF once – reuse the DataArray for both metadata extraction and MNE conversion.
    _data_xr_meta = load_xarray_from_netcdf(ncdf_path)
    time_margin_s = float(_data_xr_meta.attrs.get("time_margin_s", 0.0))
    # Sanitize event_duration: treat missing or non-finite values as None.
    _raw_event_duration = _data_xr_meta.attrs.get("event_duration")
    try:
        event_duration: Optional[float] = float(_raw_event_duration)
    except (TypeError, ValueError):
        event_duration = None
    else:
        if not np.isfinite(event_duration):
            event_duration = None
    # First value of the time coordinate is the pre-event start (e.g. -10 s when margin=10).
    time_offset = float(np.asarray(_data_xr_meta.coords["time"].values)[0])

    raw = load_eeg_ncdf_as_mne_raw(
        ncdf_path=ncdf_path,
        montage=montage,
        scale_to_volts=scale_to_volts,
        data_xr=_data_xr_meta,
    )
    del _data_xr_meta

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=float(epoch_duration_s),
        preload=True,
        verbose=verbose,
    )

    if len(epochs) == 0:
        raise ValueError("No epochs created. Check signal length and epoch_duration_s.")

    ar = AutoReject(
        n_interpolate=list(n_interpolate),
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    ar.fit(epochs)

    _, reject_log = ar.transform(epochs, return_log=True)
    labels = np.asarray(reject_log.labels)

    if hasattr(reject_log, "bad_epochs"):
        bad_epochs = np.asarray(reject_log.bad_epochs, dtype=bool)
    else:
        bad_epochs = np.any(labels == 2, axis=1)

    # Epoch times in MNE's 0-based frame; shift to actual NCDF time coordinate.
    _epoch_starts_mne = (epochs.events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    epoch_starts_actual = _epoch_starts_mne + time_offset
    epoch_ends_actual = epoch_starts_actual + float(epoch_duration_s)

    # An epoch is "in margin" when it lies entirely outside the event window.
    epoch_in_margin = [
        (end <= 0.0 or (event_duration is not None and start >= event_duration))
        for start, end in zip(epoch_starts_actual, epoch_ends_actual)
    ]

    epoch_summary = pd.DataFrame(
        {
            "epoch_idx": np.arange(len(epochs), dtype=int),
            "start_s": epoch_starts_actual,
            "end_s": epoch_ends_actual,
            "interpolated_channels": (labels == 1).sum(axis=1).astype(int),
            "rejected": bad_epochs,
            "in_margin": epoch_in_margin,
        }
    )

    n_epochs = len(epochs)
    channel_summary = pd.DataFrame(
        {
            "channel": list(epochs.ch_names),
            "interpolated_epochs": (labels == 1).sum(axis=0).astype(int),
            "bad_labels": (labels == 2).sum(axis=0).astype(int),
        }
    )
    channel_summary["interpolated_pct"] = 100.0 * channel_summary["interpolated_epochs"] / n_epochs
    channel_summary["bad_labels_pct"] = 100.0 * channel_summary["bad_labels"] / n_epochs
    channel_summary = channel_summary.sort_values(
        ["bad_labels", "interpolated_epochs", "channel"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    global_summary = {
        "ncdf_path": ncdf_path,
        "n_channels": int(len(epochs.ch_names)),
        "n_epochs": int(n_epochs),
        "epoch_duration_s": float(epoch_duration_s),
        "rejected_epochs": int(bad_epochs.sum()),
        "rejected_epochs_pct": float(100.0 * bad_epochs.mean()),
        "total_interpolations": int((labels == 1).sum()),
    }

    # Rejected windows outside margins only — margin artifacts are expected and uninformative.
    rejected_windows = epoch_summary.loc[
        epoch_summary["rejected"] & ~epoch_summary["in_margin"],
        ["start_s", "end_s"],
    ].reset_index(drop=True)
    fig, ax = plot_eeg_with_rejected_segments(
        raw,
        rejected_windows=rejected_windows,
        time_offset=time_offset,
        event_duration=event_duration,
        time_margin_s=time_margin_s,
    )

    return {
        "raw": raw,
        "epochs": epochs,
        "autoreject": ar,
        "reject_log": reject_log,
        "epoch_summary": epoch_summary,
        "channel_summary": channel_summary,
        "global_summary": global_summary,
        "figure": fig,
        "axis": ax,
    }

#------------


def save_to_file(multimodal_data: MultimodalData, output_dir: str) -> None:
    """
    Save MultimodalData instance to a joblib file.

    Args:
        multimodal_data: The multimodal data instance to save.
        output_dir: Directory path where the file will be saved.

    Returns:
        None: Saves file to {output_dir}/{dyad_id}.joblib
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{multimodal_data.id}.joblib")
    joblib.dump(multimodal_data, output_path)


def load_output_data(filename: str) -> MultimodalData | None:
    """
    Load saved MultimodalData from a joblib file.

    Args:
        filename: Path to the joblib file to load.

    Returns:
        MultimodalData or None: The loaded multimodal data instance, or None if file not found.
    """
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")
        return None