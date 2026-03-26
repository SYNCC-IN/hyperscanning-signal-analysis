"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import os
import json
import numbers
from dataclasses import asdict, is_dataclass
from typing import Optional

import joblib
import xarray as xr

from . import dataloader
from .data_structures import MultimodalData


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
    target_events = ["Peppa", "Incredible", "Brave"]
    ordered_target_events = sorted(
        (
            (event_name, multimodal_data.events[event_name]["start"])
            for event_name in target_events
            if event_name in multimodal_data.events and "start" in multimodal_data.events[event_name]
        ),
        key=lambda item: item[1],
    )

    metadata = {
        "notes": multimodal_data.notes,
        "child_info": _dataclass_or_dict(multimodal_data.child_info),
        "event_order": [event_name for event_name, _ in ordered_target_events],
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
        The DataArray will have dimensions 'time' and 'channel', and coordinates corresponding to the time points and channel names.
        Metadata attributes include information about dyad, member, sampling frequency, event details, and ``metadata_json``.
        The ``metadata_json`` payload contains ``notes`` and ``child_info`` and additionally
        ``event_order`` with the chronological order (by start time) of available target events:
        ``Peppa``, ``Incredible``, and ``Brave``.
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