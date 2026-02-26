"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import joblib
import xarray as xr

from .data_structures import MultimodalData

def export_to_xarray(multimodal_data, selected_event, selected_channels, selected_modality, member, time_margin):
    '''Export selected signals from a MultimodalData instance to an xarray DataArray.
    Args:
        multimodal_data: The MultimodalData instance containing the data.
        selected_event: The name of the event to select (e.g., 'Incredibles').
        selected_channels: List of channel names to include in the export (e.g., ['Fp1', 'Fp2'] for EEG).
        selected_modality: The modality to export (e.g., 'EEG', 'ECG', 'ET', 'IBI', or 'diode').
        member: The member to select ('ch' or 'cg').
        time_margin: Margin in seconds to include before and after the event.
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

    print(f"Event '{selected_event}' starts at {event_start:.2f}s and ends at {event_end:.2f}s")
    print(f"Selected time range with ±{time_margin}s margin: {selected_time[0]:.2f}s to {selected_time[1]:.2f}s")

    time, channels, data = multimodal_data.get_signals(
        mode=selected_modality,
        member=member,
        selected_channels=selected_channels,
        selected_times=selected_time
    )

    # convert the retrieved data to xarray DataArray, resetting time to 0 at event start
    time = time - selected_time[0] - time_margin
    # strip channel names to remove EEG_{member}_ prefix if modality is EEG
    if selected_modality == 'EEG':
        channels = [ch.replace(f'EEG_{member}_', '') for ch in channels]
    elif selected_modality == 'ET':
        channels = [ch.replace(f'ET_{member}_', '') for ch in channels]
    elif selected_modality == 'IBI':
        channels = [ch.replace(f'IBI_{member}_', '') for ch in channels]
    elif selected_modality == 'ECG':
        channels = [ch.replace(f'ECG_{member}_', '') for ch in channels]
    elif selected_modality == 'diode':
        channels = ['diode']

    data_xr = xr.DataArray(
        data,
        coords=[time, channels],
        dims=['time', 'channel'],
        name=f'{multimodal_data.id} {selected_modality} {member} {selected_event} with ±{time_margin}s margin'
    )

    # add metadata
    data_xr.attrs['dyad_id'] = multimodal_data.id
    data_xr.attrs['who'] = member
    data_xr.attrs['fs_hz'] = float(multimodal_data.fs)
    data_xr.attrs['event_name'] = selected_event
    data_xr.attrs['event_start_s'] = float(time_margin)
    data_xr.attrs['event_end_s'] = float(event_end - event_start)
    data_xr.attrs['time_margin_s'] = float(time_margin)
    data_xr.attrs['notes'] = multimodal_data.notes  # Any additional notes or comments about the data
    data_xr.attrs['child_info'] = multimodal_data.child_info  # Information about the child participant (age, gender, etc.)

    if selected_modality == 'EEG':
        data_xr.attrs['filtration'] = multimodal_data.eeg_filtration  # Information about EEG filtration
        data_xr.attrs['references'] = multimodal_data.references# Information about reference electrodes
    return data_xr


def save_to_file(multimodal_data: MultimodalData, output_dir: str) -> None:
    """
    Save MultimodalData instance to a joblib file.

    Args:
        multimodal_data: The multimodal data instance to save.
        output_dir: Directory path where the file will be saved.

    Returns:
        None: Saves file to {output_dir}/{dyad_id}.joblib
    """
    joblib.dump(multimodal_data, output_dir + f"/{multimodal_data.id}.joblib")


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
