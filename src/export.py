"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import os

import joblib
import xarray as xr

from . import dataloader
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
    if multimodal_data.notes is None:
        data_xr.attrs['notes'] = "No additional notes provided."
    else:    
        data_xr.attrs['notes'] = multimodal_data.notes  # Any additional notes or comments about the data
    # Information about the child participant (age, gender, etc.)
    data_xr.attrs['child_group'] = multimodal_data.child_info.group
    data_xr.attrs['child_age_months'] = multimodal_data.child_info.age_months
    data_xr.attrs['child_gender'] = multimodal_data.child_info.sex


    if selected_modality == 'EEG':
        # Information about EEG filtration
        # notch 
        data_xr.attrs['filtration_notch_Q'] = multimodal_data.eeg_filtration.notch['Q']
        data_xr.attrs['filtration_notch_freq_hz'] = multimodal_data.eeg_filtration.notch['freq']
        data_xr.attrs['filtration_notch_a'] = multimodal_data.eeg_filtration.notch['a']
        data_xr.attrs['filtration_notch_b'] = multimodal_data.eeg_filtration.notch['b']
        # low_pass
        data_xr.attrs['filtration_low_pass_type'] = multimodal_data.eeg_filtration.low_pass['type'] 
        data_xr.attrs['filtration_low_pass.a'] = multimodal_data.eeg_filtration.low_pass['a']
        data_xr.attrs['filtration_low_pass.b'] = multimodal_data.eeg_filtration.low_pass['b']
        # high_pass
        data_xr.attrs['filtration_high_pass_type'] = multimodal_data.eeg_filtration.high_pass['type']
        data_xr.attrs['filtration_high_pass.a'] = multimodal_data.eeg_filtration.high_pass['a']
        data_xr.attrs['filtration_high_pass.b'] = multimodal_data.eeg_filtration.high_pass['b']
        # Information about reference electrodes
        data_xr.attrs['references'] = multimodal_data.references
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

def make_uniwaw_imported(dyad_id, load_eeg=True, load_et=True, load_meta=True, lowcut=1.0, highcut=40.0, eeg_filter_type='fir',decimate_factor=8, plot_flag=False, time_margin=10, root_path="../data/UNIWAW_imported"):
    multimodal_data = dataloader.create_multimodal_data(data_base_path = "../data",
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
    members = {'ch': 'child', 'cg': 'caregiver'}
    selected_channels = {
        'EEG': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5', 'P3', 'Pz',
                'P4', 'T6', 'O1', 'O2'],
        'ET': ['x', 'y', 'pupil', 'blinks'],
        'ECG': ['ECG'],
        'IBI': ['IBI']}
    path_dyad = os.path.join(root_path, str(multimodal_data.id))
    if not os.path.exists(path_dyad):
        os.makedirs(path_dyad)

    for modality in multimodal_data.modalities:
        path_modality = os.path.join(path_dyad, modality)
        if not os.path.exists(path_modality):
            os.makedirs(path_modality)
        for who, member in members.items():
            path_member = os.path.join(path_modality, member)
            if not os.path.exists(path_member):
                os.makedirs(path_member)
            for event in multimodal_data.events.keys():
                data_xr = export_to_xarray(multimodal_data=multimodal_data,
                                           selected_event=event,
                                           selected_channels=selected_channels.get(modality),
                                           selected_modality=modality,
                                           member=who,
                                           time_margin=time_margin)
                file_path = os.path.join(path_member, f'{multimodal_data.id}_{modality}_{who}_{event}.nc')
                data_xr.to_netcdf(file_path, engine='netcdf4')


