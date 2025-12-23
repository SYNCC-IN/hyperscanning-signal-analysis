import os
from collections import deque

import numpy as np
import pandas as pd
import xmltodict
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter, sosfiltfilt, iirnotch, firwin, lfilter
import joblib

from src import eyetracker as et
from src.data_structures import MultimodalData
from src.utils import plot_filter_characteristics

# --------------  Load EEG and ECG data form SVAROG files -----------------
def load_eeg_data(dyad_id, folder_eeg, plot_flag, lowcut=4.0, highcut=40.0):
    """Set the EEG data for the DataLoader instance by loading and filtering the Warsaw pilot data.
    We assume data were recorded as multiplexed signals in SVAROG system format.
    We also assume specific channel names for child and caregiver EEG data, as specified below.
    Args:
        folder_eeg (str): Path to the folder containing the EEG data files.
        plot_flag (bool): Whether to plot intermediate results for debugging/visualization.
    """
    multimodal_data = MultimodalData()
    multimodal_data.id = dyad_id
    multimodal_data.paths.eeg_directory = folder_eeg
    multimodal_data.eeg_channel_names_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz',
                                            'C4', 'T4',
                                            'M2',
                                            'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    multimodal_data.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg',
                                            'M1_cg', 'T3_cg',
                                            'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 'T5_cg', 'P3_cg', 'Pz_cg',
                                            'P4_cg',
                                            'T6_cg', 'O1_cg', 'O2_cg']

    raw_eeg_data = _read_raw_svarog_data(multimodal_data, plot_flag)

    # extract diode signal for event detection filtering
    diode = raw_eeg_data[multimodal_data.eeg_channel_mapping['Diode'], :]

    # scan for events
    multimodal_data.events, thresholded_diode = _scan_for_events(diode, multimodal_data.eeg_fs, plot_flag,
                                                                     threshold=0.75)
    print(f"Detected events: {multimodal_data.events}")

    # mount EEG data to M1 and M2 channels and filter the data (in place)
    _mount_eeg_data(multimodal_data, raw_eeg_data)
    filters = _design_eeg_filters(multimodal_data.eeg_fs, lowcut, highcut)
    _apply_filters(multimodal_data, filters, raw_eeg_data)

    # Store EEG data in DataFrame with each channel as a column
    multimodal_data.set_eeg_data(raw_eeg_data, multimodal_data.eeg_channel_mapping)

    # Store diode in DataFrame
    multimodal_data.set_diode(thresholded_diode)

    if 'EEG' not in multimodal_data.modalities:
        multimodal_data.modalities.append('EEG')

    # set the ECG modality with ECG signals (in place)
    _extract_ecg_data(multimodal_data, raw_eeg_data)

    return multimodal_data


def _read_raw_svarog_data(multimodal_data: MultimodalData, plot_flag):
    file = multimodal_data.id + ".obci"  # SVAROG files have .obci extension
    # read meta information from xml file
    with open(os.path.join(multimodal_data.paths.eeg_directory, f"{file}.xml")) as fd:
        xml = xmltodict.parse(fd.read())

    n_channels = int(xml['rs:rawSignal']['rs:channelCount'])
    eeg_fs = int(float(xml['rs:rawSignal']['rs:samplingFrequency']))
    channel_names = xml['rs:rawSignal']['rs:channelLabels']['rs:label']

    for i, name in enumerate(channel_names):
        multimodal_data.eeg_channel_mapping[name] = i

    # if debug print N_chan, Fs_EEG, chan_names
    if plot_flag:
        print(f"n_channels: {n_channels},\n fs_EEG: {eeg_fs},\n chan_names: {channel_names}")

    multimodal_data.eeg_fs = eeg_fs
    multimodal_data.ecg_fs = eeg_fs  # ECG data is sampled at the same frequency as EEG data
    raw_eeg_data = np.fromfile(os.path.join(multimodal_data.paths.eeg_directory, f"{file}.raw"),
                               dtype='float32').reshape(
        (-1, n_channels)).T  # transpose to have channels in rows and samples in columns

    # scale the signal to microvolts
    raw_eeg_data *= 0.0715

    return raw_eeg_data


def _mount_eeg_data(multimodal_data, raw_eeg_data):
    channel_mapping = multimodal_data.eeg_channel_mapping
    # mount EEG data to M1 and M2 channels; do it separately for caregiver and child as they have different references
    for channel in multimodal_data.eeg_channel_names_ch:
        if channel in channel_mapping and channel not in ['M1', 'M2']:
            idx = channel_mapping[channel]
            raw_eeg_data[idx, :] -= 0.5 * (
                    raw_eeg_data[channel_mapping['M1'], :] + raw_eeg_data[channel_mapping['M2'], :])

    for channel in multimodal_data.eeg_channel_names_cg:
        if channel in channel_mapping and channel not in ['M1_cg', 'M2_cg']:
            idx = channel_mapping[channel]
            raw_eeg_data[idx, :] -= 0.5 * (
                    raw_eeg_data[channel_mapping['M1_cg'], :] + raw_eeg_data[channel_mapping['M2_cg'], :])

    # adjust channel lists by removing channels M1 and M2 from the caregiver and child EEG channels, as they will not be used after linked ears montage
    multimodal_data.eeg_channel_names_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                                            'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    multimodal_data.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg',
                                            'T3_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'T5_cg', 'P3_cg', 'Pz_cg',
                                            'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']



def _design_eeg_filters(fs, lowcut, highcut, notch_freq=50, notch_q=30, filter_type='iir', plot_flag=False):
    """
    Task 1: Designs notch, low-pass, and high-pass filters.
    Returns a tuple of filter coefficients.
    """
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs=fs)

    if filter_type == 'fir':
        numtaps_low = 201
        b_low = firwin(numtaps_low, highcut, fs=fs, pass_zero='lowpass')
        numtaps_high = 1025
        b_high = firwin(numtaps_high, lowcut, fs=fs, pass_zero='highpass')
        a_low = a_high = 1.0
    else:
        b_low, a_low = butter(N=4, Wn=highcut, btype='low', fs=fs)
        b_high, a_high = butter(N=4, Wn=lowcut, btype='high', fs=fs)

    if plot_flag:
        print("---- Notch filter characteristics: --------")
        f_max = 60.0
        plot_filter_characteristics(b_notch, a_notch, f=np.arange(0, f_max, 0.01), T=0.5, Fs=fs, f_lim=(30, f_max),
                                    db_lim=(-300, 0.1))
        print("---- Low-pass filter characteristics: --------")
        plot_filter_characteristics(b_low, a=[1], f=np.arange(0, fs / 2, 0.1), T=0.5, Fs=fs, f_lim=(0, 50),
                                    db_lim=(-60, 0.1))
        print("---- High-pass filter characteristics: --------")
        plot_filter_characteristics(b_high, a=[1], f=np.arange(0, fs / 2, 0.01), T=0.5, Fs=fs, f_lim=(0, 10),
                                    db_lim=(-60, 0.1))

    return (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type


def _apply_filters(multimodal_data: MultimodalData, filters, raw_eeg_data):
    """
    Applies filters to raw data in place.
    """
    (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type = filters
    print(f"Applying {filter_type} filters to EEG data.")

    # Filter and separate each channel
    for idx, ch in enumerate(multimodal_data.eeg_channel_names_all()):
        signal = raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :]

        if filter_type == 'iir':
            signal = filtfilt(b_notch, a_notch, signal, axis=0)
            signal = filtfilt(b_low, a_low, signal, axis=0)
            signal = filtfilt(b_high, a_high, signal, axis=0)
        else:
            signal = lfilter(b_notch, a_notch, signal, axis=0)
            signal = lfilter(b_low, a_low, signal, axis=0)
            signal = lfilter(b_high, a_high, signal, axis=0)

            delay = (len(b_low) - 1) // 2 + (len(b_high) - 1) // 2
            signal = np.roll(signal, -delay)
            signal[-delay:] = 0.0  # zero-pad the end to account for the delay introduced by filtering

        raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :] = signal


def _extract_ecg_data(multimodal_data: MultimodalData, raw_eeg_data):
    channel_mapping = multimodal_data.eeg_channel_mapping

    # extract and filter the ECG data
    ecg_ch = raw_eeg_data[channel_mapping['EKG1'], :] - raw_eeg_data[channel_mapping['EKG2'], :]
    ecg_cg = raw_eeg_data[channel_mapping['EKG1_cg'], :] - raw_eeg_data[channel_mapping['EKG2_cg'], :]

    # design filters:
    b_notch, a_notch = iirnotch(50, 30, fs=multimodal_data.ecg_fs)
    sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=multimodal_data.ecg_fs)
    ecg_ch_filtered = sosfiltfilt(sos_ecg, ecg_ch)
    ecg_ch_filtered = filtfilt(b_notch, a_notch, ecg_ch_filtered)
    ecg_cg_filtered = sosfiltfilt(sos_ecg, ecg_cg)
    ecg_cg_filtered = filtfilt(b_notch, a_notch, ecg_cg_filtered)

    # Store ECG data in DataFrame
    multimodal_data.set_ecg_data(ecg_ch_filtered, ecg_cg_filtered)

    if 'ECG' not in multimodal_data.modalities:
        multimodal_data.modalities.append('ECG')





def _scan_for_events(diode, eeg_fs, plot_flag, threshold=0.75):
    """Scans the diode signal to detect and identify experimental events.

    This method processes the raw diode signal to find periods corresponding to
    specific experimental events, such as watching movies or engaging in conversation.
    It first binarizes the signal based on a given threshold to identify "on"
    and "off" states. It then analyzes the durations and intervals of these states
    to classify them into predefined event categories.

    The detection logic is tailored to a specific experimental design, expecting
    three movie sessions followed by two conversation sessions.

    Args:
        threshold (float, optional): The threshold for binarizing the diode signal,
            relative to its maximum value. Defaults to 0.75.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
            detected event and contains the following keys:
            - 'name' (str): The name of the event (e.g., 'Brave', 'Talk_1').
            - 'start' (float): The start time of the event in seconds from the
              beginning of the recording.
            - 'duration' (float): The duration of the event in seconds.
    """
    # Binarize the diode signal: values above the threshold become 1, others 0.
    thresholded_diode = ((diode / (threshold * np.max(diode))) > 1).astype(float)

    # Find rising (1) and falling (-1) edges in the binarized signal.
    # Collect the sample indices of all rising and falling edges.
    up_down_events = np.where(np.abs(np.diff(thresholded_diode)) == 1)[0].tolist() + [len(diode)]

    events = [{'name': name} for name in ['Brave', 'Peppa', 'Incredibles', 'Talk_1', 'Talk_2']]

    found_movies = found_talks = 0
    queue = deque(maxlen=100)

    # Process pairs of up/down events to identify event durations and intervals.
    for i in range(len(up_down_events) // 2):
        start = up_down_events[2 * i]
        duration = up_down_events[2 * i + 1] - up_down_events[2 * i]
        # Calculate the time until the next event starts.
        following_space = up_down_events[2 * i + 2] - up_down_events[2 * i + 1]
        queue.append(start)
        # Maintain a queue of recent event start times
        while queue[0] < start - 4 * eeg_fs:  # last 4 seconds
            queue.popleft()
        # Detect movie events based on their duration and number of recent spikes
        if duration > 55 * eeg_fs and len(queue) > 1:  # movie events longer than 0:55
            events[len(queue) - 2]['start'] = queue[0] / eeg_fs
            events[len(queue) - 2]['duration'] = (up_down_events[2 * i + 1] - queue[0]) / eeg_fs
            found_movies += 1
        if found_movies > 3:
            raise ValueError("More than 3 events detected, something is wrong.")
        # Detect talk events based on their position relative to movie events
        if found_movies == 3 and duration < 2 * eeg_fs and following_space > 175 * eeg_fs:  # talk events longer than 2:55
            if found_talks < 2:
                event_index = found_movies + found_talks
                events[event_index]['start'] = up_down_events[2 * i + 1] / eeg_fs
                events[event_index]['duration'] = following_space / eeg_fs
                found_talks += 1
            else:
                raise ValueError("More than 2 talks detected, something is wrong.")

    if plot_flag:
        _plot_scanned_events(threshold, diode, thresholded_diode, np.diff(thresholded_diode), events, eeg_fs)

    return events, thresholded_diode


def _plot_scanned_events(threshold, diode, thresholded_diode, derivative, events, eeg_fs):
    plt.figure(figsize=(12, 6))
    plt.plot(diode / (threshold * np.max(diode)), 'b', label='Diode Signal normalized by threshold')
    plt.plot(thresholded_diode, 'r', label='Diode Signal Thresholded')
    plt.title('Diode Signal with events')
    plt.xlabel('Samples')
    plt.ylabel('Signal Value')
    plt.plot((derivative == 1).astype(int), 'g', label='Up Events')
    plt.plot((derivative == -1).astype(int), 'm', label='Down Events')
    for event in events:
        if 'start' in event:
            plt.plot(event['start'] * eeg_fs, 1.2, 'ko', markersize=10)
            plt.text(event['start'] * eeg_fs, 1.25, event['name'], rotation=45)
    plt.legend()

# --------------  Save and load multimodal data -----------------
def load_output_data(filename):
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")


def save_to_file(multimodal_data: MultimodalData, output_dir):
    joblib.dump(multimodal_data, output_dir + f"/{multimodal_data.id}.joblib")

# --------------  Load eye-tracking data -----------------

def _build_et_file_paths(self, et_path: str, task_id: str, member: str) -> dict:
    """
    Build file paths for eye-tracker data for a specific task and dyad member.
    
    Args:
        et_path: Base path to ET data directory
        task_id: Task identifier ('000', '001', '002')
        member: Dyad member ('child' or 'caregiver')
        
    Returns:
        Dictionary mapping data types to file paths
    """
    base_export_path = f"{et_path}{member}/{task_id}/exports/000/"
    prefix = 'ch' if member == 'child' else 'cg'
    
    paths = {
        f'annotations_{task_id}': f"{base_export_path}annotations.csv",
        f'{prefix}_pupil_{task_id}': f"{base_export_path}pupil_positions.csv"
    }
    
    # Movies task (000) has additional gaze position and blinks data
    if task_id == '000':
        paths[f'{prefix}_pos_{task_id}'] = f"{base_export_path}surfaces/gaze_positions_on_surface_Surface 1.csv"
        paths[f'{prefix}_blinks_{task_id}'] = f"{base_export_path}blinks.csv"
    
    return paths

def _check_et_files_exist(self, file_paths: dict) -> tuple[bool, list]:
    """
    Check if ET data files exist.
    
    Args:
        file_paths: Dictionary mapping data types to file paths
        
    Returns:
        Tuple of (all_exist: bool, missing_files: list)
    """
    missing_files = [name for name, path in file_paths.items() if not os.path.exists(path)]
    return len(missing_files) == 0, missing_files

def _load_et_task_data(self, file_paths: dict, task_id: str, member: str, min_max_times: list) -> dict:
    """
    Load ET data files for a specific task and dyad member.
    
    Args:
        file_paths: Dictionary mapping data types to file paths
        task_id: Task identifier ('000', '001', '002')
        member: Dyad member ('child' or 'caregiver')
        min_max_times: List to append (min_time, max_time) tuples for time alignment
        
    Returns:
        Dictionary containing loaded DataFrames
    """
    prefix = 'ch' if member == 'child' else 'cg'
    loaded_data = {}
    
    # Load annotations
    ann_key = f'annotations_{task_id}'
    if ann_key in file_paths:
        loaded_data[ann_key] = pd.read_csv(file_paths[ann_key])
    
    # Load pupil data
    pupil_key = f'{prefix}_pupil_{task_id}'
    if pupil_key in file_paths:
        pupil_df = pd.read_csv(file_paths[pupil_key])
        loaded_data[pupil_key] = pupil_df
        min_max_times.append((pupil_df['pupil_timestamp'].min(), pupil_df['pupil_timestamp'].max()))
    
    # Load gaze position data (movies task only)
    pos_key = f'{prefix}_pos_{task_id}'
    if pos_key in file_paths:
        pos_df = pd.read_csv(file_paths[pos_key])
        loaded_data[pos_key] = pos_df
        min_max_times.append((pos_df['gaze_timestamp'].min(), pos_df['gaze_timestamp'].max()))
    
    # Load blinks data (movies task only)
    blinks_key = f'{prefix}_blinks_{task_id}'
    if blinks_key in file_paths:
        loaded_data[blinks_key] = pd.read_csv(file_paths[blinks_key])
    
    return loaded_data

def _process_et_data_to_dataframe(self, et_df: pd.DataFrame, loaded_data: dict, 
                                    task_flags: dict, task_names: list) -> None:
    """
    Process loaded ET data into the main ET dataframe.
    
    Args:
        et_df: DataFrame to populate with ET data
        loaded_data: Dictionary containing all loaded ET DataFrames
        task_flags: Dictionary with flags indicating which tasks/members have data
        task_names: List of task identifiers ['000', '001', '002']
    """
    # Process movies task (000) if available
    if task_flags.get('movies_ch') or task_flags.get('movies_cg'):
        if 'annotations_000' in loaded_data:
            et.process_event_et(loaded_data['annotations_000'], et_df)
    
    if task_flags.get('movies_ch'):
        if 'ch_pos_000' in loaded_data:
            et.process_pos(loaded_data['ch_pos_000'], et_df, 'ch')
        if 'ch_pupil_000' in loaded_data:
            et.process_pupil(loaded_data['ch_pupil_000'], et_df, 'ch')
        if 'ch_blinks_000' in loaded_data:
            et.process_blinks(loaded_data['ch_blinks_000'], et_df, 'ch')
    
    if task_flags.get('movies_cg'):
        if 'cg_pos_000' in loaded_data:
            et.process_pos(loaded_data['cg_pos_000'], et_df, 'cg')
        if 'cg_pupil_000' in loaded_data:
            et.process_pupil(loaded_data['cg_pupil_000'], et_df, 'cg')
        if 'cg_blinks_000' in loaded_data:
            et.process_blinks(loaded_data['cg_blinks_000'], et_df, 'cg')
    
    # Process talk1 task (001)
    if task_flags.get('talk1_ch') or task_flags.get('talk1_cg'):
        if 'annotations_001' in loaded_data:
            et.process_event_et(loaded_data['annotations_001'], et_df, 'talk1')
    
    if task_flags.get('talk1_ch'):
        if 'ch_pupil_001' in loaded_data:
            et.process_pupil(loaded_data['ch_pupil_001'], et_df, 'ch')
    
    if task_flags.get('talk1_cg'):
        if 'cg_pupil_001' in loaded_data:
            et.process_pupil(loaded_data['cg_pupil_001'], et_df, 'cg')
    
    # Process talk2 task (002)
    if task_flags.get('talk2_ch') or task_flags.get('talk2_cg'):
        if 'annotations_002' in loaded_data:
            et.process_event_et(loaded_data['annotations_002'], et_df, 'talk2')
    
    if task_flags.get('talk2_ch'):
        if 'ch_pupil_002' in loaded_data:
            et.process_pupil(loaded_data['ch_pupil_002'], et_df, 'ch')
    
    if task_flags.get('talk2_cg'):
        if 'cg_pupil_002' in loaded_data:
            et.process_pupil(loaded_data['cg_pupil_002'], et_df, 'cg')

def load_eye_tracker_data(multimodal_data, dyad_id, et_path, plot_flag):
    """ Load eye-tracking data from CSV files and integrate into the MultimodalData instance.

    Args:
        multimodal_data: Instance of MultimodalData to populate with ET data
        dyad_id: Identifier for the dyad
        et_path: Base path to ET data directory
        plot_flag: Whether to plot intermediate results for debugging/visualization
    """
    # Configuration for tasks: 000=movies, 001=talk1, 002=talk2
    tasks = [
        {'id': '000', 'name': 'movies'},
        {'id': '001', 'name': 'talk1'}, 
        {'id': '002', 'name': 'talk2'}
    ]
    members = ['child', 'caregiver']
            
    # Build file paths and check availability for each task and member
    task_flags = {}
    all_file_paths = {}
    min_max_times = []
    loaded_data = {}
    
    for task in tasks:
        for member in members:
            # Build file paths
            file_paths = _build_et_file_paths(et_path, task['id'], member)
            
            # Check if files exist
            all_exist, missing = _check_et_files_exist(file_paths)
            
            # Set flag for this task/member combination
            flag_key = f"{task['name']}_{'ch' if member == 'child' else 'cg'}"
            task_flags[flag_key] = all_exist
            
            if not all_exist:
                print(f"Warning: Missing ET files for {multimodal_data.id} {member} {task['name']}: {missing}")
            else:
                # Load data if all files exist
                task_data = _load_et_task_data(file_paths, task['id'], member, min_max_times)
                loaded_data.update(task_data)
    
    # Skip ET processing if no data was loaded
    if not min_max_times:
        print(f"Warning: No ET data available for {multimodal_data.id}")
        return
    
    # Construct dataframe for ET data
    et_df = pd.DataFrame()
    
    # Set sampling frequency if not already set
    if multimodal_data.fs is None:
        multimodal_data.fs = 1024  # default sampling rate for UW EEG data
        print(f'Setting fs to the default Fs of EEG: {multimodal_data.fs}')
    
    # Find the overall min and max times across all tasks and members
    overall_min_time = min([t[0] for t in min_max_times])
    overall_max_time = max([t[1] for t in min_max_times])
    
    print(f"ET time range: {overall_min_time:.2f}s to {overall_max_time:.2f}s")
    
    # Create time vector
    et_df['time'] = np.arange(overall_min_time, overall_max_time, 1 / multimodal_data.fs)
    et_df['time_idx'] = (et_df['time'] * multimodal_data.fs).astype(int)
    
    # Process loaded data into the dataframe
    _process_et_data_to_dataframe(et_df, loaded_data, task_flags, [t['id'] for t in tasks])
        
    # Align ET time to EEG time by subtracting the time of the first event; 
    # We consider the time of first event in all data series to be 0; 
    # reset time_idx accordingly
    min_start_time_et = et_df[et_df['ET_event'].notna()]['time'].min()
    et_df['time'] = et_df['time'] - min_start_time_et
    et_df['time_idx'] = et_df['time_idx'] - int(min_start_time_et*multimodal_data.fs)

    #  merging ET data into the main dataframe
    if multimodal_data.data is None:
        multimodal_data.data = et_df.copy()
    else:
        multimodal_data.data = pd.merge(multimodal_data.data, et_df, how = 'outer', on = 'time_idx')
        # multimodal_data.data['time'] = multimodal_data.data['time_idx'] / multimodal_data.fs # redundant, time column already should exist, if not search for bug!
        multimodal_data.data = multimodal_data.data.drop(columns=['time_x','time_y'])
        multimodal_data.data = multimodal_data.data.replace(np.nan, None)

    multimodal_data.info['modalities'].append('ET')
    return
