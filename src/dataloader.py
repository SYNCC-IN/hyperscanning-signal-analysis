import os
from collections import deque

import numpy as np
import pandas as pd
import xmltodict
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter, sosfiltfilt, iirnotch, firwin, lfilter
import joblib

from src import eyetracker
from src.data_structures import MultimodalData
from src.utils import plot_filter_characteristics


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


def load_eyetracker_data(multimodal_data, folder_eyetracker, fs=None):
    # Load eye-tracking data from CSV files: THIS PART TO BE UPDATED AFTER THE STRUCTURE OF DATA in UW IS CLARIFIED
    # For now, we will load data from hardcoded paths for testing purposes
    # movies task 000
    ch_pos_df_0 = pd.read_csv(folder_eyetracker + '000/ch_gaze_positions_on_surface_Surface 1.csv')
    cg_pos_df_0 = pd.read_csv(folder_eyetracker + '000/cg_gaze_positions_on_surface_Surface 1.csv')
    ch_pupil_df_0 = pd.read_csv(folder_eyetracker + '000/ch_pupil_positions.csv')
    cg_pupil_df_0 = pd.read_csv(folder_eyetracker + '000/cg_pupil_positions.csv')
    annotations_0 = pd.read_csv(folder_eyetracker + '000/annotations.csv')
    cg_blinks_0 = pd.read_csv(folder_eyetracker + '000/cg_blinks.csv')
    ch_blinks_0 = pd.read_csv(folder_eyetracker + '000/ch_blinks.csv')
    # conversation task 001
    ch_pupil_df_1 = pd.read_csv(folder_eyetracker + '001/ch_pupil_positions.csv')
    cg_pupil_df_1 = pd.read_csv(folder_eyetracker + '001/cg_pupil_positions.csv')
    annotations_1 = pd.read_csv(folder_eyetracker + '001/annotations.csv')
    cg_blinks_1 = pd.read_csv(folder_eyetracker + '001/cg_blinks.csv')
    ch_blinks_1 = pd.read_csv(folder_eyetracker + '001/ch_blinks.csv')
    # conversation task 002
    ch_pupil_df_2 = pd.read_csv(folder_eyetracker + '002/ch_pupil_positions.csv')
    cg_pupil_df_2 = pd.read_csv(folder_eyetracker + '002/cg_pupil_positions.csv')
    annotations_2 = pd.read_csv(folder_eyetracker + '002/annotations.csv')
    cg_blinks_2 = pd.read_csv(folder_eyetracker + '002/cg_blinks.csv')
    ch_blinks_2 = pd.read_csv(folder_eyetracker + '002/ch_blinks.csv')

    # construct dataframe for ET data
    et_df = pd.DataFrame()
    # prepare the time column
    if fs is None:
        fs = multimodal_data.eeg_fs  # default sampling rate for UW EEG data; we want to keep all time series at the same sampling rate
        print(f"Warning: fs not provided for ET data; setting fs to the default Fs of EEG: {fs} Hz")
    et_df['time'] = eyetracker.process_time_et(ch_pos_df_0, cg_pos_df_0, ch_pupil_df_0, cg_pupil_df_0, ch_pupil_df_1,
                                               cg_pupil_df_1, ch_pupil_df_2, cg_pupil_df_2, Fs=fs)
    et_df['time_idx'] = (et_df['time'] * fs).astype(int)  # integer time indexes for merging with other modalities

    # process position, pupil, blink, and event data
    eyetracker.process_pos(ch_pos_df_0, et_df, 'ch')
    eyetracker.process_pos(cg_pos_df_0, et_df, 'cg')

    eyetracker.process_pupil(ch_pupil_df_0, et_df, 'ch')
    eyetracker.process_pupil(ch_pupil_df_1, et_df, 'ch')
    eyetracker.process_pupil(ch_pupil_df_2, et_df, 'ch')

    eyetracker.process_pupil(cg_pupil_df_0, et_df, 'cg')
    eyetracker.process_pupil(cg_pupil_df_1, et_df, 'cg')
    eyetracker.process_pupil(cg_pupil_df_2, et_df, 'cg')

    eyetracker.process_blinks(cg_blinks_0, et_df, 'cg')
    eyetracker.process_blinks(cg_blinks_1, et_df, 'cg')
    eyetracker.process_blinks(cg_blinks_2, et_df, 'cg')
    eyetracker.process_blinks(ch_blinks_0, et_df, 'ch')
    eyetracker.process_blinks(ch_blinks_1, et_df, 'ch')
    eyetracker.process_blinks(ch_blinks_2, et_df, 'ch')

    eyetracker.process_event_et(annotations_0, et_df)
    eyetracker.process_event_et(annotations_1, et_df, 'talk1')
    eyetracker.process_event_et(annotations_2, et_df, 'talk2')

    # align ET time to EEG time by subtracting the time of the first event; find the time of the first event in ET data
    min_start_time_et = et_df[et_df['ET_event'].notna()]['time'].min()
    et_df['time'] = et_df['time'] - min_start_time_et
    et_df['time_idx'] = et_df['time_idx'] - int(min_start_time_et * fs)

    #  merging ET data into the main dataframe

    multimodal_data.data = pd.merge(multimodal_data.data, et_df, how='outer', on='time_idx')
    multimodal_data.data['time'] = multimodal_data.data['time_idx'] / fs
    multimodal_data.data = multimodal_data.data.drop(columns=['time_x', 'time_y'])
    multimodal_data.data = multimodal_data.data.replace(np.nan, None)

    multimodal_data.modalities.append('ET')


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


def load_output_data(filename):
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")


def save_to_file(multimodal_data: MultimodalData, output_dir):
    joblib.dump(multimodal_data, output_dir + f"/{multimodal_data.id}.joblib")
