import os
from collections import deque

import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from scipy.signal import filtfilt, butter, sosfiltfilt, iirnotch
import joblib
from data_structures import MultimodalData


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
    multimodal_data.events, multimodal_data.diode = _scan_for_events(diode, multimodal_data.eeg_fs, plot_flag, threshold=0.75)
    print(f"Detected events: {multimodal_data.events}")

    # mount EEG data to M1 and M2 channels and filter the data (in place)
    _mount_eeg_data(multimodal_data, raw_eeg_data)
    filters = _design_eeg_filters(multimodal_data.eeg_fs, lowcut, highcut)
    _apply_filters(multimodal_data, filters)

    if 'EEG' not in multimodal_data.modalities:
        multimodal_data.modalities.append('EEG')

    # set the ECG modality with ECG signals (in place)
    _extract_ecg_data(multimodal_data)

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

    multimodal_data.eeg_data = raw_eeg_data


def _design_eeg_filters(fs, lowcut, highcut, notch_freq=50, notch_q=30):
    """
    Task 1: Designs notch, low-pass, and high-pass filters.
    Returns a tuple of filter coefficients.
    """
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs=fs)
    b_low, a_low = butter(N=4, Wn=highcut, btype='low', fs=fs)
    b_high, a_high = butter(N=4, Wn=lowcut, btype='high', fs=fs)

    return (b_notch, a_notch), (b_low, a_low), (b_high, a_high)


def _apply_filters(multimodal_data: MultimodalData, filters):
    """
    Applies filters to raw data.
    Returns filtered data.
    """
    (b_notch, a_notch), (b_low, a_low), (b_high, a_high) = filters

    # Filter and separate each channel
    for idx, ch in enumerate(multimodal_data.eeg_channel_names_all()):
        signal = multimodal_data.eeg_data[idx, :]  # .copy()
        signal = filtfilt(b_notch, a_notch, signal, axis=0)
        signal = filtfilt(b_low, a_low, signal, axis=0)
        signal = filtfilt(b_high, a_high, signal, axis=0)

        multimodal_data.eeg_data[idx, :] = signal


def _extract_ecg_data(multimodal_data: MultimodalData):
    eeg_data = multimodal_data.eeg_data
    channel_mapping = multimodal_data.eeg_channel_mapping

    t_ecg = np.arange(0, eeg_data.shape[1] / multimodal_data.ecg_fs,
                      1 / multimodal_data.ecg_fs)  # time vector for the ECG data in seconds

    # extract and filter the ECG data
    ecg_ch = eeg_data[channel_mapping['EKG1'], :] - eeg_data[channel_mapping['EKG2'], :]
    ecg_cg = eeg_data[channel_mapping['EKG1_cg'], :] - eeg_data[channel_mapping['EKG2_cg'], :]

    # design filters:
    b_notch, a_notch = iirnotch(50, 30, fs=multimodal_data.ecg_fs)
    sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=multimodal_data.ecg_fs)
    ecg_ch_filtered = sosfiltfilt(sos_ecg, ecg_ch)
    ecg_ch_filtered = filtfilt(b_notch, a_notch, ecg_ch_filtered)
    ecg_cg_filtered = sosfiltfilt(sos_ecg, ecg_cg)
    ecg_cg_filtered = filtfilt(b_notch, a_notch, ecg_cg_filtered)
    multimodal_data.ecg_data_ch = ecg_ch_filtered
    multimodal_data.ecg_data_cg = ecg_cg_filtered
    multimodal_data.ecg_times = t_ecg
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

    return events,thresholded_diode


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
