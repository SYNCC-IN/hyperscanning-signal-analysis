import os
from collections import deque

import numpy as np
import pandas as pd
import xmltodict
from matplotlib import pyplot as plt
from scipy.signal import (
    filtfilt,
    butter,
    sosfiltfilt,
    iirnotch,
    firwin,
    lfilter,
)
import joblib
import mne

import importlib
from . import eyetracker as et
importlib.reload(et)
from .data_structures import MultimodalData
from .utils import plot_filter_characteristics

# --------------  Create multimodal data instance and populate it with dat


def create_multimodal_data(
    data_base_path,
    dyad_id,
    load_eeg=True,
    load_et=True,
    lowcut=4.0,
    highcut=40.0,
    eeg_filter_type="fir",
    interpolate_et_during_blinks_threshold=0,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=4,
    pupil_model_confidence=0.9,
    decimate_factor=1,
    plot_flag=False,
):
    """Create and populate a MultimodalData instance by loading EEG and ET data.
    directory structure assumed is:
    data_base_path/
    <dyad_id>/
        eeg/
            <dyad_id>.obci
            <dyad_id>.xml
        et/
            child/
                000/
                001/
                002/
            caregiver/
                000/
                001/
                002/

    Args:
        data_base_path (str): Base path to the data directory.
        dyad_id (str): Identifier for the dyad.
        load_eeg (bool, optional): Whether to load EEG data. Defaults to True.
        load_et (bool, optional): Whether to load eye-tracker data. Defaults to True.
        lowcut (float, optional): Low cut-off frequency for EEG filtering. Defaults to 4.0 Hz.
        highcut (float, optional): High cut-off frequency for EEG filtering. Defaults to 40.0 Hz.
        eeg_filter_type (str, optional): Type of filter to use for EEG data ('fir' or 'iir'). Defaults to 'fir'.
        interpolate_et_during_blinks_threshold (float, optional): Confidence threshold for interpolating ET data during blinks. 0 means no interpolation. Defaults to 0.
        median_filter_size (int, optional): Size of the median filter for ET data processing. Defaults to 64.
        low_pass_et_order (int, optional): Order of the low-pass filter for ET data processing. Defaults to 351.
        et_pos_cutoff (float, optional): Cutoff frequency for ET position data low-pass filter. Defaults to 128 Hz.
        et_pupil_cutoff (float, optional): Cutoff frequency for ET pupil data low-pass filter. Defaults to 4 Hz.
        pupil_model_confidence (float, optional): Confidence level for 3D pupil model. Defaults to 0.9.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.

    Returns:
        MultimodalData: An instance populated with EEG and ET data.
    """
    multimodal_data = MultimodalData()
    multimodal_data.id = dyad_id
    if load_eeg:
        folder_eeg = os.path.join(data_base_path, dyad_id, "eeg")
        multimodal_data = load_eeg_data(
            multimodal_data,
            dyad_id=dyad_id,
            folder_eeg=folder_eeg,
            lowcut=lowcut,
            highcut=highcut,
            eeg_filter_type=eeg_filter_type,
            plot_flag=plot_flag,
        )
    if load_et:
        folder_et = os.path.join(data_base_path, dyad_id, "et")
        if multimodal_data.fs is None:
            # default EEG sampling frequency common to all signals if EEG data
            # not loaded or set before ET data
            multimodal_data.fs = 1024
            print("Setting default EEG sampling frequency to 1024 Hz used also in ET data.")
        multimodal_data = load_et_data(
            multimodal_data,
            dyad_id=dyad_id,
            folder_et=folder_et,
            interpolate_et_during_blinks_threshold=interpolate_et_during_blinks_threshold,
            median_filter_size=median_filter_size,
            low_pass_et_order=low_pass_et_order,
            et_pos_cutoff=et_pos_cutoff,
            et_pupil_cutoff=et_pupil_cutoff,
            pupil_model_confidence=pupil_model_confidence,
            plot_flag=plot_flag,
        )
    if decimate_factor > 1:
        multimodal_data = multimodal_data.decimate_signals(q=decimate_factor)
    multimodal_data.create_events_column()
    #result check_consistency_of_multimodal_data(multimodal_data)
    return multimodal_data


# --------------  Load EEG and ECG data form SVAROG files -----------------


def load_eeg_data(
    multimodal_data=None,
    dyad_id=None,
    folder_eeg=None,
    lowcut=4.0,
    highcut=40.0,
    eeg_filter_type="fir",
    plot_flag=False,
):
    """Load and filter EEG data from SVAROG format files into MultimodalData instance.
    Assumes data were recorded as multiplexed signals in SVAROG system format.
    Assumes specific channel names for child and caregiver EEG data.

    Args:
        multimodal_data (MultimodalData, optional): An existing MultimodalData instance to populate.
            If None, a new instance is created. Defaults to None.
        dyad_id (str, optional): Identifier for the dyad. Defaults to None.
        folder_eeg (str, optional): Path to the folder containing the EEG data files. Defaults to None.
        lowcut (float, optional): Low cut-off frequency for EEG filtering. Defaults to 4.0 Hz.
        highcut (float, optional): High cut-off frequency for EEG filtering. Defaults to 40.0 Hz.
        eeg_filter_type (str, optional): Type of filter to use ('fir' or 'iir'). Defaults to 'fir'.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.

    Returns:
        MultimodalData: The populated multimodal data instance with EEG, ECG, and event data.
    """
    if multimodal_data is None:
        multimodal_data = MultimodalData()
        multimodal_data.id = dyad_id
    multimodal_data.paths.eeg_directory = folder_eeg
    multimodal_data.eeg_channel_names_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                                            'M1', 'T3', 'C3', 'Cz','C4', 'T4','M2',
                                            'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    multimodal_data.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg',
                                            'M1_cg', 'T3_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 
                                            'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']

    raw_eeg_data = _read_raw_svarog_data(multimodal_data, plot_flag)

    # extract diode signal for event detection filtering
    diode = raw_eeg_data[multimodal_data.eeg_channel_mapping["Diode"], :]

    # scan for events
    multimodal_data.events, thresholded_diode = _scan_for_events(
        diode, multimodal_data.fs, plot_flag, threshold=0.75
    )
    print(f"Detected events: {multimodal_data.events}")

    # mount EEG data to M1 and M2 channels and filter the data (in place)
    _mount_eeg_data(multimodal_data, raw_eeg_data)
    filters = _design_eeg_filters(
        multimodal_data,
        lowcut=lowcut,
        highcut=highcut,
        filter_type=eeg_filter_type,
        notch_freq=50,
        notch_q=30,
        plot_flag=plot_flag,
    )
    _apply_filters(multimodal_data, filters, raw_eeg_data, plot_flag=plot_flag)

    # Store EEG data in DataFrame with each channel as a column and set time
    # column if not set yet
    multimodal_data.set_eeg_data(
        raw_eeg_data, multimodal_data.eeg_channel_mapping
    )
    # Set EEG events column
    multimodal_data.set_EEG_events_column(multimodal_data.events)

    # Store diode in DataFrame
    multimodal_data.set_diode(thresholded_diode)

    if "EEG" not in multimodal_data.modalities:
        multimodal_data.modalities.append("EEG")

    # set the ECG modality with ECG signals (in place)
    _extract_ecg_data(multimodal_data, raw_eeg_data)

    # reset time column to be consistent with the first movie event start at time zero; this is needed to align with ET data later; accordingly reset time_idx
    # in the column 'EEG_events' find the first occurance of one of 'Brave',
    # 'Peppa', 'Incredibles'; reset to the corresponding time
    first_movie_event = multimodal_data.data[
        multimodal_data.data["EEG_events"].isin(["Brave", "Peppa", "Incredibles"])
    ]["time"].min()
    print(
        f"Reseting the EEG time to the start of {multimodal_data.data[multimodal_data.data['time'] == first_movie_event]['EEG_events'].iloc[0]}"
    )
    multimodal_data.data["time"] = (
        multimodal_data.data["time"] - first_movie_event
    )
    multimodal_data.data["time_idx"] = (
        multimodal_data.data["time"] * multimodal_data.fs
    ).astype(int)

    return multimodal_data


def _read_raw_svarog_data(multimodal_data: MultimodalData, plot_flag):
    file = multimodal_data.id + ".obci"  # SVAROG files have .obci extension
    # read meta information from xml file
    with open(
        os.path.join(multimodal_data.paths.eeg_directory, f"{file}.xml")
    ) as fd:
        xml = xmltodict.parse(fd.read())

    n_channels = int(xml["rs:rawSignal"]["rs:channelCount"])
    fs = int(float(xml["rs:rawSignal"]["rs:samplingFrequency"]))
    channel_names = xml["rs:rawSignal"]["rs:channelLabels"]["rs:label"]

    for i, name in enumerate(channel_names):
        multimodal_data.eeg_channel_mapping[name] = i

    # if debug print N_chan, Fs_EEG, chan_names
    if plot_flag:
        print(
            f"n_channels: {n_channels},\n fs_EEG: {fs},\n chan_names: {channel_names}"
        )

    multimodal_data.fs = fs
    raw_eeg_data = (
        np.fromfile(
            os.path.join(multimodal_data.paths.eeg_directory, f"{file}.raw"),
            dtype="float32",
        )
        .reshape((-1, n_channels))
        .T
    )  # transpose to have channels in rows and samples in columns

    # scale the signal to microvolts
    raw_eeg_data *= 0.0715

    return raw_eeg_data


def _mount_eeg_data(multimodal_data, raw_eeg_data):
    """Mount EEG data to M1 and M2 channels for both caregiver and child.
    Args:
        multimodal_data (MultimodalData): The multimodal data instance containing EEG metadata.
        raw_eeg_data (np.ndarray): The raw EEG data array with shape (n_channels, n_samples).
    """
    channel_mapping = multimodal_data.eeg_channel_mapping
    # mount EEG data to M1 and M2 channels; do it separately for caregiver and
    # child as they have different references
    for channel in multimodal_data.eeg_channel_names_ch:
        if channel in channel_mapping:
            idx = channel_mapping[channel]
            raw_eeg_data[idx, :] -= 0.5 * (
                raw_eeg_data[channel_mapping["M1"], :]
                + raw_eeg_data[channel_mapping["M2"], :]
            )

    for channel in multimodal_data.eeg_channel_names_cg:
        if channel in channel_mapping:
            idx = channel_mapping[channel]
            raw_eeg_data[idx, :] -= 0.5 * (
                raw_eeg_data[channel_mapping["M1_cg"], :]
                + raw_eeg_data[channel_mapping["M2_cg"], :]
            )
    multimodal_data.references = "linked ears montage: (M1+M2)/2"


def _design_eeg_filters(
    multimodal_data: MultimodalData,
    lowcut,
    highcut,
    notch_freq=50,
    notch_q=30,
    filter_type="fir",
    plot_flag=False,
):
    """
    Design notch, low-pass, and high-pass filters for EEG data.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance containing sampling frequency.
        lowcut (float): Low cut-off frequency for high-pass filtering.
        highcut (float): High cut-off frequency for low-pass filtering.
        notch_freq (float, optional): Notch filter frequency. Defaults to 50 Hz.
        notch_q (float, optional): Quality factor for notch filter. Defaults to 30.
        filter_type (str, optional): Type of filter ('fir' or 'iir'). Defaults to 'fir'.
        plot_flag (bool, optional): Whether to plot filter characteristics. Defaults to False.

    Returns:
        tuple: ((b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type)
               Filter coefficients and filter type.
    """
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs=multimodal_data.fs)

    if filter_type == "fir":
        numtaps_low = 201
        b_low = firwin(
            numtaps_low, highcut, fs=multimodal_data.fs, pass_zero="lowpass"
        )
        numtaps_high = 3049
        b_high = firwin(
            numtaps_high, lowcut, fs=multimodal_data.fs, pass_zero="highpass"
        )
        a_low = a_high = 1.0
    else:
        b_low, a_low = butter(
            N=4, Wn=highcut, btype="low", fs=multimodal_data.fs
        )
        b_high, a_high = butter(
            N=4, Wn=lowcut, btype="high", fs=multimodal_data.fs
        )

    if plot_flag:
        print("---- Notch filter characteristics: --------")
        f_max = 60.0
        plot_filter_characteristics(
            b_notch,
            a_notch,
            f=np.arange(0, f_max, 0.01),
            T=0.5,
            Fs=multimodal_data.fs,
            f_lim=(30, f_max),
            db_lim=(-300, 0.1),
        )
        print("---- Low-pass filter characteristics: --------")
        plot_filter_characteristics(
            b_low,
            a=[1],
            f=np.arange(0, multimodal_data.fs / 2, 0.1),
            T=0.5,
            Fs=multimodal_data.fs,
            f_lim=(0, 50),
            db_lim=(-60, 0.1),
        )
        print("---- High-pass filter characteristics: --------")
        plot_filter_characteristics(
            b_high,
            a=[1],
            f=np.arange(0, multimodal_data.fs / 2, 0.01),
            T=1.0,
            Fs=multimodal_data.fs,
            f_lim=(0, 10),
            db_lim=(-60, 0.1),
        )
        # add info about filtering to the multimodal data
    multimodal_data.eeg_filtration.low_pass = highcut
    multimodal_data.eeg_filtration.low_pass_a=a_low
    multimodal_data.eeg_filtration.low_pass_b=b_low
    multimodal_data.eeg_filtration.high_pass = lowcut
    multimodal_data.eeg_filtration.high_pass_a=a_high
    multimodal_data.eeg_filtration.high_pass_b=b_high
    multimodal_data.eeg_filtration.notch_Q = notch_q
    multimodal_data.eeg_filtration.notch_freq = notch_freq
    multimodal_data.eeg_filtration.notch_a=a_notch
    multimodal_data.eeg_filtration.notch_b=b_notch
    multimodal_data.eeg_filtration.type = filter_type

    return (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type


def _apply_filters(
    multimodal_data: MultimodalData, filters, raw_eeg_data, plot_flag=False
):
    """
    Apply designed filters to raw EEG data in place.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance containing channel mapping.
        filters (tuple): Filter coefficients from _design_eeg_filters.
        raw_eeg_data (np.ndarray): Raw EEG data array to filter in place.
        plot_flag (bool, optional): Whether to plot filtered signals. Defaults to False.

    Returns:
        None: Modifies raw_eeg_data in place.
    """
    (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type = filters
    print(f"Applying {filter_type} filters to EEG data.")

    # Filter and separate each channel
    for idx, ch in enumerate(multimodal_data.eeg_channel_names_all()):
        signal = raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :]
        signal = signal - np.mean(signal)  # remove DC offset
        if filter_type == "iir":
            signal = filtfilt(b_notch, a_notch, signal, axis=0)
            signal = filtfilt(b_low, a_low, signal, axis=0)
            signal = filtfilt(b_high, a_high, signal, axis=0)
        else:
            signal = lfilter(b_notch, a_notch, signal, axis=0)
            signal = lfilter(b_low, a_low, signal, axis=0)
            signal = lfilter(b_high, a_high, signal, axis=0)

            delay = (len(b_low) - 1) // 2 + (len(b_high) - 1) // 2
            signal = np.roll(signal, -delay)
            # zero-pad the end to account for the delay introduced by filtering
            signal[-delay:] = 0.0

        raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :] = signal
        # plot filtered signals for debugging
        if plot_flag:
            plt.figure(figsize=(12, 4))
            plt.plot(signal)
            plt.title(f"Filtered signal for channel {ch}")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (uV)")
            plt.show()
    multimodal_data.eeg_filtration.applied = True


def _extract_ecg_data(multimodal_data: MultimodalData, raw_eeg_data):
    """
    Extract and filter ECG data from raw EEG recording.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance to populate with ECG.
        raw_eeg_data (np.ndarray): Raw EEG data containing ECG channels.

    Returns:
        None: Modifies multimodal_data in place, adding ECG data and modality.
    """
    channel_mapping = multimodal_data.eeg_channel_mapping

    # extract and filter the ECG data
    ecg_ch = (
        raw_eeg_data[channel_mapping["EKG1"], :]
        - raw_eeg_data[channel_mapping["EKG2"], :]
    )
    ecg_cg = (
        raw_eeg_data[channel_mapping["EKG1_cg"], :]
        - raw_eeg_data[channel_mapping["EKG2_cg"], :]
    )

    # design filters:
    b_notch, a_notch = iirnotch(50, 30, fs=multimodal_data.fs)
    sos_ecg = butter(5, 0.5, btype="high", output="sos", fs=multimodal_data.fs)
    ecg_ch_filtered = sosfiltfilt(sos_ecg, ecg_ch)
    ecg_ch_filtered = filtfilt(b_notch, a_notch, ecg_ch_filtered)
    ecg_cg_filtered = sosfiltfilt(sos_ecg, ecg_cg)
    ecg_cg_filtered = filtfilt(b_notch, a_notch, ecg_cg_filtered)

    # Store ECG data in DataFrame
    multimodal_data.set_ecg_data(ecg_ch_filtered, ecg_cg_filtered)
    if "ECG" not in multimodal_data.modalities:
        multimodal_data.modalities.append("ECG")

    # compute IBI and add IBI modality
    multimodal_data.set_ibi()
    if "IBI" not in multimodal_data.modalities:
        multimodal_data.modalities.append("IBI")

def _scan_for_events(diode, eeg_fs, plot_flag, threshold=0.75):
    """Scan the diode signal to detect and identify experimental events.

    Processes the raw diode signal to find periods corresponding to
    specific experimental events, such as watching movies or engaging in conversation.
    Binarizes the signal based on a threshold to identify "on" and "off" states,
    then analyzes durations and intervals to classify into predefined event categories.

    The detection logic is tailored to a specific experimental design, expecting
    three movie sessions followed by two conversation sessions.

    Args:
        diode (np.ndarray): The diode signal array.
        eeg_fs (float): Sampling frequency of the EEG/diode signal.
        plot_flag (bool): Whether to plot the detected events for debugging.
        threshold (float, optional): The threshold for binarizing the diode signal,
            relative to its maximum value. Defaults to 0.75.

    Returns:
        tuple: (events, thresholded_diode)
            - events (list[dict]): List of dictionaries with 'name', 'start', 'duration' keys.
            - thresholded_diode (np.ndarray): Binary array of thresholded diode signal.
    """
    # Binarize the diode signal: values above the threshold become 1, others 0.
    thresholded_diode = ((diode / (threshold * np.max(diode))) > 1).astype(
        float
    )

    # Find rising (1) and falling (-1) edges in the binarized signal.
    # Collect the sample indices of all rising and falling edges.
    up_down_events = np.where(np.abs(np.diff(thresholded_diode)) == 1)[
        0
    ].tolist() + [len(diode)]

    events = [
        {"name": name}
        for name in ["Brave", "Peppa", "Incredibles", "Talk_1", "Talk_2"]
    ]

    found_movies = found_talks = 0
    queue = deque(maxlen=100)

    # Process pairs of up/down events to identify event durations and
    # intervals.
    for i in range(len(up_down_events) // 2):
        start = up_down_events[2 * i]
        duration = up_down_events[2 * i + 1] - up_down_events[2 * i]
        # Calculate the time until the next event starts.
        following_space = up_down_events[2 * i + 2] - up_down_events[2 * i + 1]
        queue.append(start)
        # Maintain a queue of recent event start times
        while queue[0] < start - 4 * eeg_fs:  # last 4 seconds
            queue.popleft()
        # Detect movie events based on their duration and number of recent
        # spikes
        if (
            duration > 55 * eeg_fs and len(queue) > 1
        ):  # movie events longer than 0:55
            events[len(queue) - 2]["start"] = (
                queue[0] + 1
            ) / eeg_fs  # add 1 sample due to shift caused by diff
            events[len(queue) - 2]["duration"] = (
                up_down_events[2 * i + 1] - queue[0]
            ) / eeg_fs
            found_movies += 1
        if found_movies > 3:
            raise ValueError(
                "More than 3 events detected, something is wrong."
            )
        # Detect talk events based on their position relative to movie events
        if (
            found_movies == 3
            and duration < 2 * eeg_fs
            and following_space > 175 * eeg_fs
        ):  # talk events longer than 2:55
            if found_talks < 2:
                event_index = found_movies + found_talks
                # add 1 sample due to shift caused by diff
                events[event_index]["start"] = (
                    up_down_events[2 * i + 1] + 1
                ) / eeg_fs
                events[event_index]["duration"] = following_space / eeg_fs
                found_talks += 1
            else:
                raise ValueError(
                    "More than 2 talks detected, something is wrong."
                )

    if plot_flag:
        _plot_scanned_events(
            threshold,
            diode,
            thresholded_diode,
            np.diff(thresholded_diode),
            events,
            eeg_fs,
        )

    return events, thresholded_diode


def _plot_scanned_events(
    threshold, diode, thresholded_diode, derivative, events, eeg_fs
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        diode / (threshold * np.max(diode)),
        "b",
        label="Diode Signal normalized by threshold",
    )
    plt.plot(thresholded_diode, "r", label="Diode Signal Thresholded")
    plt.title("Diode Signal with events")
    plt.xlabel("Samples")
    plt.ylabel("Signal Value")
    plt.plot((derivative == 1).astype(int), "g", label="Up Events")
    plt.plot((derivative == -1).astype(int), "m", label="Down Events")
    for event in events:
        if "start" in event:
            plt.plot(event["start"] * eeg_fs, 1.2, "ko", markersize=10)
            plt.text(event["start"] * eeg_fs, 1.25, event["name"], rotation=45)
    plt.legend()
    plt.show()


# --------------  Save and load multimodal data -----------------


def load_output_data(filename):
    """
    Load saved MultimodalData from a joblib file.

    Args:
        filename (str): Path to the joblib file to load.

    Returns:
        MultimodalData or None: The loaded multimodal data instance, or None if file not found.
    """
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")
        return None


def save_to_file(multimodal_data: MultimodalData, output_dir):
    """
    Save MultimodalData instance to a joblib file.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance to save.
        output_dir (str): Directory path where the file will be saved.

    Returns:
        None: Saves file to {output_dir}/{dyad_id}.joblib
    """
    joblib.dump(multimodal_data, output_dir + f"/{multimodal_data.id}.joblib")

def get_eeg_data(df, who: str) -> tuple[np.ndarray | None, list]:
    """Returns EEG data and channel names for specified participant.
    
    Args:
        df: MultimodalData.data DataFrame instance containing EEG data
        who: Participant identifier ('ch' for child, 'cg' for caregiver)
        
    Returns:
        Tuple of (eeg_data, channel_names) where:
        - eeg_data: 2D array [n_channels x n_samples] or None if no data
        - channel_names: List of clean channel names (e.g., ['Fp1', 'Fp2', ...])
    """
    prefix = f'EEG_{who}_'
    cols = [col for col in df.columns if col.startswith(prefix)]
    
    if not cols:
        return None, []
    
    # Extract data as 2D array
    eeg_data = df[cols].values.T
    
    # Strip prefix to get clean channel names
    channel_names = [col.replace(prefix, '') for col in cols]
    
    return eeg_data, channel_names
    
def export_eeg_to_mne_raw(multimodal_data: MultimodalData, who: str, events=None, times=None) -> mne.io.Raw:
    """
    Export EEG data from MultimodalData to MNE Raw object.

    Args:
        multimodal_data: MultimodalData instance containing EEG data
        who: 'ch' for child, 'cg' for caregiver 
        events: Optional event to include by name (e.g., one of  ['Brave', 'Peppa', 'Incredibles', 'Talk_1', 'Talk_2'])
        times: tuple; Optional range of time to include (in seconds)
        Note: if both events and times are provided, times take precedence.
    Returns:
        raw: MNE Raw object with EEG data
    """
    if events is not None:
        # Handle both single event (string) and multiple events (list)
        if isinstance(events, str):
            selected_data = multimodal_data.data[
                multimodal_data.data["events"] == events
            ]
            first_sample_index = int(selected_data["time"].iloc[0] * multimodal_data.fs)
        # else:
        #     selected_data = multimodal_data.data[
        #         multimodal_data.data["events"].isin(events)
        #     ]
    elif times is not None:
        start_time, end_time = times
        selected_data = multimodal_data.data[
            (multimodal_data.data["time"] >= start_time)
            & (multimodal_data.data["time"] <= end_time)
        ]
        first_sample_index = int(start_time * multimodal_data.fs)
    else:
        selected_data = multimodal_data.data   
        first_sample_index = multimodal_data.data["time_idx"].iloc[0]    
        # now seletd channel data based on the fact that they start with EEG and who
    eeg_data, channel_names = get_eeg_data(df = selected_data, who=who)
    

    if eeg_data is None:
        raise ValueError(f"No EEG data found for {who}, events: {events}, times: {times}")

    info = mne.create_info(
        ch_names=channel_names, 
        sfreq=multimodal_data.fs, 
        ch_types='eeg'
    )
    
    raw = mne.io.RawArray(eeg_data, info, first_samp=first_sample_index)
    
    # Mark the data as pre-filtered using MNE's internal method
    # We need to use the private _update_times method or set these via the dictionary update
    if multimodal_data.eeg_filtration.applied:
        # Update the info dictionary with filter information using dict.update to bypass validation
        with raw.info._unlock():
            if multimodal_data.eeg_filtration.high_pass:
                raw.info['highpass'] = float(multimodal_data.eeg_filtration.high_pass)
            if multimodal_data.eeg_filtration.low_pass:
                raw.info['lowpass'] = float(multimodal_data.eeg_filtration.low_pass)
    
    # Add filter description to raw object
    if multimodal_data.eeg_filtration.applied:
        filter_desc = (
            f"EEG filtered with {multimodal_data.eeg_filtration.type} filters: "
            f"highpass={multimodal_data.eeg_filtration.high_pass}Hz, "
            f"lowpass={multimodal_data.eeg_filtration.low_pass}Hz, "
            f"notch={multimodal_data.eeg_filtration.notch_freq}Hz (Q={multimodal_data.eeg_filtration.notch_Q}). "
            f"Reference: {multimodal_data.references}"
        )
        raw.info['description'] = filter_desc
    
    # Set montage for electrode positions
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)

    return raw
# --------------  Load eye-tracking data -----------------


def _build_et_file_paths(et_path: str, task_id: str, member: str) -> dict:
    """
    Build file paths for eye-tracker data for a specific task and dyad member.

    Args:
        et_path: Base path to ET data directory
        task_id: Task identifier ('000', '001', '002')
        member: Dyad member ('child' or 'caregiver')

    Returns:
        Dictionary mapping data types to file paths
    """
    base_export_path = os.path.join(et_path, member, task_id, "exports", "000")
    prefix = "ch" if member == "child" else "cg"

    paths = {
        f"annotations_{task_id}": os.path.join(
            base_export_path, "annotations.csv"
        ),
        f"{prefix}_pupil_{task_id}": os.path.join(
            base_export_path, "pupil_positions.csv"
        ),
    }

    # Movies task (000) has additional gaze position and blinks data
    if task_id == "000":
        paths[f"{prefix}_pos_{task_id}"] = os.path.join(
            base_export_path,
            "surfaces",
            "gaze_positions_on_surface_Surface 1.csv",
        )
        paths[f"{prefix}_blinks_{task_id}"] = os.path.join(
            base_export_path, "blinks.csv"
        )

    return paths


def _check_et_files_exist(file_paths: dict) -> tuple[bool, list]:
    """
    Check if ET data files exist.

    Args:
        file_paths: Dictionary mapping data types to file paths

    Returns:
        Tuple of (all_exist: bool, missing_files: list)
    """
    missing_files = [
        name for name, path in file_paths.items() if not os.path.exists(path)
    ]
    return len(missing_files) == 0, missing_files


def _load_et_task_data(
    file_paths: dict, task_id: str, member: str, min_max_times: list
) -> dict:
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
    prefix = "ch" if member == "child" else "cg"
    loaded_data = {}

    # Load annotations
    ann_key = f"annotations_{task_id}"
    if ann_key in file_paths:
        loaded_data[ann_key] = pd.read_csv(file_paths[ann_key])

    # Load pupil data
    pupil_key = f"{prefix}_pupil_{task_id}"
    if pupil_key in file_paths:
        pupil_df = pd.read_csv(file_paths[pupil_key])
        loaded_data[pupil_key] = pupil_df
        min_max_times.append(
            (
                pupil_df["pupil_timestamp"].min(),
                pupil_df["pupil_timestamp"].max(),
            )
        )

    # Load gaze position data (movies task only)
    pos_key = f"{prefix}_pos_{task_id}"
    if pos_key in file_paths:
        pos_df = pd.read_csv(file_paths[pos_key])
        loaded_data[pos_key] = pos_df
        min_max_times.append(
            (pos_df["gaze_timestamp"].min(), pos_df["gaze_timestamp"].max())
        )

    # Load blinks data (movies task only)
    blinks_key = f"{prefix}_blinks_{task_id}"
    if blinks_key in file_paths:
        loaded_data[blinks_key] = pd.read_csv(file_paths[blinks_key])

    return loaded_data


def _process_et_data_to_dataframe(
    et_df: pd.DataFrame,
    loaded_data: dict,
    task_flags: dict,
    task_names: list,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=1,
    pupil_model_confidence=0.9,
    Fs=1024,
) -> None:
    """
    Process loaded ET data into the main ET dataframe.

    Args:
        et_df: DataFrame to populate with ET data
        loaded_data: Dictionary containing all loaded ET DataFrames
        task_flags: Dictionary with flags indicating which tasks/members have data
        task_names: List of task identifiers ['000', '001', '002']
        median_filter_size: Size of the median filter for smoothing
        low_pass_et_order: Order of the low-pass filter for ET data
        et_pos_cutoff: Cutoff frequency for position data low-pass filter
        et_pupil_cutoff: Cutoff frequency for pupil data low-pass filter
        Fs: Sampling frequency of ET data
    """
    # Process movies task (000) if available
    if task_flags.get("movies_ch") or task_flags.get("movies_cg"):
        if "annotations_000" in loaded_data:
            et.process_event_et(loaded_data["annotations_000"], et_df)

    if task_flags.get("movies_ch"):
        if "ch_pos_000" in loaded_data:
            et.process_pos(
                loaded_data["ch_pos_000"],
                et_df,
                "ch",
                median_filter_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pos_cutoff,
                Fs=Fs,
            )
        if "ch_pupil_000" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_000"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )
        if "ch_blinks_000" in loaded_data:
            et.process_blinks(loaded_data["ch_blinks_000"], et_df, "ch")

    if task_flags.get("movies_cg"):
        if "cg_pos_000" in loaded_data:
            et.process_pos(
                loaded_data["cg_pos_000"],
                et_df,
                "cg",
                median_filter_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pos_cutoff,
                Fs=Fs,
            )
        if "cg_pupil_000" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_000"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )
        if "cg_blinks_000" in loaded_data:
            et.process_blinks(loaded_data["cg_blinks_000"], et_df, "cg")

    # Process talk1 task (001)
    if task_flags.get("talk1_ch") or task_flags.get("talk1_cg"):
        if "annotations_001" in loaded_data:
            et.process_event_et(loaded_data["annotations_001"], et_df, "Talk1")

    if task_flags.get("talk1_ch"):
        if "ch_pupil_001" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_001"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    if task_flags.get("talk1_cg"):
        if "cg_pupil_001" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_001"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    # Process talk2 task (002)
    if task_flags.get("talk2_ch") or task_flags.get("talk2_cg"):
        if "annotations_002" in loaded_data:
            et.process_event_et(loaded_data["annotations_002"], et_df, "Talk2")

    if task_flags.get("talk2_ch"):
        if "ch_pupil_002" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_002"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    if task_flags.get("talk2_cg"):
        if "cg_pupil_002" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_002"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )


def load_et_data(
    multimodal_data,
    dyad_id,
    folder_et,
    interpolate_et_during_blinks_threshold=0,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=1,
    pupil_model_confidence=0.9,
    plot_flag=False,
):
    """Load eye-tracking data from CSV files and integrate into the MultimodalData instance.

    Args:
        multimodal_data (MultimodalData): Instance of MultimodalData to populate with ET data.
        dyad_id (str): Identifier for the dyad.
        folder_et (str): Base path to ET data directory.
        interpolate_et_during_blinks_threshold (float, optional): Confidence threshold for interpolating ET data during blinks. 0 means no interpolation. Defaults to 0.
        median_filter_size (int, optional): Size of the median filter for ET data processing. Defaults to 64.
        low_pass_et_order (int, optional): Order of the low-pass filter for ET data processing. Defaults to 351.
        et_pos_cutoff (float, optional): Cutoff frequency for ET position data low-pass filter. Defaults to 128.
        et_pupil_cutoff (float, optional): Cutoff frequency for ET pupil data low-pass filter. Defaults to 1.
        pupil_model_confidence (float, optional): Confidence level for 3D pupil model. Defaults to 0.9.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.

    Returns:
        MultimodalData: The multimodal data instance with integrated ET data.
    """
    if multimodal_data is None:
        multimodal_data = MultimodalData()
        multimodal_data.id = dyad_id
    # Configuration for tasks: 000=movies, 001=talk1, 002=talk2
    tasks = [
        {"id": "000", "name": "movies"},
        {"id": "001", "name": "talk1"},
        {"id": "002", "name": "talk2"},
    ]
    members = ["child", "caregiver"]

    # Build file paths and check availability for each task and member
    task_flags = {}
    min_max_times = []
    loaded_data = {}

    for task in tasks:
        for member in members:
            # Build file paths
            file_paths = _build_et_file_paths(folder_et, task["id"], member)

            # Check if files exist
            all_exist, missing = _check_et_files_exist(file_paths)

            # Set flag for this task/member combination
            flag_key = f"{task['name']}_{'ch' if member == 'child' else 'cg'}"
            task_flags[flag_key] = all_exist

            if not all_exist:
                print(
                    f"Warning: Missing ET files for {multimodal_data.id} {member} {task['name']}: {missing}"
                )
            else:
                # Load data if all files exist
                task_data = _load_et_task_data(
                    file_paths, task["id"], member, min_max_times
                )
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
        print(f"Setting fs to the default Fs of EEG: {multimodal_data.fs}")

    # Find the overall min and max times across all tasks and members
    overall_min_time = min([t[0] for t in min_max_times])
    overall_max_time = max([t[1] for t in min_max_times])

    print(f"ET time range: {overall_min_time:.2f}s to {overall_max_time:.2f}s")

    # Create time vector
    et_df["time"] = np.arange(
        overall_min_time, overall_max_time, 1 / multimodal_data.fs
    )
    et_df["time_idx"] = (et_df["time"] * multimodal_data.fs).astype(int)

    # Process loaded data into the dataframe
    _process_et_data_to_dataframe(
        et_df,
        loaded_data,
        task_flags,
        [t["id"] for t in tasks],
        median_filter_size=median_filter_size,
        low_pass_et_order=low_pass_et_order,
        et_pos_cutoff=et_pos_cutoff,
        et_pupil_cutoff=et_pupil_cutoff,
        pupil_model_confidence=pupil_model_confidence,
        Fs=multimodal_data.fs,
    )

    # Align ET time to EEG time by subtracting the time of the first event;
    # We consider the time of first event in all data series to be 0;
    # reset time_idx accordingly

    # reset time column to be consistent with the first movie event start at time zero; this is needed to align with EEG data later; accordingly reset time_idx
    # in the column 'ET_event' find the first occurance of one of 'Incredibles','Peppa','Brave',
    # 'm3'; reset to the corresponding time
    first_movie_event = et_df[et_df["ET_event"].isin(["Incredibles", "Peppa", "Brave"])]["time"].min()
    print(
        f"Reseting the ET time to the start of {et_df[et_df['time'] == first_movie_event]['ET_event'].iloc[0]}"
    )
    et_df["time"] = et_df["time"] - first_movie_event
    et_df["time_idx"] = (et_df["time"] * multimodal_data.fs).astype(int)

    #  merging ET data into the main dataframe
    if multimodal_data.data.empty:
        multimodal_data.data = et_df.copy()
    else:
        multimodal_data.data = pd.merge(
            multimodal_data.data, et_df, how="outer", on="time_idx"
        )
        # After merge, time_x and time_y are created; use time_x (from EEG) as
        # primary, fill missing with time_y (from ET)
        multimodal_data.data["time"] = multimodal_data.data["time_x"].fillna(
            multimodal_data.data["time_y"]
        )
        multimodal_data.data = multimodal_data.data.drop(
            columns=["time_x", "time_y"]
        )
        multimodal_data.data = multimodal_data.data.replace(np.nan, None)

    # correct alignment of x, y, blinks and pupil columns by delta_time
    # estimated to be -0.3s
    delta_time = -0.3
    for col in multimodal_data.data.columns:
        if any(
            keyword in col for keyword in ["x", "y", "blinks", "diameter3d"]
        ):
            multimodal_data.data[col] = multimodal_data.data[col].shift(
                int(delta_time * multimodal_data.fs)
            )
    if interpolate_et_during_blinks_threshold > 0:
        # correct x, y, and diameter3d columns by interpolating the values
        # during blinks, separately for child and caregiver
        for member in ["ch", "cg"]:
            blink_col = f"ET_{member}_blinks"
            print(f"Processing member: {member}, blink column: {blink_col}")
            for col in multimodal_data.data.columns:
                if any(
                    keyword in col
                    for keyword in [
                        f"ET_{member}_x",
                        f"ET_{member}_y",
                        f"ET_{member}_pupil",
                    ]
                ):
                    # Convert to numeric dtype to enable interpolation
                    multimodal_data.data[col] = pd.to_numeric(
                        multimodal_data.data[col], errors="coerce"
                    )
                    blink_indices = multimodal_data.data.index[
                        multimodal_data.data[blink_col]
                        > interpolate_et_during_blinks_threshold
                    ].tolist()
                    multimodal_data.data.loc[blink_indices, col] = np.nan
                    multimodal_data.data[col] = multimodal_data.data[
                        col
                    ].interpolate(method="linear", limit_direction="both")

    multimodal_data.modalities.append("ET")
    return multimodal_data

