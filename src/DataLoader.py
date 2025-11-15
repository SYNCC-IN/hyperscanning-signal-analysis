import os
from collections import deque

import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, butter, decimate, sosfiltfilt, iirnotch
import neurokit2 as nk
import joblib


class DataLoader:

    def __init__(self, dyad_id, plot_flag):
        """DataLoader class for loading and processing Warsaw pilot data.
        The constructor initializes the DataLoader with the given id and an empty list of possible modalities.
        The data related to the modalities will be added by setter methods.
        Retrieving the data will be done by getter methods.
        """
        self.output_dir = "../DATA/OUT"
        self.plot_flag = plot_flag
        self.fs = {}
        self.time = {}
        self.data = {}
        self.channels = {}
        self.folder = {}
        self.channel_names = {}
        self.id = dyad_id
        self.modalities = []  # list of modalities already loaded: 'EEG', 'H10', 'ET'
        self.events = []

    def set_eeg_data(self, folder_eeg, debug_flag=False):
        """Set the EEG data for the DataLoader instance by loading and filtering the Warsaw pilot data.
        We assume data were recorded as multiplexed signals in SVAROG system format.
        We also assume specific channel names for child and caregiver EEG data, as specified below.
        Args:
            folder_eeg (str): Path to the folder containing the EEG data files.
            debug_flag (bool): Whether to plot intermediate results for debugging/visualization.
        """
        self.folder['EEG'] = folder_eeg
        self.channel_names['EEG'] = {}
        # define EEG channels for child and caregiver
        self.channel_names['EEG']['ch'] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4',
                                           'T4', 'M2', 'T5',
                                           'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        self.channel_names['EEG']['cg'] = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'M1_cg',
                                           'T3_cg', 'C3_cg',
                                           'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg',
                                           'T6_cg', 'O1_cg',
                                           'O2_cg']
        self.channel_names['EEG']['all'] = self.channel_names['EEG']['ch'] + self.channel_names['EEG']['cg']
        self._read_raw_svarog_data()  # load the raw data, works inplace, no return value

    def _read_raw_svarog_data(self, lowcut=4.0, highcut=40.0, q=8):
        file = self.id + ".obci"  # SVAROG files have .obci extension
        # read meta information from xml file
        with open(os.path.join(self.folder['EEG'], f"{file}.xml")) as fd:
            xml = xmltodict.parse(fd.read())

        n_channels = int(xml['rs:rawSignal']['rs:channelCount'])
        fs_eeg = int(float(xml['rs:rawSignal']['rs:samplingFrequency']))
        chan_names = xml['rs:rawSignal']['rs:channelLabels']['rs:label']
        # create a dictionary which maps channel names and their indexes
        channels = {}
        for i, name in enumerate(chan_names):
            channels[name] = i
        self.channels['EEG'] = channels

        # if debug print N_chan, Fs_EEG, chan_names
        if self.plot_flag:
            print(f"N_chan: {n_channels},\n Fs_EEG: {fs_eeg},\n ChanNames: {chan_names}")

        self.fs['EEG'] = fs_eeg
        self.fs['ECG'] = fs_eeg  # ECG data is sampled at the same frequency as EEG data
        # read raw data from .raw file
        data = np.fromfile(os.path.join(self.folder['EEG'], f"{file}.raw"), dtype='float32').reshape((-1, n_channels))
        data = data.T  # transpose to have channels in rows and samples in columns

        # extract diode signal for event detection before any scaling and filtering
        self.diode = data[channels['Diode'], :]
        # scan for events
        self.events = self._scan_for_events(threshold=0.75)
        print(f"Detected events: {self.events}")
        # # scale the signal to microvolts
        # data *= 0.0715
        #
        # # mount EEG data to M1 and M2 channels and filter the data
        # data = self._mount_eeg_data(data, channels)
        #
        # # filter and decimate the EEG modality data
        # self._filter_decimate_and_set_eeg_signals(data, lowcut=lowcut, highcut=highcut, q=q)
        #
        # # set the ECG modality with ECG signals
        # self._extract_ecg_data(data, channels)
        #
        # # set the IBI modality computed from Porti ECG signals; IBIs are  interpolated to Fs_IBI [Hz]
        # # self._compute_IBI(self.data['ECG'])
        # self._compute_ibi()

    def _mount_eeg_data(self, data, channels):
        # mount EEG data to M1 and M2 channels; do it separately for caregiver and child as they have different references
        for ch in self.channel_names['EEG']['ch']:
            if ch in channels:
                idx = channels[ch]
                data[idx, :] = data[idx, :] - 0.5 * (data[channels['M1'], :] + data[channels['M2'], :])
        for ch in self.channel_names['EEG']['cg']:
            if ch in channels:
                idx = channels[ch]
                data[idx, :] = data[idx, :] - 0.5 * (data[channels['M1_cg'], :] + data[channels['M2_cg'], :])
        # adjust channel lists by removeing channels M1 and M2 from the caregiver and child EEG channels, as they will not be used after linked ears montage
        self.channel_names['EEG']['ch'] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
                                           'T5', 'P3', 'Pz',
                                           'P4', 'T6', 'O1', 'O2']
        self.channel_names['EEG']['cg'] = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'T3_cg',
                                           'C3_cg', 'Cz_cg',
                                           'C4_cg', 'T4_cg', 'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg',
                                           'O2_cg']
        self.channel_names['EEG']['all'] = self.channel_names['EEG']['ch'] + self.channel_names['EEG']['cg']
        return data

    def _filter_decimate_and_set_eeg_signals(self, data, lowcut=4.0, highcut=40.0, q=8):
        """
        Coordinates the full EEG processing pipeline: filtering, separating,
        decimating, and updating the object state.
        """
        # Task 1: Design filters
        filters = self._design_eeg_filters(lowcut, highcut)

        # Task 2: Apply filters and separate signals
        (eeg_cg, channels_cg), (eeg_ch, channels_ch) = self._apply_filters_and_separate(data, filters)

        # Task 3: Decimate signals
        signal_cg, signal_ch = self._decimate_signals(eeg_cg, eeg_ch, q)

        # Task 4: Update object state
        self._update_eeg_state(signal_cg, signal_ch, channels_cg, channels_ch, q)

    # --- HELPER METHODS ---

    def _design_eeg_filters(self, lowcut, highcut):
        """
        Task 1: Designs notch, low-pass, and high-pass filters.
        Returns a tuple of filter coefficients.
        """
        fs = self.fs['EEG']
        b_notch, a_notch = iirnotch(50, 30, fs=fs)
        b_low, a_low = butter(N=4, Wn=highcut, btype='low', fs=fs)
        b_high, a_high = butter(N=4, Wn=lowcut, btype='high', fs=fs)

        return (b_notch, a_notch), (b_low, a_low), (b_high, a_high)

    def _apply_filters_and_separate(self, data, filters):
        """
        Task 2: Applies filters to raw data and separates into 'cg' and 'ch' groups.
        Returns filtered data and channel mappings for both groups.
        """
        (b_notch, a_notch), (b_low, a_low), (b_high, a_high) = filters

        # Initialize arrays for filtered EEG signals
        eeg_cg = np.zeros((len(self.channel_names['EEG']['cg']), data.shape[1]))
        channels_cg = {}
        eeg_ch = np.zeros((len(self.channel_names['EEG']['ch']), data.shape[1]))
        channels_ch = {}

        chan_counter_cg = 0
        chan_counter_ch = 0

        # Filter and separate each channel
        for idx, ch in enumerate(self.channels['EEG']):
            signal = data[idx, :].copy()
            signal = filtfilt(b_notch, a_notch, signal, axis=0)
            signal = filtfilt(b_low, a_low, signal, axis=0)
            signal = filtfilt(b_high, a_high, signal, axis=0)

            if ch in self.channel_names['EEG']['cg']:
                eeg_cg[chan_counter_cg, :] = signal
                channels_cg[ch] = chan_counter_cg
                chan_counter_cg += 1
            if ch in self.channel_names['EEG']['ch']:
                eeg_ch[chan_counter_ch, :] = signal
                channels_ch[ch] = chan_counter_ch
                chan_counter_ch += 1

        return (eeg_cg, channels_cg), (eeg_ch, channels_ch)

    def _decimate_signals(self, eeg_cg, eeg_ch, q):
        """
        Task 3: Decimates the filtered 'cg' and 'ch' signals.
        Returns the decimated signals.
        """
        signal_cg = decimate(eeg_cg, q, axis=-1)
        signal_ch = decimate(eeg_ch, q, axis=-1)
        return signal_cg, signal_ch

    def _update_eeg_state(self, signal_cg, signal_ch, channels_cg, channels_ch, q):
        """
        Task 4: Updates the object's state (self) with the processed data,
        new sampling frequency, and new time vector.
        """
        self.channels['EEG'] = {'cg': channels_cg, 'ch': channels_ch}
        self.data['EEG'] = {'cg': signal_cg, 'ch': signal_ch}

        # Calculate and set new sampling frequency
        new_fs = self.fs['EEG'] // q
        self.fs['EEG'] = new_fs

        # Calculate and set new time vector
        num_samples = signal_cg.shape[1]
        self.time['EEG'] = np.arange(0, num_samples / new_fs, 1 / new_fs)

        # Add modality if it's not already listed
        if 'EEG' not in self.modalities:
            self.modalities.append('EEG')

    def _extract_ecg_data(self, data, channels):
        t_ecg = np.arange(0, data.shape[1] / self.fs['ECG'],
                          1 / self.fs['ECG'])  # time vector for the ECG data in seconds

        # extract and filter the ECG data
        ecg_ch = data[channels['EKG1'], :] - data[channels['EKG2'], :]
        ecg_cg = data[channels['EKG1_cg'], :] - data[channels['EKG2_cg'], :]

        # design filters:
        b_notch, a_notch = iirnotch(50, 30, fs=self.fs['ECG'])
        sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=self.fs['ECG'])
        ecg_ch_filtered = sosfiltfilt(sos_ecg, ecg_ch)
        ecg_ch_filtered = filtfilt(b_notch, a_notch, ecg_ch_filtered)
        ecg_cg_filtered = sosfiltfilt(sos_ecg, ecg_cg)
        ecg_cg_filtered = filtfilt(b_notch, a_notch, ecg_cg_filtered)
        self.data['ECG'] = {'ch': ecg_ch_filtered, 'cg': ecg_cg_filtered}
        self.time['ECG'] = t_ecg
        self.modalities.append('ECG')

    def _compute_ibi(self, fs_ibi=4):
        # interpolate IBI signals from ECG data
        self.fs['IBI'] = fs_ibi
        ibi_ch_interp, t_ibi_ch = self._interpolate_ibi_signals(self.data['ECG']['ch'], self.fs['ECG'])
        ibi_cg_interp, _ = self._interpolate_ibi_signals(self.data['ECG']['cg'], self.fs['ECG'])

        self.modalities.append('IBI')
        # truncate the IBI signals are of the same length
        min_length = min(len(ibi_ch_interp), len(ibi_cg_interp))
        self.time['IBI'] = t_ibi_ch[:min_length]
        # use the time vector for the child IBI as it is the same length as the caregiver IBI
        self.data['IBI'] = {'ch': ibi_ch_interp[:min_length], 'cg': ibi_cg_interp[:min_length]}

    def _interpolate_ibi_signals(self, ecg, label=''):
        # Extract R-peaks location
        _, info_ecg = nk.ecg_process(ecg, sampling_rate=self.fs['ECG'], method='neurokit')
        r_peaks = info_ecg["ECG_R_Peaks"]
        ibi = np.diff(r_peaks) / self.fs['ECG'] * 1000  # IBI in ms
        t = np.cumsum(ibi) / 1000  # time vector for the IBI signals [s]
        t_ecg = np.arange(0, t[-1], 1 / self.fs['IBI'])  # time vector for the interpolated IBI signals
        cs = CubicSpline(t, ibi)
        ibi_interp = cs(t_ecg)
        if self.plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(t_ecg, ibi_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return ibi_interp, t_ecg

    # method for load Warsaw_Data_Frame.csv
    def _load_csv_data(self, csv_file):
        pass

    def _scan_for_events(self, threshold=0.75):
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
        thresholded_diode = ((self.diode / (threshold * np.max(self.diode))) > 1).astype(float)
            
        # Find rising (1) and falling (-1) edges in the binarized signal.
        # Collect the sample indices of all rising and falling edges.
        up_down_events = np.where(np.abs(np.diff(thresholded_diode)) == 1)[0].tolist() + [len(self.diode)]

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
            while queue[0] < start - 4 * self.fs['EEG']: # last 4 seconds
                queue.popleft()
            # Detect movie events based on their duration and number of recent spikes
            if duration > 55 * self.fs['EEG'] and len(queue) > 1: # movie events longer than 0:55
                events[len(queue) - 2]['start'] = queue[0] / self.fs['EEG']
                events[len(queue) - 2]['duration'] = (up_down_events[2 * i + 1] - queue[0]) / self.fs['EEG']
                found_movies += 1
            if found_movies > 3:
                raise ValueError("More than 3 events detected, something is wrong.")
            # Detect talk events based on their position relative to movie events
            if found_movies == 3 and duration < 2 * self.fs['EEG'] and following_space > 175 * self.fs['EEG'] :# talk events longer than 2:55
                if found_talks < 2:
                    event_index = found_movies + found_talks
                    events[event_index]['start'] = up_down_events[2 * i + 1] / self.fs['EEG']
                    events[event_index]['duration'] = following_space / self.fs['EEG']
                    found_talks += 1
                else:
                    raise ValueError("More than 2 talks detected, something is wrong.")

        if self.plot_flag:
            self.plot_scanned_events(threshold, thresholded_diode, np.diff(thresholded_diode), events, self.fs['EEG'])

        return events

    def plot_scanned_events(self, threshold, thresholded_diode, derivative, events, fs_eeg):
        plt.figure(figsize=(12, 6))
        plt.plot(self.diode / (threshold * np.max(self.diode)), 'b', label='Diode Signal normalized by threshold')
        plt.plot(thresholded_diode, 'r', label='Diode Signal Thresholded')
        plt.title('Diode Signal with events')
        plt.xlabel('Samples')
        plt.ylabel('Signal Value')
        plt.plot((derivative == 1).astype(int), 'g', label='Up Events')
        plt.plot((derivative == -1).astype(int), 'm', label='Down Events')
        for event in events:
            if 'start' in event:
                plt.plot(event['start'] * fs_eeg, 1.2, 'ko', markersize=10)
                plt.text(event['start'] * fs_eeg, 1.25, event['name'], rotation=45)
        plt.legend()

    @staticmethod
    def load_output_data(filename):
        try:
            results = joblib.load(filename)
            return results
        except FileNotFoundError:
            print(f"File not found {filename}")

    def save_to_file(self):
        joblib.dump(self, self.output_dir + f"/{self.id}.joblib")
