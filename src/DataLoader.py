import os
import numpy as np
import xmltodict
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import filtfilt, butter, decimate, sosfiltfilt, iirnotch
import neurokit2 as nk
import joblib


class DataLoader:

    def __init__(self, id, plot_flag):
        '''DataLoader class for loading and processing Warsaw pilot data.
        The constructor initializes the DataLoader with the given id and an empty list of possible modalities.
        The data related to the modalities will be added by setter methods.
        Retrieving the data will be done by getter methods.
            '''
        self.output_dir = "../DATA/OUT"
        self.plot_flag = plot_flag
        self.fs = {}
        self.time = {}
        self.data = {}
        self.channels = {}
        self.folder = {}
        self.channel_names = {}
        self.id = id
        self.modalities = []  # list of modalities already loaded: 'EEG', 'H10', 'ET'
        self.events = []

    def set_eeg_data(self, folder_eeg, debug_flag=False):
        '''Set the EEG data for the DataLoader instance by loading and filtering the Warsaw pilot data.
        We assume data were recorded as multiplexed signals in SVAROG system format.
        We also assume specific channel names for child and caregiver EEG data, as specified below.
        Args:
            folder_eeg (str): Path to the folder containing the EEG data files.
            debug_flag (bool): Whether to plot intermediate results for debugging/visualization.
        '''
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
        # read meta informations from xml file
        with open(os.path.join(self.folder['EEG'], f"{file}.xml")) as fd:
            xml = xmltodict.parse(fd.read())

        N_ch = int(xml['rs:rawSignal']['rs:channelCount'])
        Fs_EEG = int(float(xml['rs:rawSignal']['rs:samplingFrequency']))
        ChanNames = xml['rs:rawSignal']['rs:channelLabels']['rs:label']
        # create a dictionary which maps channel names and their indexes
        channels = {}
        for i, name in enumerate(ChanNames):
            channels[name] = i
        self.channels['EEG'] = channels

        # if debug print N_chan, Fs_EEG, ChanNames
        if self.plot_flag:
            print(f"N_chan: {N_ch},\n Fs_EEG: {Fs_EEG},\n ChanNames: {ChanNames}")

        self.fs['EEG'] = Fs_EEG
        self.fs['ECG'] = Fs_EEG  # ECG data is sampled at the same frequency as EEG data
        # read raw data from .raw file
        data = np.fromfile(os.path.join(self.folder['EEG'], f"{file}.raw"), dtype='float32').reshape((-1, N_ch))
        data = data.T  # transpose to have channels in rows and samples in columns

        # extract diode signal for event detection before any scaling and filtering
        self.diode = data[channels['Diode'], :]
        # scan for events
        self.events = self._scan_for_events(threshold=20000)
        # scale the signal to microvolts
        data *= 0.0715

        # mount EEG data to M1 and M2 channels and filter the data
        data = self._mount_EEG_data(data, channels)

        # filter and decimate the EEG modality data
        self._filter_decimate_and_set_EEG_signals(data, lowcut=lowcut, highcut=highcut, q=q)

        # set the ECG modality with ECG signals
        self._extract_ECG_data(data, channels)

        # set the IBI modality computed from Porti ECG signals; IBIs are  interpolated to Fs_IBI [Hz]
        # DO SPRAWDZENIA
        # self._compute_IBI(self.data['ECG'])
        self._compute_IBI()

    def _mount_EEG_data(self, data, channels):
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

    def _filter_decimate_and_set_EEG_signals(self, data, lowcut=4.0, highcut=40.0, q=8):

        # design EEG filters
        b_notch, a_notch = iirnotch(50, 30, fs=self.fs['EEG'])
        b_low, a_low = butter(N=4, Wn=highcut, btype='low', fs=self.fs['EEG'])
        b_high, a_high = butter(N=4, Wn=lowcut, btype='high', fs=self.fs['EEG'])

        #  arrays for filtered EEG signals
        EEG_cg = np.zeros((len(self.channel_names['EEG']['cg']), data.shape[1]))
        channels_cg = {}
        EEG_ch = np.zeros((len(self.channel_names['EEG']['ch']), data.shape[1]))
        channels_ch = {}

        # filter the caregiver EEG data
        chan_counter_cg = 0
        chan_counter_ch = 0
        for idx, ch in enumerate(self.channels['EEG']):
            signal = data[idx, :].copy()
            signal = filtfilt(b_notch, a_notch, signal, axis=0)
            signal = filtfilt(b_low, a_low, signal, axis=0)
            signal = filtfilt(b_high, a_high, signal, axis=0)
            if ch in self.channel_names['EEG']['cg']:
                EEG_cg[chan_counter_cg, :] = signal
                channels_cg[ch] = chan_counter_cg
                chan_counter_cg += 1
            if ch in self.channel_names['EEG']['ch']:
                EEG_ch[chan_counter_ch, :] = signal
                channels_ch[ch] = chan_counter_ch
                chan_counter_ch += 1

        # decimate the data to reduce the sampling frequency q times
        signal_cg = decimate(EEG_cg, q, axis=-1)
        signal_ch = decimate(EEG_ch, q, axis=-1)

        # set the filtered and decimated EEG data
        self.channels['EEG'] = {'cg': channels_cg, 'ch': channels_ch}
        self.data['EEG'] = {'cg': signal_cg, 'ch': signal_ch}
        self.fs['EEG'] = self.fs['EEG'] // q  # new sampling frequency for the EEG data after decimation
        # time vector for the EEG data after decimation
        self.time['EEG'] = np.arange(0, signal_cg.shape[1] / self.fs['EEG'], 1 / self.fs['EEG'])
        self.modalities.append('EEG')

    def _extract_ECG_data(self, data, channels):
        t_ECG = np.arange(0, data.shape[1] / self.fs['ECG'],
                          1 / self.fs['ECG'])  # time vector for the ECG data in seconds

        # extract and filter the ECG data
        ECG_ch = data[channels['EKG1'], :] - data[channels['EKG2'], :]
        ECG_cg = data[channels['EKG1_cg'], :] - data[channels['EKG2_cg'], :]

        # design filters:
        b_notch, a_notch = iirnotch(50, 30, fs=self.fs['ECG'])
        sos_ecg = butter(5, 0.5, btype='high', output="sos", fs=self.fs['ECG'])
        ECG_ch_filtered = sosfiltfilt(sos_ecg, ECG_ch)
        ECG_ch_filtered = filtfilt(b_notch, a_notch, ECG_ch_filtered)
        ECG_cg_filtered = sosfiltfilt(sos_ecg, ECG_cg)
        ECG_cg_filtered = filtfilt(b_notch, a_notch, ECG_cg_filtered)
        self.data['ECG'] = {'ch': ECG_ch_filtered, 'cg': ECG_cg_filtered}
        self.time['ECG'] = t_ECG
        self.modalities.append('ECG')

    def _compute_IBI(self, Fs_IBI=4):
        # interpolate IBI signals from ECG data
        self.fs['IBI'] = Fs_IBI
        IBI_ch_interp, t_IBI_ch = self._interpolate_IBI_signals(self.data['ECG']['ch'], self.fs['ECG'])
        IBI_cg_interp, t_IBI_cg = self._interpolate_IBI_signals(self.data['ECG']['cg'], self.fs['ECG'])

        self.modalities.append('IBI')
        # truncate the IBI signals are of the same length
        min_length = min(len(IBI_ch_interp), len(IBI_cg_interp))
        self.time['IBI'] = t_IBI_ch[:min_length]
        # use the time vector for the child IBI as it is the same length as the caregiver IBI
        self.data['IBI'] = {'ch': IBI_ch_interp[:min_length], 'cg': IBI_cg_interp[:min_length]}

    def _interpolate_IBI_signals(self, ECG, label=''):
        # Extract R-peaks location
        _, info_ECG = nk.ecg_process(ECG, sampling_rate=self.fs['ECG'], method='neurokit')
        rpeaks = info_ECG["ECG_R_Peaks"]
        IBI = np.diff(rpeaks) / self.fs['ECG'] * 1000  # IBI in ms
        t = np.cumsum(IBI) / 1000  # time vector for the IBI signals [s]
        # DO SPRAWDZENIA
        t_ECG = np.arange(0, t[-1], 1 / self.fs['IBI'])  # time vector for the interpolated IBI signals
        cs = CubicSpline(t, IBI)
        IBI_interp = cs(t_ECG)
        if self.plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(t_ECG, IBI_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return IBI_interp, t_ECG

    # method for load Warsaw_Data_Frame.csv
    def _load_csv_data(self, csv_file):
        pass

    def _scan_for_events(self, threshold=20000):
        '''Scan for events in the diode signal and plot them if required.
        Args:
            threshold
        Returns:
            events (dict): Dictionary containing the start and end time of detected events measured in seconds from the start of the recording. The expected events are:
                - Movie_1
                - Movie_2
                - Movie_3
                - Talk_1
                - Talk_2'''
        events = {'Talk_1': None, 'Talk_2': None, 'Movie_1': None, 'Movie_2': None, 'Movie_3': None}

        Fs_EEG = self.fs['EEG']
        x = np.zeros(self.diode.shape)
        d = self.diode.copy()
        d /= threshold
        x[d > 1] = 1
        if self.plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(d, 'b', label='Diode Signal normalized by threshold')
            plt.plot(x, 'r', label='Diode Signal Thresholded')
            plt.title('Diode Signal with events')
            plt.xlabel('Samples')
            plt.ylabel('Signal Value')
            plt.legend()

        y = np.diff(x)
        up = np.zeros(y.shape, dtype=int)
        down = np.zeros(y.shape, dtype=int)
        up[y == 1] = 1
        down[y == -1] = 1
        if self.plot_flag:
            plt.plot(up, 'g', label='Up Events')
            plt.plot(down, 'm', label='Down Events')
            plt.legend()

        dt = 17  # ms between frames
        i = 0
        while i < len(down):
            if down[i] == 1:
                s1 = int(np.sum(up[i + int(0.5 * Fs_EEG) - 2 * dt: i + int(0.5 * Fs_EEG) + 2 * dt]))
                s2 = int(np.sum(up[i + int(1.0 * Fs_EEG) - 3 * dt: i + int(1.0 * Fs_EEG) + 3 * dt]))
                s3 = int(np.sum(up[i + int(1.5 * Fs_EEG) - 4 * dt: i + int(1.5 * Fs_EEG) + 4 * dt]))
                s4 = int(np.sum(up[i + int(2.0 * Fs_EEG) - 5 * dt: i + int(2.0 * Fs_EEG) + 5 * dt]))
                s5 = int(np.sum(up[i + int(2.5 * Fs_EEG) - 6 * dt: i + int(2.5 * Fs_EEG) + 6 * dt]))
                # plt.plot(x, 'b'), plt.plot(i,x[i],'bo')
                if s1 == 1 and s2 == 0 and s3 == 0 and s4 == 0 and s5 == 0:
                    print(f"Movie 1 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_1'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'ro')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 0:
                    print(f"Movie 2 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_2'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'go')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 1 and s2 == 0 and s3 == 1 and s4 == 0 and s5 == 1:
                    print(f"Movie 3 starts at {i / Fs_EEG:.2f} seconds")
                    events['Movie_3'] = i / Fs_EEG
                    if self.plot_flag:
                        plt.plot(x, 'b'), plt.plot(i, x[i], 'yo')
                    i += int(2.5 * Fs_EEG)
                elif s1 == 0 and s2 == 1 and s3 == 0 and s4 == 0 and s5 == 0:
                    if events['Talk_1'] is None:
                        print(f"Talk 1 starts at {i / Fs_EEG:.2f} seconds")
                        events['Talk_1'] = i / Fs_EEG
                        if self.plot_flag:
                            plt.plot(x, 'b'), plt.plot(i, x[i], 'co')
                        i += int(2.5 * Fs_EEG)
                    else:
                        print(f"Talk 2 starts at {i / Fs_EEG:.2f} seconds")
                        events['Talk_2'] = i / Fs_EEG
                        if self.plot_flag:
                            plt.plot(x, 'b'), plt.plot(i, x[i], 'mo')
                            plt.show()
                        i = len(down)  # talk 2 is the last event so finish scaning for events
            i += 1
        return events

    @staticmethod
    def load_output_data(filename):
        try:
            results = joblib.load(filename)
            return results
        except FileNotFoundError:
            print(f"File not found {filename}")

    def save_to_file(self):
        joblib.dump(self, self.output_dir + f"/{self.id}.joblib")
