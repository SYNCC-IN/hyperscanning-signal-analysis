import numpy as np
import pandas as pd
import neurokit2 as nk
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from datetime import date
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from scipy.signal import decimate


@dataclass
class Filtration:
    """Stores information about signal filtration."""
    notch_Q: Optional[float] = None
    notch_freq: Optional[float] = None
    notch_a: Optional[np.ndarray] = None
    notch_b: Optional[np.ndarray] = None
    low_pass: Optional[float] = None
    low_pass_a: Optional[np.ndarray] = None
    low_pass_b: Optional[np.ndarray] = None
    high_pass: Optional[float] = None
    high_pass_a: Optional[np.ndarray] = None
    high_pass_b: Optional[np.ndarray] = None
    type: Optional[str] = None
    applied: Optional[bool] = False


@dataclass
class Paths:
    """Stores relevant file and directory paths."""
    eeg_directory: Optional[str] = None
    et_directory: Optional[str] = None
    hrv_directory: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class DualHRV:
    """Stores information about dual HRV tasks."""
    secore: Optional[bool] = None
    movies: Optional[bool] = None
    conversation: Optional[bool] = None


@dataclass
class DualEEG:
    """Stores information about dual EEG tasks."""
    movies: Optional[bool] = None
    conversation: Optional[bool] = None


@dataclass
class DualET:
    """Stores information about dual eye-tracking tasks."""
    movies: Optional[bool] = None
    conversation: Optional[bool] = None


@dataclass
class Tasks:
    """Container for all task-related information."""
    dual_hrv: DualHRV = field(default_factory=DualHRV)
    dual_eeg: DualEEG = field(default_factory=DualEEG)
    dual_et: DualET = field(default_factory=DualET)


@dataclass
class ChildInfo:
    """Stores information about the child participant."""
    birth_date: Optional[date] = None
    age_years: Optional[int] = None
    age_months: Optional[int] = None
    age_days: Optional[int] = None
    rec_date: Optional[date] = None
    group: Optional[str] = None
    sex: Optional[str] = None


class MultimodalData:
    """
    Represents multimodal child-caregiver data, including EEG, ET, and IBI signals.

    This class is based on the EEGLAB-style multimodal data structure specification.
    All signal data is stored in a single pandas DataFrame called 'data',
    where each EEG channel, ECG signal, etc. becomes its own column.
    The sampling frequency 'fs' applies to all signals (i.e., the EEG sampling frequency).
    Time column is shared across all signals.
    time_idx column is shared across all signals and is used for merging data that may not span the entire recording duration.
    Events are stored in the 'events' column of the DataFrame.
    Filtration information for EEG signals is stored in the 'eeg_filtration' attribute.


    Column naming conventions:
    - EEG channels: 'EEG_ch_{channel}' for child, 'EEG_cg_{channel}' for caregiver
    - ECG: 'ECG_ch', 'ECG_cg'
    - IBI: 'IBI_ch', 'IBI_cg'
    - ET: 'ET_ch_{x, y, pupil}', 'ET_cg_{x, y, pupil}'
    - Diode: 'diode'
    - Time: 'time', 'time_idx'
    - Events: 'events'
    """
    def __init__(self):
        # Core data stored in DataFrame - all signal data combined here
        self.data: pd.DataFrame = pd.DataFrame()

        # Sampling frequency - in the multimodal DataFrame approach, all signals share the same sampling frequency, i.e., the EEG sampling frequency
        self.fs: Optional[float] = None  # EEG sampling rate (Hz)
        

        # Core metadata
        self.id: Optional[str] = None  # Dyad ID
        self.eeg_channel_names: List[str] = []  # list of channel names in order
        self.eeg_channel_mapping: Dict[str, int] = {}  # mapping: channel name → index

        # EEG metadata
        self.references: Optional[str] = None  # Information about reference electrodes or common average
        self.eeg_filtration: Filtration = Filtration()
        self.eeg_channel_names_ch: List[str] = []  # child EEG channels after montage
        self.eeg_channel_names_cg: List[str] = []  # caregiver EEG channels after montage

        # Events and epochs
        self.events: List[Any] = []  # list of event markers (stimuli, triggers, etc.)
        self.epoch: Optional[List[Any]] = None

        # Paths
        self.paths: Paths = Paths()

        # Task information
        self.tasks: Tasks = Tasks()

        # Modalities
        self.modalities: List[str] = []

        # Child information
        self.child_info: ChildInfo = ChildInfo()

        # Notes
        self.notes: Optional[str] = None  # notes from experiment

    # -------------------------------------------------------------------------
    # Methods for managing data in DataFrame
    # -------------------------------------------------------------------------
    # TODO: all the fuctions for setting data in the dataframe should require 
    # time vector for proper alignment with the time column. The functions for getting 
    # data can return data aligned to the time column, together with the time vector.
    def get_signals(self, mode='EEG', member='ch', selected_channels=None, selected_events=None, selected_times=None):
        """
        Retrieve signals from the DataFrame based on mode, member, selected channels, events, and times.

        Args:
            mode: 'EEG', 'ECG', 'IBI', 'ET', 'diode
            member: 'ch' for child, 'cg' for caregiver
            selected_channels: List of channel names to retrieve (for EEG and ET)
            selected_events: List of event names to filter data
            selected_times: Tuple (start_time, end_time) to filter data by time
        """
        # Implementation of get_signals method goes here
        prefix = ''
        if mode == 'EEG':
            prefix = f'EEG_{member}_'
        elif mode == 'ET':
            prefix = f'ET_{member}_'
        elif mode == 'ECG':
            prefix = f'ECG_{member}'
        elif mode == 'IBI':
            prefix = f'IBI_{member}'
        elif mode == 'diode':
            prefix = 'diode'    
        else:
            raise ValueError("Invalid mode. Choose from 'EEG', 'ECG', 'IBI', 'ET'.")
        # Filter columns based on selected channels
        if mode in ['EEG', 'ET']:
            cols = [f'{prefix}{ch}' for ch in selected_channels if f'{prefix}{ch}' in self.data.columns]
        else:
            cols = [prefix] if prefix in self.data.columns else []  
        if not cols:
            return None
        df_filtered = self.data[cols + ['time', 'events']].copy()   
        # Filter by selected events
        if selected_events:
            df_filtered = df_filtered[df_filtered['events'].isin(selected_events)]
        # Filter by selected times
        if selected_times:
            start_time, end_time = selected_times
            df_filtered = df_filtered[(df_filtered['time'] >= start_time) & (df_filtered['time'] <= end_time)]
        # Retrive the time vector
        time_vector = df_filtered['time'].values
        # Remove 'time' and 'events' columns before returning data
        df_filtered = df_filtered.drop(columns=['time', 'events'])
        # Return data as 2D array [n_channels x n_samples]
        return time_vector, df_filtered[cols].values.T  # return as 2D array [n_channels x n_samples]
    
    def set_eeg_data(self, eeg_data: np.ndarray, channel_mapping: Dict[str, int]):
        """
        Store EEG data in DataFrame with each channel as a separate column.

        Args:
            eeg_data: 2D array [n_channels x n_samples]
            channel_mapping: Dict mapping channel names to indices in eeg_data
        """
        # Initialize DataFrame with time columns if empty
        n_samples = eeg_data.shape[1]
        if len(self.data) == 0:
            self.data = pd.DataFrame(index=range(n_samples))
            if self.fs is not None:
                self.data['time'] = np.arange(n_samples) / self.fs
                self.data['time_idx'] = np.arange(n_samples)

        # Add each EEG channel as a column
        for chan_name, chan_idx in channel_mapping.items():
            if chan_name in self.eeg_channel_names_ch or chan_name in self.eeg_channel_names_cg:
                chan_parts = chan_name.split('_')
                if len(chan_parts) == 2 and chan_parts[1] == 'cg':
                    col_name = f'EEG_cg_{chan_parts[0]}'
                else:
                    col_name = f'EEG_ch_{chan_name}'
                self.data[col_name] = eeg_data[chan_idx, :]

                
    def get_eeg_data_ch(self) -> Optional[np.ndarray]:
        """Returns EEG data for child channels only as 2D array."""
        ch_cols = [col for col in self.data.columns if col.startswith('EEG_ch_')]
        if not ch_cols:
            return None
        return self.data[ch_cols].values.T

    def get_eeg_data_cg(self) -> Optional[np.ndarray]:
        """Returns EEG data for caregiver channels only as 2D array."""
        cg_cols = [col for col in self.data.columns if col.startswith('EEG_cg_')]
        if not cg_cols:
            return None
        return self.data[cg_cols].values.T


    def set_ecg_data(self, ecg_ch: np.ndarray, ecg_cg: np.ndarray):
        """Store ECG data in DataFrame."""
        self._ensure_data_length(len(ecg_ch))
        self.data['ECG_ch'] = ecg_ch
        self.data['ECG_cg'] = ecg_cg

    def set_ibi_data(self, ibi_ch: np.ndarray, ibi_cg: np.ndarray, ibi_times: np.ndarray):
        """Store interpolated IBI data in DataFrame."""
        self._ensure_data_length(len(ibi_ch))
        self.data['IBI_ch'] = ibi_ch
        self.data['IBI_cg'] = ibi_cg
        #self.data['IBI_times'] = ibi_times - not needed since time is shared

    def set_diode(self, diode: np.ndarray):
        """Store diode data in DataFrame."""
        self._ensure_data_length(len(diode))
        self.data['diode'] = diode

    def set_events_column(self, events: List[dict]):
        """Populate events column based on event timing and duration."""
        if 'events' not in self.data.columns:
            self.data['events'] = None

        for ev in events:
            if 'start' in ev and 'duration' in ev:
                mask = (self.data['time'] >= ev['start']) & (self.data['time'] <= ev['start'] + ev['duration'])
                self.data.loc[mask, 'events'] = ev['name']

    def align_time_to_first_event(self):
        """Align time columns so that first event starts at t=0."""
        if 'events' not in self.data.columns or 'time' not in self.data.columns:
            return

        min_start_time = self.data[self.data['events'].notna()]['time'].min()
        if pd.isna(min_start_time):
            min_start_time = 0

        self.data['time'] = self.data['time'] - min_start_time
        if 'time_idx' in self.data.columns:
            self.data['time_idx'] = self.data['time_idx'] - int(min_start_time * self.fs)

    def _ensure_data_length(self, length: int):
        """Ensure DataFrame has enough rows to hold data of given length."""
        if len(self.data) == 0:
            self.data = pd.DataFrame(index=range(length))
        elif len(self.data) < length:
            # Extend DataFrame with NaN rows
            extra_rows = pd.DataFrame(index=range(len(self.data), length))
            self.data = pd.concat([self.data, extra_rows], ignore_index=True)

    def eeg_channel_names_all(self) -> List[str]:
        """Returns a combined list of child and caregiver EEG channel names."""
        return self.eeg_channel_names_ch + self.eeg_channel_names_cg

    def interpolate_ibi_signals(self, who, label='', plot_flag=False):
        """Extract R-peaks and interpolate IBI signals from ECG data."""
        _, info_ecg = nk.ecg_process(self.data[f'ECG_{who}'].values, sampling_rate=self.fs, method='neurokit')
        r_peaks = info_ecg["ECG_R_Peaks"]
        ibi = np.diff(r_peaks) / self.fs * 1000  # IBI in ms
        times = np.cumsum(ibi) / 1000  # time vector for the IBI signals [s]
        # correct the times so that the first IBI corresponds to the time of the first R-peak
        # this should corrspond to the time vector of the ECG signal
        times = times + (r_peaks[0] / self.fs)
        ecg_times = np.arange(times[0], times[-1], 1 / self.fs)  # time vector for the interpolated IBI signals
        cubic_spline = CubicSpline(times, ibi)
        ibi_interp = cubic_spline(ecg_times)
        df_ibi = pd.DataFrame({'time': ecg_times, f'IBI_{who}': ibi_interp})
        self.data = pd.merge_asof(self.data.sort_values('time'), df_ibi.sort_values('time'), on='time', direction='nearest')
     
        if plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(ecg_times, ibi_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return ibi_interp, ecg_times

    def set_ibi(self):
        """Interpolate IBI signals from ECG data and store in DataFrame."""
        # ecg_ch = self.data['ECG_ch'].values
        # ecg_cg = self.data['ECG_cg'].values

        self.interpolate_ibi_signals('ch')
        self.interpolate_ibi_signals('cg')

        if 'IBI' not in self.modalities:
            self.modalities.append('IBI')


    def decimate_signals(self, q=8):
        """
        Decimate all signals by factor q while maintaining synchronization across modalities.
        Note:
            in the multimodal dataframe approach we keep all types of signals in the same dataframe, 
            with the same sampling frequency (i.e., the EEG sampling frequency). 
            However, decimating the EEG signals only may lead to inconsistencies if other signals (e.g., ECG, IBI) are not decimated accordingly. 
            Therefore, we need to ensure that all signals are decimated properly to maintain synchronization across modalities.
        Args:
            q: Decimation factor
        Returns:
            multimodal_data_dec: New MultimodalData instance with decimated signals       
        """
        # create a new multimodal data structure to hold decimated signals
        multimodal_data_dec = MultimodalData()
        multimodal_data_dec.fs = self.fs / q
        multimodal_data_dec.id = self.id
        multimodal_data_dec.eeg_channel_names_ch = self.eeg_channel_names_ch
        multimodal_data_dec.eeg_channel_names_cg = self.eeg_channel_names_cg
        multimodal_data_dec.eeg_channel_mapping = self.eeg_channel_mapping
        multimodal_data_dec.references = self.references
        multimodal_data_dec.eeg_filtration = self.eeg_filtration
        multimodal_data_dec.events = self.events
        multimodal_data_dec.paths = self.paths
        multimodal_data_dec.tasks = self.tasks
        multimodal_data_dec.modalities = self.modalities
        multimodal_data_dec.child_info = self.child_info
        multimodal_data_dec.notes = self.notes  

        # decimate time, events, and diode columns by selectiging every q-th sample
        multimodal_data_dec.data['time'] = self.data['time'].values[::q]
        multimodal_data_dec.data['time_idx'] = self.data['time_idx'].values[::q]
        multimodal_data_dec.data['events'] = self.data['events'].values[::q]
        multimodal_data_dec.data['ET_event'] = self.data['ET_event'].values[::q]
        multimodal_data_dec.data['diode'] = self.data['diode'].values[::q]

        # create a list of column that need to be antialiased filtered and decimated, these are all EEG, ECG, and IBI and ET columns
        columns_to_decimate = [ col for col in self.data.columns 
                               if col.startswith('EEG') 
                               or col.startswith('ECG') 
                               or col.startswith('IBI') 
                               or col.startswith('ET_ch_') 
                               or col.startswith('ET_cg_')]
        for col in columns_to_decimate:
            # check if the column contain NaN values, if so, fill them with the previous value (forward fill) before decimation
            if self.data[col].isnull().any():
                print(f'Column {col} contains NaN values, applying forward fill before decimation.')
                data_filled = self.data[col].infer_objects(copy=False).ffill().values
            else:
                data_filled = self.data[col].values
            decimated = decimate( (data_filled).astype(float), q, ftype='fir', zero_phase=True)
            multimodal_data_dec.data[col] = decimated

        return multimodal_data_dec

