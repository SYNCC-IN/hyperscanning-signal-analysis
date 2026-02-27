import numpy as np
import pandas as pd
import warnings
from enum import StrEnum

# Suppress pandas ChainedAssignmentError from neurokit2 (external library using deprecated patterns)
warnings.filterwarnings('ignore', category=pd.errors.ChainedAssignmentError)

import neurokit2 as nk
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from datetime import date
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from scipy.signal import decimate

class WhoEnum(StrEnum):
    CG_Only = "CG_Only",
    CH_Only = "CH_Only",
    Both = "Both",
    Neither = "Neither"

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
    secore: Optional[WhoEnum] = None
    movies: Optional[WhoEnum] = None
    conversation: Optional[WhoEnum] = None


@dataclass
class DualEEG:
    """Stores information about dual EEG tasks."""
    movies: Optional[WhoEnum] = None
    conversation: Optional[WhoEnum] = None

@dataclass
class DualFnirs:
    """Stores information about dual fnirs tasks."""
    movies: Optional[WhoEnum] = None
    conversation: Optional[WhoEnum] = None

@dataclass
class DualET:
    """Stores information about dual eye-tracking tasks."""
    movies: Optional[WhoEnum] = None
    conversation: Optional[WhoEnum] = None


@dataclass
class Tasks:
    """Container for all task-related information."""
    dual_hrv: DualHRV = field(default_factory=DualHRV)
    dual_eeg: DualEEG = field(default_factory=DualEEG)
    dual_et: DualET = field(default_factory=DualET)
    dual_fnirs: DualFnirs = field(default_factory=DualFnirs)


@dataclass
class ChildInfo:
    """Stores information about the child participant."""
    birth_date: Optional[date] = None
    age_years: Optional[str] = None
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
        self.events: Dict[str, Any] = {}  # dictionary of event dictionaries definig the events by name, time and duration. E.g., {'name': 'Incredibles', 'start': 12.5, 'duration': 300.0}
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
            mode (str): Signal modality to retrieve. Options: 'EEG', 'ECG', 'IBI', 'ET', 'diode'. Default: 'EEG'.
            member (str): Participant role. Options: 'ch' (child), 'cg' (caregiver). Default: 'ch'.
            selected_channels (list of str, optional): List of channel names to retrieve. 
                Required for 'EEG' and 'ET' modes. Ignored for other modes.
            selected_events (list of str, optional): List of event names to filter data by. 
                Only returns samples where events column matches one of these event names.
            selected_times (tuple of float, optional): Time range (start_time, end_time) in seconds 
                to filter data. Only returns samples within this time window.
            Note: If both selected_events and selected_times are provided, data is filtered by events first, then by times.

        Returns:
            tuple or None: Returns None if no matching columns are found. Otherwise returns:
                - time_vector (np.ndarray): 1D array of time points in seconds
                - channel_names (list of str): List of column names corresponding to retrieved channels
                - data (np.ndarray): 2D array of shape [n_channels, n_samples] containing signal data

        """
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
        channel_names = df_filtered.columns.tolist()
        # Return data as 2D array [ n_samples x n_channels ] 
        return time_vector, channel_names, df_filtered[cols].values  
    
    def get_events_as_marker_channel(self,  selected_times: Optional[tuple] = None):
        """
        Retrieve events as a marker channel aligned with the time column.
        Args:
            selected_times (tuple, optional): Time window (start_time, end_time) to filter events.

        Returns:
            tuple: (time_vector, marker_channel, event_to_marker) where
                - time_vector (np.ndarray): 1D array of time points in seconds
                - marker_channel (np.ndarray): 1D array of event markers aligned with time_vector
                - event_to_marker (dict): Mapping of event names to integer markers
        """
        if 'events' not in self.data.columns:
            return None
        time_vector = self.data['time'].values
        # create a dictionary mapping event names to integer markers
        unique_events = self.data['events'].dropna().unique()
        event_to_marker = {event: idx + 1 for idx, event in enumerate(unique_events)}
        # copy the events column to avoid modifying the original data
        events_copy = self.data['events'].copy()
        # create marker channel based on events column
        events_copy = events_copy.map(event_to_marker).fillna(0).astype(int)    
        marker_channel = events_copy.values
        if selected_times:
            start_time, end_time = selected_times
            mask = (time_vector >= start_time) & (time_vector <= end_time)
            time_vector = time_vector[mask]
            marker_channel = marker_channel[mask]
        return time_vector, marker_channel, event_to_marker
    
    def _set_eeg_data(self, eeg_data: np.ndarray, channel_mapping: Dict[str, int]):
        """
        Store EEG data in DataFrame with each channel as a separate column.
        
        Private method - should only be called by DataLoader.

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

    def _set_ecg_data(self, ecg_ch: np.ndarray, ecg_cg: np.ndarray):
        """Store ECG data in DataFrame. 
        Private method - should only be called by DataLoader."""
        # Initialize DataFrame with time columns if empty
        n_samples = ecg_ch.shape[0]
        if len(self.data) == 0:
            self.data = pd.DataFrame(index=range(n_samples))
            if self.fs is not None:
                self.data['time'] = np.arange(n_samples) / self.fs
                self.data['time_idx'] = np.arange(n_samples)
        self._ensure_data_length(len(ecg_ch))
        self.data['ECG_ch'] = ecg_ch
        self.data['ECG_cg'] = ecg_cg


    def _set_diode(self, diode: np.ndarray):
        """Store diode data in DataFrame. 
        Private method - should only be called by DataLoader.
        """
        self._ensure_data_length(len(diode))
        self.data['diode'] = diode

    def _set_EEG_events_column(self, events: List[dict]):
        """Populate EEG_events column based on event timing and duration. 
        Private method - should only be called by DataLoader."""
        if 'EEG_events' not in self.data.columns:
            self.data['EEG_events'] = None

        for ev in events:
            if 'start' in ev and 'duration' in ev:
                mask = (self.data['time'] >= ev['start']) & (self.data['time'] <= ev['start'] + ev['duration'])
                self.data.loc[mask, 'EEG_events'] = ev['name']


    def _ensure_data_length(self, length: int):
        """Ensure DataFrame has enough rows to hold data of given length.
        Private method - should only be called by DataLoader."""
        if len(self.data) == 0:
            self.data = pd.DataFrame(index=range(length))
        elif len(self.data) < length:
            # Extend DataFrame with NaN rows
            extra_rows = pd.DataFrame(index=range(len(self.data), length))
            self.data = pd.concat([self.data, extra_rows], ignore_index=True)

    def eeg_channel_names_all(self) -> List[str]:
        """Returns a combined list of child and caregiver EEG channel names."""
        return self.eeg_channel_names_ch + self.eeg_channel_names_cg
    
    def get_eeg_data_ch(self) -> np.ndarray | None:
        """
        Returns child EEG data as 2D array.
        
        Returns:
            np.ndarray or None: 2D array [n_channels x n_samples] or None if no data
        """
        prefix = 'EEG_ch_'
        cols = [col for col in self.data.columns if col.startswith(prefix)]
        
        if not cols:
            return None
        
        return self.data[cols].values.T
    
    def get_eeg_data_cg(self) -> np.ndarray | None:
        """
        Returns caregiver EEG data as 2D array.
        
        Returns:
            np.ndarray or None: 2D array [n_channels x n_samples] or None if no data
        """
        prefix = 'EEG_cg_'
        cols = [col for col in self.data.columns if col.startswith(prefix)]
        
        if not cols:
            return None
        
        return self.data[cols].values.T
    
    @staticmethod
    def get_eeg_data(df: pd.DataFrame, who: str) -> tuple[np.ndarray | None, list]:
        """
        Returns EEG data and channel names for specified participant from a DataFrame.
        
        This is a utility method for extracting EEG data from any DataFrame
        following the MultimodalData column naming convention.
        
        Args:
            df: DataFrame instance containing EEG data 
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
    
    def to_mne_raw(self, who: str, times: tuple[float, float] | None = None, 
                   event: str | None = None, margin_around_event: float = 0) -> tuple:
        """
        Export EEG data to MNE Raw object.
        
        Args:
            who: 'ch' for child, 'cg' for caregiver 
            times: Optional range of time to include (in seconds) (start_time, end_time)
            event: Optional event to include by name (e.g., 'Brave', 'Peppa', 'Incredibles', 'Talk_1', 'Talk_2')
            margin_around_event: Margin in seconds to add before and after the event time range  
            Note: if both event and times are provided, event takes precedence.
        
        Returns:
            Tuple of (raw, time) where:
            - raw: MNE Raw object with EEG data
            - time: np.ndarray of time points in seconds
            
        Note:
            Requires mne package to be installed.
        """
        import mne
        
        # Determine time range and select data
        annotations = None
        if event is not None:
            # Handle single event (string) 
            if isinstance(event, str):
                if event not in self.events:
                    raise ValueError(f"Event '{event}' not found in self.events")
                start_time = self.events[event]["start"] - margin_around_event
                end_time = self.events[event]["start"] + self.events[event]["duration"] + margin_around_event
                
                selected_data = self.data[
                    (self.data["time"] >= start_time)
                    & (self.data["time"] <= end_time)    
                ]
                annotations = [(event, margin_around_event, self.events[event]["duration"])]
                time = selected_data["time"].to_numpy()
            else:
                raise TypeError(f"event must be a string, got {type(event)}")
        elif times is not None:
            start_time, end_time = times
            selected_data = self.data[
                (self.data["time"] >= start_time)
                & (self.data["time"] <= end_time)
            ]
            # Set the annotations for events within the time range
            annotations = []
            for ev_name, ev in self.events.items():
                ev_start = ev["start"]
                ev_end = ev["start"] + ev["duration"]
                if (ev_start >= start_time) and (ev_end <= end_time):
                    annotations.append((ev_name, ev_start - start_time, ev["duration"]))
            time = selected_data["time"].to_numpy()
        else:
            selected_data = self.data   
            time = selected_data["time"].to_numpy()
        
        # Extract EEG data
        eeg_data, channel_names = self.get_eeg_data(df=selected_data, who=who)
        
        if eeg_data is None:
            raise ValueError(f"No EEG data found for {who}, event: {event}, times: {times}")
        
        # Create MNE Raw object
        info = mne.create_info(
            ch_names=channel_names, 
            sfreq=self.fs, 
            ch_types='eeg'
        )
        
        raw = mne.io.RawArray(eeg_data, info)
        
        # Add annotations if available
        if annotations is not None:
            onsets = [ann[1] for ann in annotations]
            durations = [ann[2] for ann in annotations]
            descriptions = [ann[0] for ann in annotations]
            raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
        
        # Mark the data as pre-filtered
        if self.eeg_filtration.applied:
            with raw.info._unlock():
                if self.eeg_filtration.high_pass:
                    raw.info['highpass'] = float(self.eeg_filtration.high_pass)
                if self.eeg_filtration.low_pass:
                    raw.info['lowpass'] = float(self.eeg_filtration.low_pass)
        
        # Add filter description
        if self.eeg_filtration.applied:
            filter_desc = (
                f"EEG filtered with {self.eeg_filtration.type} filters: "
                f"highpass={self.eeg_filtration.high_pass}Hz, "
                f"lowpass={self.eeg_filtration.low_pass}Hz, "
                f"notch={self.eeg_filtration.notch_freq}Hz (Q={self.eeg_filtration.notch_Q}). "
                f"Reference: {self.references}"
            )
            raw.info['description'] = filter_desc
        
        # Set montage for electrode positions
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False)
        
        return raw, time

    def _interpolate_ibi_signals(self, who, label='', plot_flag=False):
        """Extract R-peaks and interpolate IBI signals from ECG data. 
        Private method - called by _set_ibi()."""
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

    def _set_ibi(self):
        """Interpolate IBI signals from ECG data and store in DataFrame. 
        Private method - called by DataLoader."""
        # ecg_ch = self.data['ECG_ch'].values
        # ecg_cg = self.data['ECG_cg'].values

        self._interpolate_ibi_signals('ch')
        self._interpolate_ibi_signals('cg')

        if 'IBI' not in self.modalities:
            self.modalities.append('IBI')


    def _decimate_signals(self, q=8):
        """
        Decimate all signals by factor q while maintaining synchronization across modalities.
        Private method - should only be called by DataLoader.
        
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
        # check if the events column exists
        if 'events' in self.data.columns:   
            multimodal_data_dec.data['events'] = self.data['events'].values[::q]
        if 'ET_event' in self.data.columns:
            multimodal_data_dec.data['ET_event'] = self.data['ET_event'].values[::q]
        if 'EEG_events' in self.data.columns:
            multimodal_data_dec.data['EEG_events'] = self.data['EEG_events'].values[::q]
        if 'diode' in self.data.columns:
            multimodal_data_dec.data['diode'] = self.data['diode'].values[::q]

        # create a list of column that need to be antialiased filtered and decimated, these are all EEG, ECG, and IBI and ET columns
        columns_to_decimate = [ col for col in self.data.columns 
                               if col.startswith('EEG_ch_') 
                                 or col.startswith('EEG_cg_')
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

    def _create_events_column(self, start_error= 0.0):
        """Create events column based on EEG_events and ET_event columns. 
        Private method - called by DataLoader.
        - check if the columnes are present in the dataframe
        - create a new events column
        - populate the events column based on EEG_events and ET_event columns
        -check consistency of the coresponding events in EEG_events and ET_event columns
        Args:
            start_error: allowable error in start time between EEG and ET events (in seconds)
        Returns:
            None
          
        """

        if ('EEG_events' not in self.data.columns) and ('ET_event' not in self.data.columns):
            print('No event data: no EEG_events and ET_event columns found in the data.')
            return
        self.data['events'] = None
        EEG_events_dicts = []
        ET_events_dicts = []
        # if EEG_evnets present, populate structure of EEG_events with time and duration nad name
        if 'EEG_events' in self.data.columns:
            eeg_events_list = self.data['EEG_events'].dropna().unique()
            for ev_name in eeg_events_list:
                mask = self.data['EEG_events'] == ev_name
                start_time = self.data.loc[mask, 'time'].min()
                end_time = self.data.loc[mask, 'time'].max()
                duration = end_time - start_time
                event_dict = {'name': ev_name, 'start': start_time, 'duration': duration}
                EEG_events_dicts.append(event_dict)
        if 'ET_event' in self.data.columns:
            et_events_list = self.data['ET_event'].dropna().unique()
            for ev_name in et_events_list:
                mask = self.data['ET_event'] == ev_name
                start_time = self.data.loc[mask, 'time'].min()
                end_time = self.data.loc[mask, 'time'].max()
                duration = end_time - start_time
                event_dict = {'name': ev_name, 'start': start_time, 'duration': duration}
                ET_events_dicts.append(event_dict)
        # check consistency of the coresponding events in EEG_events_dicts and ET_events_dicts 
        for eeg_ev in EEG_events_dicts:
            for et_ev in ET_events_dicts:
                if eeg_ev['name'] == et_ev['name']:
                    diff = abs(eeg_ev['start'] - et_ev['start'])  
                    if diff > start_error:
                        print(f'\033[91mEvent {eeg_ev["name"]} differ in start times by: abs({diff}) seconds.\033[0m')
                    else:
                        print(f'\033[92mEvent {eeg_ev["name"]} start times are consistent within {start_error} seconds.\033[0m')

        # combine the two lists of event dicts, assuming that the times in EEG_events are more precise than in ET_events, but events at each lists may be unique
        # if the EEG_evnts contains an event we strat from it, otherwise we add the ET_events
        if not EEG_events_dicts:
            combined_events = ET_events_dicts
            print('No EEG_events found, using ET_event data only.')

        elif not ET_events_dicts:
            combined_events = EEG_events_dicts
            print('No ET_event found, using EEG_events data only.')
        else:
            combined_events = EEG_events_dicts.copy()
            for et_ev in ET_events_dicts:
                if not any(eeg_ev['name'] == et_ev['name'] for eeg_ev in EEG_events_dicts):
                    combined_events.append(et_ev)
        # populate the events column based on combined_events
        for ev in combined_events:
            if 'start' in ev and 'duration' in ev:
                mask = (self.data['time'] >= ev['start']) & (self.data['time'] <= ev['start'] + ev['duration'])
                self.data.loc[mask, 'events'] = ev['name']  
        print('Events column created based on EEG_events and ET_event columns.')

    def print_events(self):
        """Print event structure in a formatted table."""
        if not self.events:
            print('No events found.')
            return
        
        print('\n' + '='*70)
        print(f'{"Event Name":<30} {"Start (s)":<15} {"Duration (s)":<15}')
        print('='*70)
        for ev in self.events.values():
            name = ev.get('name', 'N/A')
            start = ev.get('start', 'N/A')
            duration = ev.get('duration', 'N/A')
            
            if isinstance(start, (int, float)):
                start_str = f'{start:.2f}'
            else:
                start_str = str(start)
                
            if isinstance(duration, (int, float)):
                duration_str = f'{duration:.2f}'
            else:
                duration_str = str(duration)
                
            print(f'{name:<30} {start_str:<15} {duration_str:<15}')
        print('='*70 + '\n')

    def _create_event_structure(self):   
        """Create event structure based on event column. 
        Private method - called by DataLoader.
        Args:
            None
        Returns:
            None
        """

        
        if 'events' not in self.data.columns:
            print('No events column found in the data.')
            return
        
        # Get unique event names
        unique_events = self.data['events'].dropna().unique()
        
        # Create list of event dictionaries
        events_list = []
        for ev_name in unique_events:
            mask = self.data['events'] == ev_name
            start_time = self.data.loc[mask, 'time'].min()
            end_time = self.data.loc[mask, 'time'].max()
            duration = end_time - start_time
            event_dict = {'name': ev_name, 'start': start_time, 'duration': duration}
            events_list.append(event_dict)
        
        self.events = {ev_dict['name']: ev_dict for ev_dict in events_list}
        print('Event structure created based on events column.')
        self.print_events()