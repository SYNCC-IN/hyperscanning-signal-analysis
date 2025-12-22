import numpy as np
import neurokit2 as nk
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from datetime import date
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import dataloader
from scipy.signal import decimate
import pandas as pd
import os


@dataclass
class Filtration:
    """Stores information about signal filtration."""
    notch: Optional[bool] = None
    low_pass: Optional[float] = None
    high_pass: Optional[float] = None
    type: Optional[str] = None


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
    """
    def __init__(self):
        self.diode = None  # normalized diode signal
        # Core EEG data
        self.id: Optional[str] = None  # Dyad ID
        self.eeg_data: Optional[np.ndarray] = None  # EEG data [n_channels x n_samples]
        self.eeg_decimated_data_ch: Optional[np.ndarray] = None
        self.eeg_decimated_data_cg: Optional[np.ndarray] = None
        self.eeg_fs: Optional[float] = None  # EEG sampling rate (Hz)
        self.eeg_times: Optional[np.ndarray] = None  # time vector (s) [1 x n_samples]
        self.eeg_channel_names: List[str] = []  # list of channel names in order
        self.eeg_channel_mapping: Dict[str, int] = {}  # mapping: channel name → index in 'data'

        # EEG metadata
        self.references: Optional[str] = None  # Information about reference electrodes or common average
        self.filtration: Filtration = Filtration()
        self.eeg_channel_names_ch: List[str] = []  # child EEG channels after montage
        self.eeg_channel_names_cg: List[str] = []  # caregiver EEG channels after montage

        # ECG and IBI data
        self.ecg_data_ch: Optional[np.ndarray] = None  # filtered ECG (child)
        self.ecg_data_cg: Optional[np.ndarray] = None  # filtered ECG (caregiver)
        self.ecg_fs: Optional[int] = None  # ECG sampling frequency
        self.ecg_times: Optional[np.ndarray] = None  # time vector for ECG
        self.ibi_ch_interp: Optional[np.ndarray] = None  # interpolated IBI (child)
        self.ibi_cg_interp: Optional[np.ndarray] = None  # interpolated IBI (caregiver)
        self.ibi_fs: Optional[int] = None  # IBI sampling frequency (default: 4 Hz)
        self.ibi_times: Optional[np.ndarray] = None  # time vector for interpolated IBI

        # Eye-tracking data
        self.eyetracker_ch: Optional[np.ndarray] = None  # ET (child)
        self.eyetracker_cg: Optional[np.ndarray] = None  # ET (caregiver)
        self.eyetracker_fs: Optional[int] = None  # ET sampling frequency
        self.eyetracker_times: Optional[np.ndarray] = None  # time vector for interpolated IBI

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



    def eeg_channel_names_all(self) -> List[str]:
        """Returns a combined list of child and caregiver EEG channel names."""
        return self.eeg_channel_names_ch + self.eeg_channel_names_cg

    def eeg_data_ch(self) -> Optional[np.ndarray]:
        """Returns EEG data for child channels only."""
        if self.eeg_data is None:
            return None
        ch_indices = [self.eeg_channel_mapping[name] for name in self.eeg_channel_names_ch]
        return self.eeg_data[ch_indices, :]

    def eeg_data_cg(self) -> Optional[np.ndarray]:
        """Returns EEG data for caregiver channels only."""
        if self.eeg_data is None:
            return None
        cg_indices = [self.eeg_channel_mapping[name] for name in self.eeg_channel_names_cg]
        return self.eeg_data[cg_indices, :]

    def interpolate_ibi_signals(self, ecg, label='', plot_flag=False):
        # Extract R-peaks location
        _, info_ecg = nk.ecg_process(ecg, sampling_rate=self.ecg_fs, method='neurokit')
        r_peaks = info_ecg["ECG_R_Peaks"]
        ibi = np.diff(r_peaks) / self.ecg_fs * 1000  # IBI in ms
        times = np.cumsum(ibi) / 1000  # time vector for the IBI signals [s]
        ecg_times = np.arange(0, times[-1], 1 / self.ibi_fs)  # time vector for the interpolated IBI signals
        cubic_spline = CubicSpline(times, ibi)
        ibi_interp = cubic_spline(ecg_times)
        if plot_flag:
            plt.figure(figsize=(12, 6))
            plt.plot(ecg_times, ibi_interp)
            plt.xlabel('time [s]')
            plt.ylabel('IBI [ms]')
            plt.title(f'Interpolated IBI signal of {label} as a function of time')
            plt.show()
        return ibi_interp, ecg_times

    def compute_ibi(self, ibi_fs=4):
        # interpolate IBI signals from ECG data
        self.ibi_fs = ibi_fs
        self.ibi_ch_interp, t_ibi_ch = self.interpolate_ibi_signals(self.ecg_data_ch)
        self.ibi_cg_interp, _ = self.interpolate_ibi_signals(self.ecg_data_cg)

        if 'IBI' not in self.modalities:
            self.modalities.append('IBI')

        # truncate the IBI signals are of the same length
        min_length = min(len(self.ibi_ch_interp), len(self.ibi_cg_interp))
        self.ibi_times = t_ibi_ch[:min_length]
        self.ibi_ch_interp = self.ibi_ch_interp[min_length:]
        self.ibi_cg_interp = self.ibi_cg_interp[min_length:]

    def decimate_signals(self, eeg_cg, eeg_ch, q=8):
        """
        Task 3: Decimates the filtered 'cg' and 'ch' signals.
        Returns the decimated signals.
        """
        self.eeg_decimated_data_cg = decimate(eeg_cg, q, axis=-1)
        self.eeg_decimated_data_ch = decimate(eeg_ch, q, axis=-1)

class MultiModalDataPd:
    '''
    This is a class for storing data in pandas DataFrame format.
    It mirrors the MultimodalData class but uses DataFrames for easier data manipulation.
    '''
    def __init__(self, id: str):
        self.id = id  # Dyad ID
        self.data = None  # Placeholder for DataFrame storage
        self.fs = None  # Placeholder for sampling frequency
        self.info = {'modalities': []  }  # Placeholder for metadata storage
        self.history = None  # Placeholder for processing history storage

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
                dataloader.process_event_et(loaded_data['annotations_000'], et_df)
        
        if task_flags.get('movies_ch'):
            if 'ch_pos_000' in loaded_data:
                dataloader.process_pos(loaded_data['ch_pos_000'], et_df, 'ch')
            if 'ch_pupil_000' in loaded_data:
                dataloader.process_pupil(loaded_data['ch_pupil_000'], et_df, 'ch')
            if 'ch_blinks_000' in loaded_data:
                dataloader.process_blinks(loaded_data['ch_blinks_000'], et_df, 'ch')
        
        if task_flags.get('movies_cg'):
            if 'cg_pos_000' in loaded_data:
                dataloader.process_pos(loaded_data['cg_pos_000'], et_df, 'cg')
            if 'cg_pupil_000' in loaded_data:
                dataloader.process_pupil(loaded_data['cg_pupil_000'], et_df, 'cg')
            if 'cg_blinks_000' in loaded_data:
                dataloader.process_blinks(loaded_data['cg_blinks_000'], et_df, 'cg')
        
        # Process talk1 task (001)
        if task_flags.get('talk1_ch') or task_flags.get('talk1_cg'):
            if 'annotations_001' in loaded_data:
                dataloader.process_event_et(loaded_data['annotations_001'], et_df, 'talk1')
        
        if task_flags.get('talk1_ch'):
            if 'ch_pupil_001' in loaded_data:
                dataloader.process_pupil(loaded_data['ch_pupil_001'], et_df, 'ch')
        
        if task_flags.get('talk1_cg'):
            if 'cg_pupil_001' in loaded_data:
                dataloader.process_pupil(loaded_data['cg_pupil_001'], et_df, 'cg')
        
        # Process talk2 task (002)
        if task_flags.get('talk2_ch') or task_flags.get('talk2_cg'):
            if 'annotations_002' in loaded_data:
                dataloader.process_event_et(loaded_data['annotations_002'], et_df, 'talk2')
        
        if task_flags.get('talk2_ch'):
            if 'ch_pupil_002' in loaded_data:
                dataloader.process_pupil(loaded_data['ch_pupil_002'], et_df, 'ch')
        
        if task_flags.get('talk2_cg'):
            if 'cg_pupil_002' in loaded_data:
                dataloader.process_pupil(loaded_data['cg_pupil_002'], et_df, 'cg')
    def add_data(self, eeg_path: str = None, et_path: str = None, ibi_path: str = None, plot_flag = False ):
        '''
        Docstring for add_data
        This method loads data from specified paths and adds them to the DataFrame.
        It supports EEG, ET, and IBI data modalities.
        1. Loads EEG data if eeg_path is provided.
        2. Loads ET data if et_path is provided.
        3. Merges the loaded data into a single DataFrame.
        4. Updates metadata about the modalities present in the data.
        5. Aligns time vectors across modalities based on the first event time.
        6. Optionally plots the loaded data if plot_flag is set to True.
        7. Handles missing files gracefully by printing warnings.
        8. Ensures that the sampling frequency (fs) is set based on the EEG data if not already specified.
        9. Uses helper functions from the dataloader module for data processing.
        10. Returns None.

        TODO: handle cases where some files are missing.
        - we need to make sure that if some files are missing, 
            we still load the available data and print warnings for the missing files.
        - if a file form a cretain task of a certaiin member of the dyad is missing, 
            we skip loading data for that task only for that member; we fill the missing data with NaNs.
        11. Currently, IBI loading is not implemented.

        '''
        if eeg_path:
            EEG_CHANS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                         'M1',  'T3', 'C3', 'Cz', 'C4', 'T4', 'M2',
                        'T5',  'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
            eeg_ecg_data = dataloader.load_eeg_data(self.id, eeg_path, plot_flag)
            eeg_ecg_df = pd.DataFrame()
            if self.fs is None:
                self.fs = eeg_ecg_data.eeg_fs
                print('Based on EEG, setting fs to:', self.fs)
            # construct time column in seconds
            time = np.arange(0, eeg_ecg_data.eeg_data.shape[1],1 )
            eeg_ecg_df['time'] = time/self.fs # convert to  seconds
            eeg_ecg_df['time_idx'] = time # this column keeps integer indexes for merging with other modalities
            # construct the events column
            eeg_ecg_df['events'] = None
            # Populate events column based on event timing and duration
            for ev in eeg_ecg_data.events:
                if 'start' in ev:
                    mask = (eeg_ecg_df['time'] >= ev['start']) & (eeg_ecg_df['time'] <= ev['start'] + ev['duration'])
                    eeg_ecg_df.loc[mask, 'events'] = ev['name']

            # find the time of the first event
            min_start_time = float('inf')
            for ev in eeg_ecg_data.events:
                if 'start' in ev:
                    min_start_time = min(min_start_time, ev['start'])
            if min_start_time == float('inf'):
                min_start_time = 0
            # reset the time to the first event
            eeg_ecg_df['time'] = eeg_ecg_df['time'] - min_start_time
            eeg_ecg_df['time_idx'] = eeg_ecg_df['time_idx'] - min_start_time*self.fs

            # copy EEG data of child and caregiver to columns in the dataframe
            for chan in eeg_ecg_data.eeg_channel_mapping:
                chan_parts = chan.split('_')
                if chan_parts[0] in EEG_CHANS:
                    if len(chan_parts)==2:
                        col_name = f'EEG_cg_{chan_parts[0]}'
                    else:
                        col_name = f'EEG_ch_{chan_parts[0]}'
                    eeg_ecg_df[col_name] = eeg_ecg_data.eeg_data[eeg_ecg_data.eeg_channel_mapping[chan]]
            self.info['modalities'].append('EEG')
            # copy ECG data of child and caregiver to columns in the dataframe
            eeg_ecg_df['ECG_cg'] = eeg_ecg_data.ecg_data_cg
            eeg_ecg_df['ECG_ch'] = eeg_ecg_data.ecg_data_ch
            self.info['modalities'].append('ECG')
            # copy Diode data to dataframe
            eeg_ecg_df['DIODE'] = eeg_ecg_data.diode
            self.info['modalities'].append('DIODE')
            if self.data is None:
                self.data = eeg_ecg_df.copy()
            else:
                self.data = pd.merge(self.data, eeg_ecg_df, how = 'outer', on = 'time_idx')
                self.data['time'] = self.data['time_idx'] / self.fs
                self.data = self.data.drop(columns=['time_x','time_y'])


        if et_path:
            # Load eye-tracking data from CSV files
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
                    file_paths = self._build_et_file_paths(et_path, task['id'], member)
                    
                    # Check if files exist
                    all_exist, missing = self._check_et_files_exist(file_paths)
                    
                    # Set flag for this task/member combination
                    flag_key = f"{task['name']}_{'ch' if member == 'child' else 'cg'}"
                    task_flags[flag_key] = all_exist
                    
                    if not all_exist:
                        print(f"Warning: Missing ET files for {self.id} {member} {task['name']}: {missing}")
                    else:
                        # Load data if all files exist
                        task_data = self._load_et_task_data(file_paths, task['id'], member, min_max_times)
                        loaded_data.update(task_data)
            
            # Skip ET processing if no data was loaded
            if not min_max_times:
                print(f"Warning: No ET data available for {self.id}")
                return
            
            # Construct dataframe for ET data
            et_df = pd.DataFrame()
            
            # Set sampling frequency if not already set
            if self.fs is None:
                self.fs = 1024  # default sampling rate for UW EEG data
                print(f'Setting fs to the default Fs of EEG: {self.fs}')
            
            # Find the overall min and max times across all tasks and members
            overall_min_time = min([t[0] for t in min_max_times])
            overall_max_time = max([t[1] for t in min_max_times])
            
            print(f"ET time range: {overall_min_time:.2f}s to {overall_max_time:.2f}s")
            
            # Create time vector
            et_df['time'] = np.arange(overall_min_time, overall_max_time, 1 / self.fs)
            et_df['time_idx'] = (et_df['time'] * self.fs).astype(int)
            
            # Process loaded data into the dataframe
            self._process_et_data_to_dataframe(et_df, loaded_data, task_flags, [t['id'] for t in tasks])
                
            # Align ET time to EEG time by subtracting the time of the first event
            min_start_time_et = et_df[et_df['ET_event'].notna()]['time'].min()
            et_df['time'] = et_df['time'] - min_start_time_et
            et_df['time_idx'] = et_df['time_idx'] - int(min_start_time_et*self.fs)

            #  merging ET data into the main dataframe
            if self.data is None:
                self.data = et_df.copy()
            else:
                self.data = pd.merge(self.data, et_df, how = 'outer', on = 'time_idx')
                self.data['time'] = self.data['time_idx'] / self.fs
                self.data = self.data.drop(columns=['time_x','time_y'])
                self.data = self.data.replace(np.nan, None)

            self.info['modalities'].append('ET')
            
            
        pass
    def save_to_file(self, folder_path: str):
        pass

    def load_from_file(folder_path: str):
        pass