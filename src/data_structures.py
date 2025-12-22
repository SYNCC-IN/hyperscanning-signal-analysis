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
            # Load eye-tracking data from CSV files: THIS PART TO BE UPDATED AFTER THE STRUCTURE OF DATA in UW IS CLARIFIED
            # For now, we will load data from hardcoded paths for testing purposes
            # movies task 000
            # Check if files exist before loading. If any file is missing, print a warning and skip loading ET data for that task.
            

            base_paths_movies_ch = {
                'annotations_0': et_path + 'child/000/exports/000/annotations.csv',
                'ch_pos_0':      et_path + 'child/000/exports/000/surfaces/gaze_positions_on_surface_Surface 1.csv',
                'ch_pupil_0':    et_path + 'child/000/exports/000/pupil_positions.csv',
                'ch_blinks_0':   et_path + 'child/000/exports/000/blinks.csv'
            }
            base_paths_movies_cg = {
                'annotations_0': et_path + 'caregiver/000/exports/000/annotations.csv',
                'cg_pos_0':      et_path + 'caregiver/000/exports/000/surfaces/gaze_positions_on_surface_Surface 1.csv',
                'cg_pupil_0':    et_path + 'caregiver/000/exports/000/pupil_positions.csv',
                'cg_blinks_0':   et_path + 'caregiver/000/exports/000/blinks.csv'
            }
            base_paths_talk1_ch = {
                'annotations_1': et_path + 'child/001/exports/000/annotations.csv',
                'ch_pupil_1': et_path + 'child/001/exports/000/pupil_positions.csv',
            }
            base_paths_talk1_cg = {
                'annotations_1': et_path + 'caregiver/001/exports/000/annotations.csv',
                'cg_pupil_1':    et_path + 'caregiver/001/exports/000/pupil_positions.csv'
            }
            base_paths_talk2_ch = {
                'annotations_2': et_path + 'child/002/exports/000/annotations.csv',
                'ch_pupil_2':    et_path + 'child/002/exports/000/pupil_positions.csv'
            }   
            base_paths_talk2_cg = {
                'annotations_2': et_path + 'caregiver/002/exports/000/annotations.csv',
                'cg_pupil_2':    et_path + 'caregiver/002/exports/000/pupil_positions.csv'
            } 
            
            # Check for missing files in movies task
            movies_flag_ch = True
            missing_files = [name for name, path in base_paths_movies_ch.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                movies_flag_ch = False
            movies_flag_cg = True
            missing_files = [name for name, path in base_paths_movies_cg.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                movies_flag_cg = False
            # Check for missing files in talk1 task
            talk1_flag_ch = True
            missing_files = [name for name, path in base_paths_talk1_ch.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                talk1_flag_ch = False
            talk1_flag_cg = True
            missing_files = [name for name, path in base_paths_talk1_cg.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                talk1_flag_cg = False
            # Check for missing files in talk2 task
            talk2_flag_ch = True
            missing_files = [name for name, path in base_paths_talk2_ch.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                talk2_flag_ch = False
            talk2_flag_cg = True
            missing_files = [name for name, path in base_paths_talk2_cg.items() if not os.path.exists(path)]
            if missing_files:
                print(f"Warning: Missing ET files for {self.id}: {missing_files}")
                talk2_flag_cg = False
            # Prepare list of min and max times for each task to align time vectors later
            min_max_times = []

            # Load the data from CSV files
            if movies_flag_ch:
                annotations_0 = pd.read_csv(base_paths_movies_ch['annotations_0'])
                ch_pos_df_0 =   pd.read_csv(base_paths_movies_ch['ch_pos_0'])
                min_max_times.append( (ch_pos_df_0['gaze_timestamp'].min(), ch_pos_df_0['gaze_timestamp'].max()) )
                ch_pupil_df_0 = pd.read_csv(base_paths_movies_ch['ch_pupil_0'])
                min_max_times.append( (ch_pupil_df_0['pupil_timestamp'].min(), ch_pupil_df_0['pupil_timestamp'].max()) )
                ch_blinks_0 =   pd.read_csv(base_paths_movies_ch['ch_blinks_0']) # we assume blinks are within the range of pupil timestamps
               

            if movies_flag_cg:
                annotations_0 = pd.read_csv(base_paths_movies_cg['annotations_0'])  # we assume annotations are the same for both members of the dyad
                cg_pos_df_0 =   pd.read_csv(base_paths_movies_cg['cg_pos_0'])
                min_max_times.append( (cg_pos_df_0['gaze_timestamp'].min(), cg_pos_df_0['gaze_timestamp'].max()) )
                cg_pupil_df_0 = pd.read_csv(base_paths_movies_cg['cg_pupil_0'])
                min_max_times.append( (cg_pupil_df_0['pupil_timestamp'].min(), cg_pupil_df_0['pupil_timestamp'].max()) )
                cg_blinks_0 =   pd.read_csv(base_paths_movies_cg['cg_blinks_0'])
            if talk1_flag_ch:
                annotations_1 = pd.read_csv(base_paths_talk1_ch['annotations_1'])
                ch_pupil_df_1 = pd.read_csv(base_paths_talk1_ch['ch_pupil_1'])
                min_max_times.append( (ch_pupil_df_1['pupil_timestamp'].min(), ch_pupil_df_1['pupil_timestamp'].max()) )
            if talk1_flag_cg:   
                annotations_1 = pd.read_csv(base_paths_talk1_cg['annotations_1'])
                cg_pupil_df_1 = pd.read_csv(base_paths_talk1_cg['cg_pupil_1'])  
                min_max_times.append( (cg_pupil_df_1['pupil_timestamp'].min(), cg_pupil_df_1['pupil_timestamp'].max()) )
            if talk2_flag_ch:
                annotations_2 = pd.read_csv(base_paths_talk2_ch['annotations_2'])
                ch_pupil_df_2 = pd.read_csv(base_paths_talk2_ch['ch_pupil_2'])
                min_max_times.append( (ch_pupil_df_2['pupil_timestamp'].min(), ch_pupil_df_2['pupil_timestamp'].max()) )
            if talk2_flag_cg:   
                annotations_2 = pd.read_csv(base_paths_talk2_cg['annotations_2'])
                cg_pupil_df_2 = pd.read_csv(base_paths_talk2_cg['cg_pupil_2'])      
                min_max_times.append( (cg_pupil_df_2['pupil_timestamp'].min(), cg_pupil_df_2['pupil_timestamp'].max()) )


            # construct dataframe for ET data
            et_df = pd.DataFrame()
            # prepare the time column
            if self.fs is None:
                self.fs = 1024  # default sampling rate for UW EEG data; we want to keep all time series at the same sampling rate
                print('setting fs to the default Fs of EEG:', self.fs)
            # find the overall min and max times across all tasks and members of the dyad
            overall_min_time = min([t[0] for t in min_max_times])
            overall_max_time = max([t[1] for t in min_max_times])

            print("min", overall_min_time)
            print("max", overall_max_time)
            et_df['time'] = np.arange(overall_min_time, overall_max_time, 1 / self.fs)
            et_df['time_idx'] = (et_df['time']*self.fs).astype(int)  # integer time indexes for merging with other modalities
            if movies_flag_ch or movies_flag_cg:
                dataloader.process_event_et(annotations_0, et_df)
            if movies_flag_ch: 
                dataloader.process_pos(ch_pos_df_0, et_df, 'ch')
                dataloader.process_pupil(ch_pupil_df_0, et_df,'ch')
                dataloader.process_blinks(ch_blinks_0, et_df,'ch')
            if movies_flag_cg:
                dataloader.process_pos(cg_pos_df_0, et_df, 'cg')
                dataloader.process_pupil(cg_pupil_df_0, et_df,'cg')
                dataloader.process_blinks(cg_blinks_0, et_df,'cg')
            if talk1_flag_ch or talk1_flag_cg:
                dataloader.process_event_et(annotations_1, et_df, 'talk1')
            if talk1_flag_ch:
                dataloader.process_pupil(ch_pupil_df_1, et_df,'ch')
            if talk1_flag_cg:
                dataloader.process_pupil(cg_pupil_df_1, et_df,'cg')
            if talk2_flag_ch or talk2_flag_cg:
                dataloader.process_event_et(annotations_2, et_df, 'talk2')
            if talk2_flag_ch:   
                dataloader.process_pupil(ch_pupil_df_2, et_df,'ch') 
            if talk2_flag_cg:
                dataloader.process_pupil(cg_pupil_df_2, et_df,'cg')
                
            # align ET time to EEG time by subtracting the time of the first event; find the time of the first event in ET data
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