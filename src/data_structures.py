import numpy as np
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

