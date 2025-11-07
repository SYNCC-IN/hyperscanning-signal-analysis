import numpy as np
from datetime import date
from typing import List, Dict, Optional
from dataclasses import dataclass, field


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
    directory_eeg: Optional[str] = None
    directory_et: Optional[str] = None
    directory_hrv: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class DualHRV:
    """Stores information about dual HRV tasks."""
    SECORE: Optional[bool] = None
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
        # Core EEG data
        self.ID: Optional[str] = None  # Dyad ID
        self.data_EEG: Optional[np.ndarray] = None  # EEG data [n_channels x n_samples]
        self.Fs_EEG: Optional[float] = None  # EEG sampling rate (Hz)
        self.times_EEG: Optional[np.ndarray] = None  # time vector (s) [1 x n_samples]
        self.chanNames_EEG: List[str] = []  # list of channel names in order
        self.channels_EEG: Dict[str, int] = {}  # mapping: channel name → index in 'data'

        # EEG metadata
        self.references: Optional[str] = None  # Information about reference electrodes or common average
        self.filtration: Filtration = Filtration()
        self.EEG_channels_ch: List[str] = []  # child EEG channels after montage
        self.EEG_channels_cg: List[str] = []  # caregiver EEG channels after montage

        # ECG and IBI data
        self.ECG_ch: Optional[np.ndarray] = None  # filtered ECG (child)
        self.ECG_cg: Optional[np.ndarray] = None  # filtered ECG (caregiver)
        self.Fs_ECG: Optional[int] = None  # ECG sampling frequency
        self.t_ECG: Optional[np.ndarray] = None  # time vector for ECG
        self.IBI_ch_interp: Optional[np.ndarray] = None  # interpolated IBI (child)
        self.IBI_cg_interp: Optional[np.ndarray] = None  # interpolated IBI (caregiver)
        self.Fs_IBI: Optional[int] = None  # IBI sampling frequency (default: 4 Hz)
        self.t_IBI: Optional[np.ndarray] = None  # time vector for interpolated IBI

        # Eye-tracking data
        self.ET_ch: Optional[np.ndarray] = None  # ET (child)
        self.ET_cg: Optional[np.ndarray] = None  # ET (caregiver)
        self.Fs_ET: Optional[int] = None  # ET sampling frequency
        self.t_ET: Optional[np.ndarray] = None  # time vector for interpolated IBI

        # Events and epochs
        self.event: List[any] = []  # list of event markers (stimuli, triggers, etc.)
        self.epoch: Optional[List[any]] = None

        # Paths
        self.paths: Paths = Paths()

        # Task information
        self.tasks: Tasks = Tasks()

        # Child information
        self.child_info: ChildInfo = ChildInfo()

        # Notes
        self.notes: Optional[str] = None  # notes from experiment
