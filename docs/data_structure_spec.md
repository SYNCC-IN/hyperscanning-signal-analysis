# Data Structure Specification

### EEGLAB-style multimodal structure for EEG, ET, and IBI signals

**Version:** 2.1  
**Last updated:** 2026-01-01  
**Author:** Joanna Duda-Goławska/Jarosław Żygierewicz

This document defines the unified **Python data structure** for handling multimodal child-caregiver data recorded with:

- EEG (electroencephalography),
- ET (eye-tracking),
- IBI (inter-beat interval, derived from ECG).

The structure is designed for transparent integration across different signal types, maintaining synchronisation and
consistent metadata.

In order to popoulate the data-structure, the expected directory-structure with raw data is:

```
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
```

## Core Architecture

The data are stored in a `MultimodalData` class that uses a **DataFrame-based approach** where all signal data is combined in a single pandas DataFrame called `data`. This approach ensures:
- Unified sampling frequency across all signals
- Shared time indexing
- Easy data alignment and merging
- Consistent column naming conventions

## MultimodalData Class Structure

### Core Attributes

```python
class MultimodalData:
    # Core data storage
    data: pd.DataFrame                      # All signal data in single DataFrame
    fs: float                               # Unified sampling frequency (Hz) - typically EEG sampling rate
    
    # Metadata
    id: str                                 # Dyad ID

    
    # EEG-specific metadata
    eeg_channel_names: list[str]            # List of all channel names in order
    eeg_channel_mapping: dict[str, int]     # Mapping: channel name → index in raw data
    references: str                         # Information about reference electrodes (e.g., "linked ears montage: (M1+M2)/2")
    eeg_filtration: Filtration              # Filtration parameters (see Filtration dataclass below)
    eeg_channel_names_ch: list[str]         # Child EEG channels after montage
    eeg_channel_names_cg: list[str]         # Caregiver EEG channels after montage
    
    # Events and epochs
    events: list[dict]                      # List of event markers with 'name', 'start', 'duration'
    epoch: list or None                     # Epoch information (if applicable)
    
    # Organizational structures
    paths: Paths                            # File and directory paths (see Paths dataclass)
    tasks: Tasks                            # Task information (see Tasks dataclass)
    modalities: list[str]                   # List of available modalities (e.g., ['EEG', 'ECG', 'IBI', 'ET'])
    child_info: ChildInfo                   # Child participant information (see ChildInfo dataclass)
    
    # Notes
    notes: str or None                      # Experimental notes
```

### DataFrame Column Naming Conventions

The `data` DataFrame uses consistent naming patterns for different signal types:

- **Time columns**: `time` (seconds), `time_idx` (sample indices)
- **EEG channels**: 
  - Child: `EEG_ch_{channel}` (e.g., `EEG_ch_Fp1`, `EEG_ch_Cz`)
  - Caregiver: `EEG_cg_{channel}` (e.g., `EEG_cg_Fp1`, `EEG_cg_Cz`)
- **ECG signals**: `ECG_ch` (child), `ECG_cg` (caregiver)
- **IBI signals**: `IBI_ch` (child), `IBI_cg` (caregiver)
- **Eye-tracking**:
  - Position: `ET_ch_x`, `ET_ch_y`, `ET_cg_x`, `ET_cg_y`
  - Pupil: `ET_ch_pupil`, `ET_cg_pupil`
  - Blinks: `ET_ch_blinks`, `ET_cg_blinks`
  - Events: `ET_event`
- **Diode signal**: `diode`
- **Events**: `events`

### Supporting Dataclasses

#### Filtration

Stores EEG signal filtration parameters:

```python
@dataclass
class Filtration:
    notch_Q: float or None              # Quality factor for notch filter
    notch_freq: float or None           # Notch filter frequency (Hz, typically 50 or 60)
    low_pass: float or None             # Low-pass filter cutoff frequency (Hz)
    high_pass: float or None            # High-pass filter cutoff frequency (Hz)
    type: str or None                   # Filter type ('fir' or 'iir')
    applied: bool                       # Whether filters have been applied (default: False)
```

#### Paths

Stores file and directory paths:

```python
@dataclass
class Paths:
    eeg_directory: str or None          # Path to EEG raw data
    et_directory: str or None           # Path to eye-tracking files
    hrv_directory: str or None          # Path to HRV/IBI files
    output_dir: str or None             # Path for saving results/figures
```

#### Tasks

Container for task-related information:

```python
@dataclass
class Tasks:
    dual_hrv: DualHRV                   # HRV task information
    dual_eeg: DualEEG                   # EEG task information
    dual_et: DualET                     # ET task information

@dataclass
class DualHRV:
    secore: bool or None                # Active HRV during SECORE
    movies: bool or None                # Passive HRV during movies
    conversation: bool or None          # Active HRV during conversation

@dataclass
class DualEEG:
    movies: bool or None                # Passive EEG during movies
    conversation: bool or None          # Active EEG during conversation

@dataclass
class DualET:
    movies: bool or None                # Passive ET during movies
    conversation: bool or None          # Active ET during conversation
```

#### ChildInfo

Stores child participant information:

```python
@dataclass
class ChildInfo:
    birth_date: datetime.date or None   # Child's birth date
    age_years: int or None              # Age in years at recording
    age_months: int or None             # Age in months at recording
    age_days: int or None               # Additional days beyond months
    rec_date: datetime.date or None     # Recording date
    group: str or None                  # Group: 'T' (Typical), 'ASD', 'P' (Premature)
    sex: str or None                    # Sex: 'M' (male), 'F' (female)
```

## Key Methods

### Data Population Methods

```python
set_eeg_data(eeg_data: np.ndarray, channel_mapping: Dict[str, int])
    # Store EEG data with each channel as a DataFrame column
    # Creates time and time_idx columns if fs is set
    # Args:
    #   eeg_data: 2D array [n_channels x n_samples]
    #   channel_mapping: Dict mapping channel names to indices in eeg_data
    
set_ecg_data(ecg_ch: np.ndarray, ecg_cg: np.ndarray)
    # Store ECG signals for child and caregiver
    # Creates columns 'ECG_ch' and 'ECG_cg'
    
set_ibi_data(ibi_ch: np.ndarray, ibi_cg: np.ndarray, ibi_times: np.ndarray)
    # Store interpolated IBI signals (note: ibi_times parameter not used in current implementation)
    # Creates columns 'IBI_ch' and 'IBI_cg'
    
set_diode(diode: np.ndarray)
    # Store diode signal for event detection
    # Creates column 'diode'
    
set_events_column(events: List[dict])
    # Populate events column based on event timing and duration
    # Events should have format: [{'name': str, 'start': float, 'duration': float}, ...]
```

### Data Retrieval Methods

```python
get_signals(mode: str = 'EEG', member: str = 'ch', selected_channels: List[str] = None,
            selected_events: List[str] = None, selected_times: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]
    # Retrieve signals from the DataFrame based on mode, member, selected channels, events, and times
    # Args:
    #   mode: Signal type - 'EEG', 'ECG', 'IBI', 'ET' (eye-tracking), or 'diode'
    #   member: 'ch' (child) or 'cg' (caregiver)
    #   selected_channels: List of channel names to retrieve (None = all channels for that mode/member)
    #   selected_events: List of event names to filter by (None = no event filtering)
    #   selected_times: Time range tuple (start, end) in seconds (None = all times)
    # Returns:
    #   Tuple of (time_vector, signal_data)
    #   - time_vector: 1D array of timestamps in seconds
    #   - signal_data: 2D array [n_channels x n_samples] or 1D array for single channel modes

get_eeg_data_ch() -> np.ndarray or None
    # Returns child EEG data as 2D array [n_channels x n_samples]
    
get_eeg_data_cg() -> np.ndarray or None
    # Returns caregiver EEG data as 2D array [n_channels x n_samples]
    
eeg_channel_names_all() -> list[str]
    # Returns combined list of child and caregiver channel names
```

### Utility Methods

```python
align_time_to_first_event()
    # Shifts time columns so first event starts at t=0
    # Finds earliest event start time and subtracts from all time values
    
decimate_signals(q: int = 8) -> MultimodalData
    # Decimates all signal columns by factor q while maintaining synchronization
    # Args:
    #   q: Decimation factor (default=8) - reduce sampling rate by this factor
    # Returns:
    #   MultimodalData: New instance with decimated signals and updated sampling frequency
    # Process:
    #   - Applies anti-aliasing FIR filter before decimation to prevent frequency folding
    #   - Handles NaN values by forward filling before decimation to maintain signal continuity
    #   - Decimates time columns, signal columns (EEG, ECG, IBI, ET), and event markers
    #   - Updates fs (sampling frequency) by dividing by q
    # Note: Creates a new MultimodalData object; does not modify in place
    # Args:
    #   q: Decimation factor (default: 8)
    # Returns:
    #   New MultimodalData instance with decimated signals and fs/q sampling rate
    # Note: Time, events, and diode columns are downsampled by selecting every q-th sample
    #       EEG, ECG, IBI, and ET columns are properly anti-aliased filtered then decimated
    
interpolate_ibi_signals(who: str, label: str = '', plot_flag: bool = False)
    # Extracts R-peaks and interpolates IBI signals from ECG data using neurokit2
    # Args:
    #   who: 'ch' (child) or 'cg' (caregiver)
    #   label: Optional label for plotting
    #   plot_flag: Whether to plot the results

set_ibi()
    # Convenience method to interpolate IBI signals for both child and caregiver
    # Calls interpolate_ibi_signals for both 'ch' and 'cg'

_ensure_data_length(length: int)
    # Internal method to ensure DataFrame has enough rows
    # Creates or extends DataFrame as needed
```

## Example Usage

```python
from src.data_structures import MultimodalData
import numpy as np

# Create instance
md = MultimodalData()
md.id = "W001"
md.fs = 1024  # Hz

# Set EEG data
eeg_data = np.random.randn(21, 10000)  # 21 channels, 10000 samples
channel_mapping = {'Fp1': 0, 'Fp2': 1, ...}
md.eeg_channel_names_ch = ['Fp1', 'Fp2', 'F3', ...]
md.set_eeg_data(eeg_data, channel_mapping)

# Set ECG data
ecg_ch = np.random.randn(10000)
ecg_cg = np.random.randn(10000)
md.set_ecg_data(ecg_ch, ecg_cg)

# Add events
md.events = [
    {'name': 'Brave', 'start': 10.0, 'duration': 120.0},
    {'name': 'Talk_1', 'start': 200.0, 'duration': 180.0}
]
md.set_events_column(md.events)

# Access data
print(md.data.columns)  # All column names
print(md.data['EEG_ch_Fp1'])  # Specific channel data
ch_eeg = md.get_eeg_data_ch()  # All child EEG as array
```

## Version History

- **2.1 (2026-01-01)**: Enhanced signal retrieval and decimation methods
  - Added comprehensive `get_signals()` method for flexible signal retrieval with event/time filtering
  - Enhanced `decimate_signals()` documentation to clarify return behavior and NaN handling
  - Improved method documentation with detailed parameter and return value descriptions

- **2.0 (2025-12-30)**: Updated to reflect DataFrame-based architecture
  - Changed from separate arrays to unified DataFrame storage
  - Unified sampling frequency (`fs` instead of separate `eeg_fs`, `ecg_fs`, `ibi_fs`)
  - Updated filtration structure (`eeg_filtration` with `notch_Q` and `notch_freq`)
  - Added modalities tracking
  - Updated column naming conventions

- **1.0 (2025-10-19)**: Initial specification with separate data structures
