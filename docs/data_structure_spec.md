# Data Structure Specification

### EEGLAB-style multimodal structure for EEG, ET, and IBI signals

**Version:** 1.0  
**Last updated:** 2025-10-19  
**Author:** Joanna Duda-Goławska

This document defines the unified **Python data structure** for handling multimodal child-caregiver data recorded with:

- EEG (electroencephalography),
- ET (eye-tracking),
- IBI (inter-beat interval, derived from ECG).

The structure is designed for transparent integration across different signal types, maintaining synchronisation and
consistent metadata.

The data are stored as a single Python object, named `data`.

# Core structure

```python
data = {
    "id": str,                              # Dyad ID
    "eeg_data": np.ndarray,                 # EEG data [n_channels x n_samples]
    "eeg_fs": float,                        # EEG sampling rate (Hz)
    "eeg_times": np.ndarray,                # time vector (s) [1 x n_samples]
    "eeg_channel_names_all()": list[str],   # list of channel names in order
    "eeg_channel_mapping": dict[str, int],  # mapping: channel name → index in 'data'

    "references": str,                      # Information about reference electrodes or common average

    "filtration": {                         # Information about filtering
        "notch": bool,                      # If True, notch filter applied
        "low_pass": float,                  # Low-pass filter cutoff frequency (Hz)
        "high_pass": float,                 # High-pass filter cutoff frequency (Hz)
        "type": str                         # Type of filter (e.g., 'FIR', 'IIR')
    }

    'eeg_channel_names_ch': list[str],      # child EEG channels after montage
    'eeg_channel_names_cg': list[str],      # caregiver EEG channels after montage

    'ecg_ch': np.ndarray,                   # filtered ECG (child)
    'ecg_cg': np.ndarray,                   # filtered ECG (caregiver)
    'ecg_fs': int,                          # ECG sampling frequency
    'ecg_times': np.ndarray,                # time vector for ECG

    'ibi_ch_interp': np.ndarray,            # interpolated IBI (child)
    'ibi_cg_interp': np.ndarray,            # interpolated IBI (caregiver)
    'ibi_fs': int,                          # IBI sampling frequency (default: 4 Hz)
    'ibi_times': np.ndarray                 # time vector for interpolated IBI

    'eyetracker_ch': np.ndarray,            # ET (child)
    'eyetracker_cg': np.ndarray,            # ET (caregiver)
    'eyetracker_Fs': int,                   # ET sampling frequency 
    'eyetracker_times': np.ndarray          # time vector for interpolated IBI

    "events": list,                         # list of event markers (stimuli, triggers, etc.) 
    "epoch": list or None,                  # 

    "paths": {
        "eeg_directory": str,               # path to EEG data raw
        "et_directory": str,                # path to eye-tracking files
        "hrv_directory": str,               # path to HRV -> IBI files
        "output_dir": str,                  # path where to save results/figures
    },

   "tasks": {
        "dual_HRV": {
            "SECORE": bool,                 # True if active HRV during SECORE was recorded
            "movies": bool,                 # True if passive HRV recorded
            "conversation": bool            # True if active HRV recorded
        },
        "dual_EEG": {
            "movies": bool,                 # True if passive EEG recorded
            "conversation": bool,           # True if active EEG recorded
        }
        "dual_ET": {
            "movies": bool,                 # True if passive ET recorded
            "conversation": bool            # True if active ET recorded
        }
    }

   "child_info": {                          # Information about child
        "birth_date": datetime.date,        # Child birth date
        "age_years": int,                   # Child age in months at the time of recording
        "age_months": int,                  # Child age in months at the time of recording
        "age_days": int,                    # Additional days beyond months
        "rec_date: datetime.date,           # Date when recording was done
        "group": str,                       # Child group: 'T' (Typical),  'ASD' (Autism Spectrum Disorder), 'P' (Premature)
        "sex": str                          # Child sex: 'M' (male), 'F' (female)
    }

    "notes": str or None,                   # notes from experiment
}
