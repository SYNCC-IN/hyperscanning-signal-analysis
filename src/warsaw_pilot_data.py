"""
Warsaw pilot data analysis script for SYNCC-IN project.

This script provides functions for analyzing multimodal hyperscanning data:
- HRV (Heart Rate Variability) DTF analysis
- EEG DTF analysis  
- Combined EEG+HRV DTF analysis

Author: Warsaw Team
Date: 2025
"""

import sys
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, firwin, lfilter, hilbert, welch
from scipy.stats import zscore

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import dataloader
from src.mtmvar import mvar_plot, multivariate_spectra, dtf_multivariate, graph_plot
from src.utils import plot_eeg_channels_pl, overlay_eeg_channels_hyperscanning_pl


# ==================== Configuration Constants ====================

# Default analysis parameters
DEFAULT_DYAD_ID = "W030"
DEFAULT_DATA_PATH = "../data"
DEFAULT_EEG_LOWCUT = 1.0  # Hz
DEFAULT_EEG_HIGHCUT = 40.0  # Hz
DEFAULT_EEG_FILTER_TYPE = "fir"  # 'fir' or 'iir'
DEFAULT_DECIMATION_FACTOR = 8

# Eye-tracking parameters
DEFAULT_ET_INTERPOLATE_THRESHOLD = 0.3
DEFAULT_ET_MEDIAN_FILTER_SIZE = 64
DEFAULT_ET_LOW_PASS_ORDER = 351
DEFAULT_ET_POS_CUTOFF = 128
DEFAULT_ET_PUPIL_CUTOFF = 4
DEFAULT_ET_PUPIL_CONFIDENCE = 0.9

# Analysis parameters
THETA_BAND_LOW = 3.5  # Hz
THETA_BAND_HIGH = 5.5  # Hz
HRV_TARGET_SAMPLING_FREQ = 2  # Hz
DTF_FREQ_RANGE_HRV = np.arange(0.01, 1, 0.01)
DTF_FREQ_RANGE_EEG = np.arange(1, 30, 0.5)
DTF_FREQ_RANGE_EEG_HRV = np.arange(0.01, 1, 0.01)

# EEG channel selection for DTF
EEG_CHANNELS_FOR_DTF = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 
                         'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']

PLOT_FLAG = False


# ==================== Main Function ====================

def main(plot_debug: bool = False, 
         analyze_hrv_dtf: bool = False, 
         analyze_eeg_dtf: bool = False, 
         analyze_eeg_hrv_dtf: bool = False) -> int:
    """
    Example of analysis function for Warsaw pilot data.
    
    Args:
        plot_debug: Whether to plot debug EEG channels
        analyze_hrv_dtf: Whether to analyze HRV DTF
        analyze_eeg_dtf: Whether to analyze EEG DTF
        analyze_eeg_hrv_dtf: Whether to analyze combined EEG+HRV DTF
        
    Returns:
        Exit code (0 for success)
    """
    selected_events = ['Brave']
    
    # Load multimodal data
    mmd = dataloader.create_multimodal_data(
        data_base_path=DEFAULT_DATA_PATH, 
        dyad_id=DEFAULT_DYAD_ID, 
        load_eeg=True, 
        load_et=True, 
        lowcut=DEFAULT_EEG_LOWCUT, 
        highcut=DEFAULT_EEG_HIGHCUT, 
        eeg_filter_type=DEFAULT_EEG_FILTER_TYPE, 
        interpolate_et_during_blinks_threshold=DEFAULT_ET_INTERPOLATE_THRESHOLD,
        median_filter_size=DEFAULT_ET_MEDIAN_FILTER_SIZE,
        low_pass_et_order=DEFAULT_ET_LOW_PASS_ORDER,
        et_pos_cutoff=DEFAULT_ET_POS_CUTOFF,
        et_pupil_cutoff=DEFAULT_ET_PUPIL_CUTOFF,
        pupil_model_confidence=DEFAULT_ET_PUPIL_CONFIDENCE,
        decimate_factor=DEFAULT_DECIMATION_FACTOR,
        plot_flag=PLOT_FLAG
    )

    # Debug plotting
    if plot_debug:
        _plot_debug_eeg_channels(mmd, selected_events)

    # Run requested analyses
    if analyze_hrv_dtf:
        analyze_hrv_dtf_for_event(mmd, selected_events)

    if analyze_eeg_dtf:
        analyze_eeg_dtf_for_events(mmd, selected_events)

    if analyze_eeg_hrv_dtf:
        analyze_eeg_hrv_dtf_for_events(mmd, selected_events)

    return 0


# ==================== HRV DTF Analysis ====================

def analyze_hrv_dtf_for_event(mmd, selected_event: List[str]) -> None:
    """
    Analyze HRV DTF for a given event.
    
    Extracts IBI signals for child and caregiver, decimates them to 2 Hz,
    and computes DTF to analyze directional coupling.
    
    Args:
        mmd: MultimodalData object with loaded signals
        selected_event: List containing single event name
    """
    # Extract IBI data
    selected_data = mmd.data[mmd.data.events == selected_event[0]]
    ibi_ch = selected_data['IBI_ch'].values
    ibi_cg = selected_data['IBI_cg'].values

    # Normalize IBI signals
    ibi_ch = zscore(ibi_ch)
    ibi_cg = zscore(ibi_cg)
    
    # Prepare data array
    data = np.zeros((2, len(ibi_ch)))
    data[0, :] = ibi_ch
    data[1, :] = ibi_cg
    
    # Downsample IBI signals to target frequency
    nyq = mmd.fs / 2
    cutoff = 0.9 * HRV_TARGET_SAMPLING_FREQ / 2
    sos = butter(8, cutoff / nyq, btype='low', output='sos')
    filtered = sosfiltfilt(sos, data, axis=-1)
    decimated_data = filtered[:, ::64]
    
    # Visualization
    time = selected_data['time'].values
    time_decimated = time[::64]
    
    plt.figure()
    plt.plot(time, data[0, :], alpha=0.3, label='Child IBI (original)')
    plt.plot(time, data[1, :], alpha=0.3, label='Caregiver IBI (original)')
    plt.plot(time_decimated, decimated_data[0, :], label='Child IBI (decimated)')
    plt.plot(time_decimated, decimated_data[1, :], label='Caregiver IBI (decimated)')
    plt.title(f'IBI signals for event: {selected_event[0]}')
    plt.xlabel('Time (s)')
    plt.ylabel('Z-scored IBI')
    plt.legend()
    plt.show()

    # Compute DTF
    dtf = dtf_multivariate(decimated_data, DTF_FREQ_RANGE_HRV, 
                           HRV_TARGET_SAMPLING_FREQ, max_model_order=15, 
                           crit_type='AIC')
    spectra = multivariate_spectra(decimated_data, DTF_FREQ_RANGE_HRV, 
                                    HRV_TARGET_SAMPLING_FREQ, max_model_order=15, 
                                    crit_type='AIC')
    
    # Plot results
    mvar_plot(spectra, dtf, DTF_FREQ_RANGE_HRV, 'From ', 'To ', 
              ['Child', 'Caregiver'], f'DTF {selected_event[0]}', 'sqrt')
    plt.show()


# ==================== EEG DTF Analysis ====================

def analyze_eeg_dtf_for_events(mmd, selected_events: List[str]) -> None:
    """
    Analyze EEG DTF for selected events.
    
    Extracts EEG signals from selected channels for both child and caregiver,
    and computes DTF separately for each.
    
    Args:
        mmd: MultimodalData object with loaded signals
        selected_events: List of event names to analyze
    """
    # Extract EEG data
    time, eeg_ch = mmd.get_signals(mode='EEG', member='ch', 
                                    selected_channels=EEG_CHANNELS_FOR_DTF, 
                                    selected_events=selected_events, 
                                    selected_times=None)
    time, eeg_cg = mmd.get_signals(mode='EEG', member='cg', 
                                    selected_channels=EEG_CHANNELS_FOR_DTF, 
                                    selected_events=selected_events, 
                                    selected_times=None)
 
    # Visualize with Plotly
    overlay_eeg_channels_hyperscanning_pl(
        eeg_ch, eeg_cg, 
        event=selected_events[0], 
        selected_channels_ch=EEG_CHANNELS_FOR_DTF, 
        selected_channels_cg=EEG_CHANNELS_FOR_DTF,
        title='Filtered EEG Channels - Hyperscanning (Plotly)'
    )

    # DTF analysis with optimal model order
    p_opt = 9  # Optimal model order for EEG DTF

    # Child DTF
    dtf_ch = dtf_multivariate(eeg_ch, DTF_FREQ_RANGE_EEG, mmd.fs, 
                               optimal_model_order=p_opt, comment='child')
    spectra_ch = multivariate_spectra(eeg_ch, DTF_FREQ_RANGE_EEG, mmd.fs, 
                                       optimal_model_order=p_opt)
    mvar_plot(spectra_ch, dtf_ch, DTF_FREQ_RANGE_EEG, 'From ', 'To ', 
              EEG_CHANNELS_FOR_DTF, 'Child DTF', 'sqrt')

    # Caregiver DTF
    dtf_cg = dtf_multivariate(eeg_cg, DTF_FREQ_RANGE_EEG, mmd.fs, 
                               optimal_model_order=p_opt, comment='caregiver')
    spectra_cg = multivariate_spectra(eeg_cg, DTF_FREQ_RANGE_EEG, mmd.fs, 
                                       optimal_model_order=p_opt)
    mvar_plot(spectra_cg, dtf_cg, DTF_FREQ_RANGE_EEG, 'From ', 'To ', 
              EEG_CHANNELS_FOR_DTF, 'Caregiver DTF', 'sqrt')
    
    plt.show()


# ==================== Combined EEG+HRV DTF Analysis ====================

def analyze_eeg_hrv_dtf_for_events(mmd, selected_events: List[str]) -> None:
    """
    Analyze combined EEG+HRV DTF for selected events.
    
    Extracts theta band activity from Fz channel and combines with HRV
    to analyze directional coupling between brain and heart signals.
    
    Args:
        mmd: MultimodalData object with loaded signals
        selected_events: List of event names to analyze
    """
    selected_channels = ['Fz']

    # Extract Fz channel data
    eeg_ch_Fz = mmd.get_signals(mode='EEG', member='ch', 
                                  selected_channels=selected_channels, 
                                  selected_events=selected_events, 
                                  selected_times=None)[1]
    eeg_cg_Fz = mmd.get_signals(mode='EEG', member='cg', 
                                  selected_channels=selected_channels, 
                                  selected_events=selected_events, 
                                  selected_times=None)[1]
    
    # Plot spectra
    _plot_spectra(eeg_ch_Fz, eeg_cg_Fz, mmd.fs, selected_events[0])

    # Filter in theta band and extract envelope
    b = firwin(numtaps=255, cutoff=[THETA_BAND_LOW, THETA_BAND_HIGH], 
               fs=mmd.fs, pass_zero=False)
    
    filtered_eeg_ch_Fz = lfilter(b, [1.0], eeg_ch_Fz[0, :])
    filtered_eeg_cg_Fz = lfilter(b, [1.0], eeg_cg_Fz[0, :])
    
    # Correct for filter delay
    delay = (len(b) - 1) // 2
    filtered_eeg_ch_Fz = np.roll(filtered_eeg_ch_Fz, -delay)
    filtered_eeg_cg_Fz = np.roll(filtered_eeg_cg_Fz, -delay)
    
    # Extract instantaneous amplitude
    eeg_ch_Fz_theta_amp = np.abs(hilbert(filtered_eeg_ch_Fz))
    eeg_cg_Fz_theta_amp = np.abs(hilbert(filtered_eeg_cg_Fz))

    # Extract IBI data
    ibi_ch = mmd.get_signals(mode='IBI', member='ch', 
                              selected_channels=['IBI_ch'], 
                              selected_events=selected_events, 
                              selected_times=None)[1]
    ibi_cg = mmd.get_signals(mode='IBI', member='cg', 
                                     selected_channels=['IBI_cg'], 
                                     selected_events=selected_events, 
                                     selected_times=None)[1]

    # Combine into DTF data array
    dtf_data = np.zeros((4, eeg_ch_Fz_theta_amp.shape[0]))
    dtf_data[0, :] = ibi_ch
    dtf_data[1, :] = ibi_cg
    dtf_data[2, :] = eeg_ch_Fz_theta_amp
    dtf_data[3, :] = eeg_cg_Fz_theta_amp
    
    # Z-score normalization
    dtf_data = zscore(dtf_data, axis=1)
    fs_dtf = 2  # Hz

    # Decimate to target frequency
    decimation_factor = int(mmd.fs // fs_dtf)
    sos = butter(8, 0.8 * (fs_dtf / 2) / (mmd.fs / 2), 
                 btype='low', output='sos')
    dtf_data = sosfiltfilt(sos, dtf_data, axis=1)
    dtf_data = dtf_data[:, ::decimation_factor]

    # Compute DTF
    dtf = dtf_multivariate(dtf_data, DTF_FREQ_RANGE_EEG_HRV, fs_dtf, 
                           max_model_order=15, crit_type='AIC')
    spectra = multivariate_spectra(dtf_data, DTF_FREQ_RANGE_EEG_HRV, 
                                    fs_dtf, max_model_order=15, crit_type='AIC')
    
    # Plot results
    mvar_plot(spectra, dtf, DTF_FREQ_RANGE_EEG_HRV, 'From ', 'To ',
              ['Child IBI', 'Caregiver IBI', 'Child Fz theta', 'Caregiver Fz theta'],
              'EEG+HRV DTF', 'sqrt')
    plt.show()
    # Finally let's plot the DTF results in the graph form using graph_plot  from mtmvar
    _, ax = plt.subplots(figsize=(10, 8))
    graph_plot(connectivity_matrix=dtf, ax=ax, freqs=DTF_FREQ_RANGE_EEG_HRV, freq_range=[0.2, 0.6], chan_names=['Child IBI', 'Caregiver IBI', 'Child Fz theta', 'Caregiver Fz theta'],
                title='DTF ' + selected_events[0] + ' (0.2-0.6 Hz)')
    plt.show()

# ==================== Helper Functions ====================

def _plot_debug_eeg_channels(mmd, selected_events: List[str]) -> None:
    """Plot EEG channels for both child and caregiver."""
    plot_eeg_channels_pl(
        mmd, 
        selected_events=selected_events, 
        selected_channels=[col for col in mmd.data.columns if col.startswith('EEG_ch_')],
        title='Filtered Child EEG Channels (offset for clarity)'
    )
    plot_eeg_channels_pl(
        mmd, 
        selected_events=selected_events, 
        selected_channels=[col for col in mmd.data.columns if col.startswith('EEG_cg_')],
        title='Filtered Caregiver EEG Channels (offset for clarity)'
    )


def _plot_spectra(eeg_ch: np.ndarray, eeg_cg: np.ndarray, 
                     fs: float, event_name: str) -> None:
    """Plot power spectra of Fz channels."""
    f_ch, pxx_ch = welch(eeg_ch[0, :], fs=fs, nperseg=1024)
    f_cg, pxx_cg = welch(eeg_cg[0, :], fs=fs, nperseg=1024)
    
    plt.figure(figsize=(12, 6))
    plt.plot(f_ch, pxx_ch, label='Child Fz channel')
    plt.plot(f_cg, pxx_cg, label='Caregiver Fz_cg channel')
    plt.title(f'Power Spectrum of {event_name} for Child and Caregiver Fz channels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    plt.xlim(0, 30)
    plt.legend()
    plt.grid()
    plt.show()



# ==================== Entry Point ====================

if __name__ == "__main__":
    sys.exit(main(
        plot_debug=False, 
        analyze_hrv_dtf=False, 
        analyze_eeg_dtf=False, 
        analyze_eeg_hrv_dtf=True
    ))
