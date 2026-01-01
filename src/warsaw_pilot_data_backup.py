import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.stats import zscore

# Add the parent directory to the path to import src as a package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import dataloader

from src.mtmvar import graph_plot, mvar_plot, multivariate_spectra, dtf_multivariate
from src.utils import plot_eeg_channels_pl, overlay_eeg_channels_hyperscanning_pl, clean_data_with_ica, \
    get_data_for_selected_channel_and_event, get_ibi_signal_from_ecg_for_selected_event

# Reload dataloader to reflect any changes made during development
import importlib
importlib.reload(dataloader)

import mne

plot_flag = False

# WARNING: this code is incompatible with refactored DataLoader class
# FIXME: update the code to use the new DataLoader structure

def main(plot_debug=False, analyze_hrv_dtf=False, analyze_eeg_dtf=False, analyze_eeg_hrv_dtf=False):
    selected_events = ['Brave']  # # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'


    # data = DataLoader("W_010", False)
    # data.load_eeg_data("../DATA/W_010/")
    # events = data.events
    dyad_id = "W030"
    lowcut=1.0
    highcut=40.0
    eeg_filter_type = 'iir' # choose 'fir' or 'iir' for EEG filtering
    q=8  # decimation factor
    mmd = dataloader.create_multimodal_data(data_base_path = "../data", 
                                                        dyad_id = dyad_id, 
                                                        load_eeg=True, 
                                                        load_et=True, 
                                                        lowcut=lowcut, 
                                                        highcut=highcut, 
                                                        eeg_filter_type=eeg_filter_type, 
                                                        interpolate_et_during_blinks_threshold=0.3,
                                                        median_filter_size=64,
                                                        low_pass_et_order=351,
                                                        et_pos_cutoff=128,
                                                        et_pupil_cutoff=4,
                                                        pupil_model_confidence=0.9,
                                                        decimate_factor=q,
                                                        plot_flag=plot_flag)

    if plot_debug:
        #debug_plot(data, events)
        plot_eeg_channels_pl(mmd, 
                             selected_events=selected_events, 
                             selected_channels=[col for col in mmd.data.columns if col.startswith('EEG_ch_')],
                             title='Filtered Child EEG Channels (offset for clarity)')
        plot_eeg_channels_pl(mmd, 
                             selected_events=selected_events, 
                             selected_channels=[col for col in mmd.data.columns if col.startswith('EEG_cg_')],
                             title='Filtered Caregiver EEG Channels (offset for clarity)')

    if analyze_hrv_dtf:
        hrv_dtf(mmd, selected_events)

    if analyze_eeg_dtf:
        eeg_dtf(mmd, selected_events)

    if analyze_eeg_hrv_dtf:
        eeg_hrv_dtf(mmd, selected_events)


def eeg_hrv_dtf_analyze_event(filtered_data, selected_channels_ch, selected_channels_cg, events, event):
    # design a bandpass filter for the theta band
    lowcut = 5.0  # Hz
    highcut = 7.5  # Hz
    sos_theta = butter(4, [lowcut, highcut], btype='band', fs=filtered_data['Fs_EEG'], output='sos')

    data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
    data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)

    # compute and plot spectra of the selected channels using Welch's method
    f_ch, pxx_ch = welch(data_ch[0, :], fs=filtered_data['Fs_EEG'], nperseg=1024)
    f_cg, pxx_cg = welch(data_cg[0, :], fs=filtered_data['Fs_EEG'], nperseg=1024)
    # plot the power spectral density of the child and caregiver Fz channels
    plt.figure(figsize=(12, 6))
    plt.plot(f_ch, pxx_ch, label='Child Fz channel')
    plt.plot(f_cg, pxx_cg, label='Caregiver Fz_cg channel')
    plt.title(f'Power Spectrum of {event} for Child and Caregiver Fz channels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    # Highlight the theta band
    plt.axvspan(lowcut, highcut, color='yellow', alpha=0.5, label='Theta Band (5-7.5 Hz)')
    plt.xlim(0, 30)  # Limit x-axis to 30 Hz
    plt.legend()
    plt.grid()
    plt.show()
    # Now filter the data in the theta band and compute the instantaneous amplitude using Hilbert transform

    # filter the data in the theta band
    data_ch_theta = sosfiltfilt(sos_theta, data_ch[0, :])
    data_cg_theta = sosfiltfilt(sos_theta, data_cg[0, :])
    # get the instantaneous amplitude (envelope) of the filtered signal using Hilbert transform
    data_ch_theta_amp = np.abs(hilbert(data_ch_theta))
    data_cg_theta_amp = np.abs(hilbert(data_cg_theta))

    # plot the envelope for the child and caregiver EEG channels, add the filtered signal as the background
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f'Child EEG channel Fz theta amplitude for {event}')
    ax[1].set_title(f'Caregiver EEG channel Fz_cg theta amplitude for {event}')
    ax[0].plot(data_ch_theta_amp, 'r', label='Fz theta amplitude')
    ax[1].plot(data_cg_theta_amp, 'r', label='Fz_cg theta amplitude')
    ax[0].plot(data_ch_theta, 'k', alpha=0.5, label='Fz theta filtered signal')
    ax[1].plot(data_cg_theta, 'k', alpha=0.5, label='Fz_cg theta filtered signal')
    ax[0].set_ylabel('Amplitude (uV)')
    ax[1].set_ylabel('Amplitude (uV)')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # downsample the envelope to the same frequency as the IBI signals
    data_ch_theta_amp = decimate(data_ch_theta_amp, filtered_data['Fs_EEG'] // filtered_data['Fs_IBI'], axis=-1)
    data_cg_theta_amp = decimate(data_cg_theta_amp, filtered_data['Fs_EEG'] // filtered_data['Fs_IBI'], axis=-1)

    # zscore the theta amplitude signals
    data_ch_theta_amp = zscore(data_ch_theta_amp)  # normalize the theta amplitude signals
    data_cg_theta_amp = zscore(data_cg_theta_amp)  # normalize the theta amplitude signals

    # Now we have the theta amplitude signals for both child and caregiver, let's get the IBI signals for the selected event
    ibi_ch_interp, ibi_cg_interp, t_ibi = get_ibi_signal_from_ecg_for_selected_event(filtered_data, events, event)
    # zscore the IBI signals
    ibi_ch_interp = zscore(ibi_ch_interp)  # normalize the IBI   signals
    ibi_cg_interp = zscore(ibi_cg_interp)  # normalize the IBI   signals

    # construct a numpy data array with the shape (4, N_samples), it will contain HRV and Fz theta amplitude of both child and caregiver
    dtf_data = np.zeros((4, len(data_ch_theta_amp)))
    # fill the data array with the IBI signals and Fz theta amplitude signals
    dtf_data[0, :] = ibi_ch_interp
    dtf_data[1, :] = ibi_cg_interp
    dtf_data[2, :] = data_ch_theta_amp
    dtf_data[3, :] = data_cg_theta_amp

    # plot the data for the child and caregiver IBI signals and Fz theta amplitude signals
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f'Child IBI signal and Fz theta amplitude for {event}')
    ax[1].set_title(f'Caregiver IBI signal and Fz_cg theta amplitude for {event}')
    ax[2].set_title(f'Child Fz theta amplitude for {event}')
    ax[3].set_title(f'Caregiver Fz_cg theta amplitude for {event}')
    ax[0].plot(t_ibi, ibi_ch_interp, 'b', label='Child IBI signal')
    ax[1].plot(t_ibi, ibi_cg_interp, 'b', label='Caregiver IBI signal')
    ax[2].plot(t_ibi, data_ch_theta_amp, 'r', label='Child Fz theta amplitude')
    ax[3].plot(t_ibi, data_cg_theta_amp, 'r', label='Caregiver Fz_cg theta amplitude')
    ax[0].set_ylabel('IBI (ms)')
    ax[1].set_ylabel('IBI (ms)')
    ax[2].set_ylabel('Fz theta amplitude (uV)')
    ax[3].set_ylabel('Fz_cg theta amplitude (uV)')
    ax[3].set_xlabel('Samples (after decimation)')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')
    ax[3].legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return dtf_data


def debug_plot(filtered_data, events):
    print("Filtered data shape:", filtered_data['data'].shape)
    print("Filtered EEG channels:", filtered_data['EEG_channels_ch'])
    print("Filtered ECG channels:", filtered_data['EEG_channels_cg'])
    print("Events detected:", events)

    # separately (in subplots) for child and caregiver, plot the filtered ECG and overlay it with the interpolated IBI signals, highlithing the events

    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Plot Child ECG on left y-axis
    ax[0].plot(filtered_data['t_ECG'], filtered_data['ECG_ch'], label='Child ECG', color='tab:blue')
    ax[0].set_ylabel('ECG (uV)', color='tab:blue')
    ax[0].tick_params(axis='y', labelcolor='tab:blue')

    # Create a twin y-axis to plot IBI
    ax0b = ax[0].twinx()
    ax0b.plot(filtered_data['t_IBI'], filtered_data['IBI_ch_interp'], label='Child IBI', color='tab:orange')
    ax0b.set_ylabel('IBI (ms)', color='tab:orange')
    ax0b.tick_params(axis='y', labelcolor='tab:orange')
    ax[0].plot(filtered_data['t_IBI'], filtered_data['IBI_ch_interp'], label='Child IBI')
    colors = ['r', 'g', 'y', 'c', 'm']  # colors for different events
    for i, event in enumerate(events):
        if events[event] is not None:
            ax[0].axvspan(events[event], events[event] + 60, color=colors[i], alpha=0.2, label=f'{event} (60s)')
    ax[0].legend()

    ax[1].plot(filtered_data['t_ECG'], filtered_data['ECG_cg'], label='Caregiver ECG', color='tab:blue')
    ax[1].set_ylabel('ECG (uV)', color='tab:blue')
    ax[1].tick_params(axis='y', labelcolor='tab:blue')
    # Create a twin y-axis to plot IBI
    ax1b = ax[1].twinx()
    ax1b.plot(filtered_data['t_IBI'], filtered_data['IBI_cg_interp'], label='Caregiver IBI', color='tab:orange')
    ax1b.set_ylabel('IBI (ms)', color='tab:orange')
    ax1b.tick_params(axis='y', labelcolor='tab:orange')
    for i, event in enumerate(events):
        if events[event] is not None:
            ax[1].axvspan(events[event], events[event] + 60, color=colors[i], alpha=0.2, label=f'{event} (60s)')
    ax[1].legend()
    ax[1].set_xlabel('Time (s)')
    plt.suptitle('Filtered ECG and IBI signals with events highlighted')
    plt.tight_layout()
    plt.show()


def hrv_dtf(mmd, selected_event):
    # extract the IBI from the multimodal data structure for selected event
    # of child and of the caregiver
    # construct a numpy data array with the shape (N_samples, 2)
    # and estimate DTF for the event

    f = np.arange(0.01, 1, 0.01)  # frequency vector for the DTF estimation
    selected_data = mmd.data[mmd.data.events == selected_event]  # assuming mmd is a MultimodalData object with a 'data' attribute
    ibi_ch = selected_data['IBI_ch'].values
    ibi_cg = selected_data['IBI_cg'].values


    # zscore the IBI signals
    ibi_ch = zscore(ibi_ch)  # normalize the IBI   signals
    ibi_cg = zscore(ibi_cg)  # normalize the IBI   signals
    data = np.zeros((2, len(ibi_ch)))
    data[0, :] = ibi_ch  # [start_idx:end_idx]
    data[1, :] = ibi_cg  # [start_idx:end_idx]
    # downsample the IBI signals to 2 Hz 
    fs_IBI = 2  # target sampling frequency for IBI signals
    
    # Design low-pass filter at Nyquist for target rate
    nyq = mmd.fs / 2 # original Nyquist frequency
    cutoff = 0.9 * fs_IBI/2  # Hz, safely below target Nyquist 
    sos = butter(8, cutoff / nyq, btype='low', output='sos')
    # Filter then downsample
    filtered = sosfiltfilt(sos, data, axis=-1)
    decimated_data = filtered[:,::64]
    time = selected_data['time'].values
    time_decimated = time[::64]
    plt.figure()
    plt.plot(time, data[0, :], alpha=0.3, label='Child IBI (original)')
    plt.plot(time, data[1, :], alpha=0.3, label='Caregiver IBI (original)')
    plt.plot(time_decimated, decimated_data[0, :], label='Child IBI (decimated)')
    plt.plot(time_decimated, decimated_data[1, :], label='Caregiver IBI (decimated)')
    plt.title(f'IBI signals for event: {selected_event}')
    plt.xlabel('Time (s)')
    plt.ylabel('Z-scored IBI')
    plt.legend()
    plt.show()


    dtf = dtf_multivariate(decimated_data, f, fs_IBI, max_model_order=15, crit_type='AIC')
    spectra = multivariate_spectra(decimated_data, f, fs_IBI, max_model_order=15, crit_type='AIC')
    # Let's  plot the results in the table form.
    mvar_plot(spectra, dtf, f, 'From ', 'To ', ['Child', 'Caregiver'], 'DTF ' + selected_event, 'sqrt')
    plt.show()


def eeg_dtf(mmd, selected_events):
    # for each event extract the EEG signals
    # of child and of the caregiver
    # construct a numpy data array with the shape (N_samples, 19)
    # and estimate DTF for each event separately for child and caregiver EEG channels

    f = np.arange(1, 30, 0.5)  # frequency vector for the DTF estimation
    selected_eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1',
                            'O2']  # , , 'T3','T4',  'T6',  'T5'

    time, eeg_ch = mmd.get_signals(mode='EEG', member='ch', selected_channels=selected_eeg_channels, selected_events=selected_events, selected_times=None)
    #get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
    time, eeg_cg = mmd.get_signals(mode='EEG', member='cg', selected_channels=selected_eeg_channels, selected_events=selected_events, selected_times=None)
 

    # plot data using Plotly for interactive visualization
    overlay_eeg_channels_hyperscanning_pl(eeg_ch, eeg_cg, 
                                          event=selected_events[0], 
                                          selected_channels_ch =selected_eeg_channels, 
                                          selected_channels_cg =selected_eeg_channels,
                                          title='Filtered EEG Channels - Hyperscanning (Plotly)')

    p_opt = 9  # force the model order to be 9, this is a good compromise between the model complexity and the estimation accuracy

    dtf = dtf_multivariate(eeg_ch, f, mmd.fs, optimal_model_order=p_opt, comment='child')
    spectra = multivariate_spectra(eeg_ch, f, mmd.fs, optimal_model_order=p_opt)
    mvar_plot(spectra, dtf, f, 'From ', 'To ', selected_eeg_channels, 'DTF ch ' + selected_events[0], 'sqrt')

    dtf = dtf_multivariate(eeg_cg, f, mmd.fs, optimal_model_order=p_opt, comment='caregiver')
    spectra = multivariate_spectra(eeg_cg, f, mmd.fs, optimal_model_order=p_opt)
    mvar_plot(spectra, dtf, f, 'From ', 'To ', selected_eeg_channels, 'DTF cg ' + selected_events[0], 'sqrt')
    plt.show()


def eeg_hrv_dtf(mmd, selected_events):
    # Something interesting seems to happen in the theta band in the Fz electrode of both child and caregiver
    # Let's filter the channel Fz in the theta band, get the instantaneous amplitude of the activity and evaluate DTF for the system consisting of
    # HRV and Fz theta instantaneous amplitude of both members

    f = np.arange(0.01, 1, 0.01)  # frequency vector for the DTF estimation
    selected_channels = ['Fz']


    eeg_ch_Fz = mmd.get_signals(mode='EEG', member='ch', selected_channels=selected_channels, selected_events=selected_events, selected_times=None)[1]
    eeg_cg_Fz = mmd.get_signals(mode='EEG', member='cg', selected_channels=selected_channels, selected_events=selected_events, selected_times=None)[1]  
    # plot spectra of Fz channels
    f_ch, pxx_ch = signal.welch(eeg_ch_Fz[0, :], fs=mmd.fs, nperseg=1024)
    f_cg, pxx_cg = signal.welch(eeg_cg_Fz[0, :], fs=mmd.fs, nperseg=1024)
    plt.figure(figsize=(12, 6))
    plt.plot(f_ch, pxx_ch, label='Child Fz channel')
    plt.plot(f_cg, pxx_cg, label='Caregiver Fz_cg channel')
    plt.title(f'Power Spectrum of {selected_events[0]} for Child and Caregiver Fz channels')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (uV^2/Hz)')
    plt.xlim(0, 30)  # Limit x-axis to 30 Hz
    plt.legend()
    plt.grid()
    plt.show()
    # design a bandpass fir filter for the band of interest
    lowcut = 3.5  # Hz
    highcut = 5.5  # Hz
    b = signal.firwin(numtaps=255, cutoff=[lowcut, highcut], fs=mmd.fs, pass_zero=False)
    # filter the Fz channels
    filtered_eeg_ch_Fz = signal.lfilter(b, [1.0], eeg_ch_Fz[0, :])
    filtered_eeg_cg_Fz = signal.lfilter(b, [1.0], eeg_cg_Fz[0, :])
    # correct for the filter delay
    delay = (len(b) - 1) // 2
    filtered_eeg_ch_Fz = np.roll(filtered_eeg_ch_Fz, -delay)
    filtered_eeg_cg_Fz = np.roll(filtered_eeg_cg_Fz, -delay)
    # get the instantaneous amplitude (envelope) of the filtered signal using Hilbert transform
    eeg_ch_Fz_theta_amp = np.abs(signal.hilbert(filtered_eeg_ch_Fz))
    eeg_cg_Fz_theta_amp = np.abs(signal.hilbert(filtered_eeg_cg_Fz))    

    ibi_ch = mmd.get_signals(mode='IBI', member='ch', selected_channels=['IBI_ch'], selected_events=selected_events, selected_times=None)[1]
    time, ibi_cg = mmd.get_signals(mode='IBI', member='cg', selected_channels=['IBI_cg'], selected_events=selected_events, selected_times=None)
 

    # construct a numpy data array with the shape (4, N_samples), it will contain HRV and Fz theta amplitude of both child and caregiver
    dtf_data = np.zeros((4, eeg_ch_Fz_theta_amp.shape[0]))
    # fill the data array with the IBI signals and Fz theta amplitude signals
    dtf_data[0, :] = ibi_ch[0, :]
    dtf_data[1, :] = ibi_cg[0, :]
    dtf_data[2, :] = eeg_ch_Fz_theta_amp
    dtf_data[3, :] = eeg_cg_Fz_theta_amp    
    # zscore the data
    dtf_data = zscore(dtf_data, axis=1)
    fs_dtf = 2 # downsampled frequency Hz

    # plot the data for the child and caregiver IBI signals and Fz theta amplitude signals before downsampling
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f'Child IBI signal and Fz theta amplitude for {selected_events[0]}')
    ax[1].set_title(f'Caregiver IBI signal and Fz_cg theta amplitude for {selected_events[0]}')
    ax[2].set_title(f'Child Fz theta amplitude for {selected_events[0]}')
    ax[3].set_title(f'Caregiver Fz_cg theta amplitude for {selected_events[0]}')
    ax[0].plot(time, dtf_data[0, :], 'b', label='Child IBI signal')
    ax[1].plot(time, dtf_data[1, :], 'b', label='Caregiver IBI signal')
    ax[2].plot(time, dtf_data[2, :], 'r', label='Child Fz theta amplitude')
    ax[3].plot(time, dtf_data[3, :], 'r', label='Caregiver Fz_cg theta amplitude')
    ax[3].set_xlabel('Samples')
    plt.tight_layout()

    # downsample the data to the fs_dtf sampling frequency
    decimation_factor = int(mmd.fs // fs_dtf)
    # need to apply a low-pass filter before downsampling to avoid aliasing
    sos = signal.butter(8, 0.8 * (fs_dtf / 2) / (mmd.fs / 2), btype='low', output='sos')
    dtf_data = signal.sosfiltfilt(sos, dtf_data, axis=1)
    dtf_data = dtf_data[:, ::decimation_factor]
    time_dtf = time[::decimation_factor]

    # plot the data for the child and caregiver IBI signals and Fz theta amplitude signals after downsampling on the same figure
    ax[0].plot(time_dtf, dtf_data[0, :], 'b', label='Child IBI signal')
    ax[1].plot(time_dtf, dtf_data[1, :], 'b', label='Caregiver IBI signal')
    ax[2].plot(time_dtf, dtf_data[2, :], 'r', label='Child Fz theta amplitude')
    ax[3].plot(time_dtf, dtf_data[3, :], 'r', label='Caregiver Fz_cg theta amplitude')
    ax[0].set_ylabel('Z-scored IBI')
    ax[1].set_ylabel('Z-scored IBI')
    ax[2].set_ylabel('Z-scored Fz theta amp')
    ax[3].set_ylabel('Z-scored Fz_cg theta amp')
    ax[3].set_xlabel('Samples')
    plt.tight_layout()
    plt.show()

  
    #dtf_data = eeg_hrv_dtf_analyze_event(filtered_data, selected_channels_ch,
    #                                        selected_channels_cg, events, event)

    # estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
    dtf = dtf_multivariate(dtf_data, f, fs_dtf, max_model_order=15, crit_type='AIC')
    spectra = multivariate_spectra(dtf_data, f, fs_dtf, max_model_order=15, crit_type='AIC')
    # Let's  plot the results in the table form. 
    chan_names = ['Child IBI', 'Caregiver IBI', 'Child Fz theta amp', 'Caregiver Fz_cg theta amp']
    mvar_plot(spectra, dtf, f, 'From ', 'To ', chan_names, 'DTF ' + selected_events[0], 'sqrt')
    plt.show()

    # Finally let's plot the DTF results in the graph form using graph_plot  from mtmvar
    _, ax = plt.subplots(figsize=(10, 8))
    graph_plot(connectivity_matrix=dtf, ax=ax, freqs=f, freq_range=[0.2, 0.6], chan_names=chan_names,
                title='DTF ' + selected_events[0] + ' (0.2-0.6 Hz)')
    plt.show()


if __name__ == "__main__":
    sys.exit(main(plot_debug=False, 
                  analyze_hrv_dtf=False, 
                  analyze_eeg_dtf=False, 
                  analyze_eeg_hrv_dtf=True)
                  )