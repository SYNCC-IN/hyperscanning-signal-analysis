import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import decimate, hilbert, sosfiltfilt, welch, butter
from scipy.stats import zscore

from src.DataLoader import DataLoader
from src.mtmvar import graph_plot, mvar_plot, multivariate_spectra, dtf_multivariate
from utils import plot_eeg_channels_pl, overlay_eeg_channels_hyperscanning_pl, clean_data_with_ica, \
    get_data_for_selected_channel_and_event, get_ibi_signal_from_ecg_for_selected_event


def main(plot_debug=False, analyze_hrv_dtf=False, analyze_eeg_dtf=False, analyze_eeg_hrv_dtf=False):
    selected_events = ['Movie_1', 'Movie_2',
                       'Movie_3']  # # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'


    data = DataLoader("W_010", False)
    data.set_eeg_data("../DATA/W_010/")
    events = data.events

    if plot_debug:
        debug_plot(data, events)
        plot_eeg_channels_pl(data, events, data.channel_names['EEG']['ch'],
                             title='Filtered Child EEG Channels (offset for clarity)')
        plot_eeg_channels_pl(data, events, data.channel_names['EEG']['cg'],
                             title='Filtered Caregiver EEG Channels (offset for clarity)')

    if analyze_hrv_dtf:
        hrv_dtf(data, events, selected_events)

    if analyze_eeg_dtf:
        eeg_dtf(data, events, selected_events)

    if analyze_eeg_hrv_dtf:
        eeg_hrv_dtf(data, events, selected_events)



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

    # filter the data in the theta band
    data_ch_theta = sosfiltfilt(sos_theta, data_ch[0, :])
    data_cg_theta = sosfiltfilt(sos_theta, data_cg[0, :])
    # get the instantaneous amplitude (envelpe) of the filtered signal using Hilbert transform
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


def hrv_dtf(filtered_data, events, selected_events):
    # for each event extract the IBI signals from the ECG amplifier
    # of child and of the caregiver
    # construct a numpy data array with the shape (N_samples, 2)
    # and estimate DTF for each event

    f = np.arange(0.01, 1, 0.01)  # frequency vector for the DTF estimation
    for event in selected_events:
        if event in events:
            # extract 60 seconds after the event
            data = np.zeros((2, 60 * filtered_data['Fs_IBI']))
            ibi_ch_interp, ibi_cg_interp, _ = get_ibi_signal_from_ecg_for_selected_event(filtered_data, events, event)
            # zscore the IBI signals
            ibi_ch_interp = zscore(ibi_ch_interp)  # normalize the IBI   signals
            ibi_cg_interp = zscore(ibi_cg_interp)  # normalize the IBI   signals
            data[0, :] = ibi_ch_interp  # [start_idx:end_idx]
            data[1, :] = ibi_cg_interp  # [start_idx:end_idx]

            dtf = dtf_multivariate(data, f, filtered_data['Fs_IBI'], max_model_order=15, crit_type='AIC')
            spectra = multivariate_spectra(data, f, filtered_data['Fs_IBI'], max_model_order=15, crit_type='AIC')
            """Let's  plot the results in the table form."""
            mvar_plot(spectra, dtf, f, 'From ', 'To ', ['Child', 'Caregiver'], 'DTF ' + event, 'sqrt')
    plt.show()


def eeg_dtf(filtered_data, events, selected_events, clean_with_ica=True):
    # for each event extract the EEG signals
    # of child and of the caregiver
    # construct a numpy data array with the shape (N_samples, 19)
    # and estimate DTF for each event separately for child and caregiver EEG channels

    f = np.arange(1, 30, 0.5)  # frequency vector for the DTF estimation
    selected_channels_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1',
                            'O2']  # , , 'T3','T4',  'T6',  'T5'
    selected_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'C3_cg', 'Cz_cg', 'C4_cg',
                            'P3_cg', 'Pz_cg', 'P4_cg', 'O1_cg', 'O2_cg']  # , 'T3_cg', , 'T4_cg', , 'T5_cg', , 'T6_cg'

    for event in selected_events:
        data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
        data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)

        if clean_with_ica:  # clean EEG data with ICA separately for child and caregiver EEG channels
            data_ch = clean_data_with_ica(data_ch, selected_channels_ch, event)
            data_cg = clean_data_with_ica(data_cg, selected_channels_cg, event)

        # plot data using Plotly for interactive visualization
        overlay_eeg_channels_hyperscanning_pl(data_ch, data_cg, filtered_data['channels'], event, selected_channels_ch,
                                              selected_channels_cg,
                                              title='Filtered EEG Channels - Hyperscanning (Plotly)')

        p_opt = 9  # force the model order to be 9, this is a good compromise between the model complexity and the estimation accuracy

        dtf = dtf_multivariate(data_ch, f, filtered_data['Fs_EEG'], optimal_model_order=p_opt, comment='child')
        spectra = multivariate_spectra(data_ch, f, filtered_data['Fs_EEG'], optimal_model_order=p_opt)
        mvar_plot(spectra, dtf, f, 'From ', 'To ', selected_channels_ch, 'DTF ch ' + event, 'sqrt')

        dtf = dtf_multivariate(data_cg, f, filtered_data['Fs_EEG'], optimal_model_order=p_opt, comment='caregiver')
        spectra = multivariate_spectra(data_cg, f, filtered_data['Fs_EEG'], optimal_model_order=p_opt)
        mvar_plot(spectra, dtf, f, 'From ', 'To ', selected_channels_cg, 'DTF cg ' + event, 'sqrt')
        plt.show()


def eeg_hrv_dtf(filtered_data, events, selected_events):
    # Something interesting seems to happen in the theta band in the Fz electrode of both child and caregiver
    # Let's filter the channel Fz in the theta band, get the instantaneous amplitude of the activity and evaluate DTF for the system consisting of
    # HRV and Fz theta instantaneous amplitude of both members

    f = np.arange(0.01, 1, 0.01)  # frequency vector for the DTF estimation
    selected_channels_ch = ['Fz']
    selected_channels_cg = ['Fz_cg']

    for event in selected_events:
        dtf_data = eeg_hrv_dtf_analyze_event(filtered_data, selected_channels_ch,
                                             selected_channels_cg, events, event)

        # estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
        dtf = dtf_multivariate(dtf_data, f, filtered_data['Fs_IBI'], max_model_order=15, crit_type='AIC')
        spectra = multivariate_spectra(dtf_data, f, filtered_data['Fs_IBI'], max_model_order=15, crit_type='AIC')
        """Let's  plot the results in the table form."""
        chan_names = ['Child IBI', 'Caregiver IBI', 'Child Fz theta amp', 'Caregiver Fz_cg theta amp']
        mvar_plot(spectra, dtf, f, 'From ', 'To ', chan_names, 'DTF ' + event, 'sqrt')
        plt.show()

        # Finally let's plot the DTF results in the graph form using graph_plot  from mtmvar
        _, ax = plt.subplots(figsize=(10, 8))
        graph_plot(connectivity_matrix=dtf, ax=ax, freqs=f, freq_range=[0.2, 0.6], chan_names=chan_names,
                   title='DTF ' + event)
        plt.show()


if __name__ == "__main__":
    sys.exit(main(plot_debug=True, analyze_hrv_dtf=True, analyze_eeg_dtf=True, analyze_eeg_hrv_dtf=True))