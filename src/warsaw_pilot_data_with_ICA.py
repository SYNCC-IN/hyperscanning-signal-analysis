import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from mtmvar import mvar_plot, mvar_plot_dense, multivariate_spectra
from src.mtmvar import DTF_multivariate
from src.utils import plot_EEG_channels_pl, hrv_dtf
from utils import load_warsaw_pilot_data, scan_for_events, filter_warsaw_pilot_data, \
    get_IBI_signal_from_ECG_for_selected_event, get_data_for_selected_channel_and_event, \
    eeg_hrv_dtf_analyze_event, debug_plot

if __name__ == "__main__":
    folder = './DATA/W_010/' #'./DATA/W_009/'
    file  =  'W_010.obci'   #'W_009.obci'
    selected_events = ['Movie_1', 'Movie_2', 'Movie_3'] # events to extract data for ; #, 'Talk_1', 'Talk_2'
    
    debug_PLOT = True
    HRV_DTF = True # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True

    data = load_warsaw_pilot_data(folder, file, plot=False)
    events = scan_for_events(data, plot = True) #indexes of events in the data, this is done before filtering to avoid artifacts in the diode signal
    filtered_data = filter_warsaw_pilot_data(data)
    if debug_PLOT:
        debug_plot(filtered_data, events)
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_ch'],
                             title='Filtered Child EEG Channels (offset for clarity)')
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_cg'],
                             title='Filtered Caregiver EEG Channels (offset for clarity)')

    if HRV_DTF:
        hrv_dtf(filtered_data, events, selected_events)

    if EEG_DTF:
        # for each event extract the EEG signals 
        # of child and of the caregiver
        # costruct a numpy data array with the shape (N_samples, 19) 
        # and estimate DTF for each event separately for child and caregiver EEG channels

        f = np.arange(1,30,0.5 ) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'] #, , 'T3','T4',  'T6',  'T5'
        selected_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'O1_cg', 'O2_cg'] # , 'T3_cg', , 'T4_cg', , 'T5_cg', , 'T6_cg'
        selected_events = ['Movie_1', 'Movie_2', 'Movie_3'] # events to extract data for ; #, 'Talk_1', 'Talk_2'
        for event in selected_events:
            data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
            data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)

            # ICA = True # if True, apply ICA to the EEG data to remove artifacts
            # if ICA: # clean EEG data with ICA separately for child and caregiver EEG channels
            #     data_ch = clean_data_with_ICA(data_ch, selected_channels_ch, event)
            #     data_cg = clean_data_with_ICA(data_cg, selected_channels_cg, event)

            # plot the data for the child and caregiver EEG channels
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax[0].set_title(f'Child EEG channels for {event}')
            ax[1].set_title(f'Caregiver EEG channels for {event}')
            for i, ch in enumerate(selected_channels_ch):   
                if ch in filtered_data['channels']:
                    idx_ch = filtered_data['channels'][ch]
                    ax[0].plot(data_ch[i, :], label=ch)
            for i, ch in enumerate(selected_channels_cg):       
                if ch in filtered_data['channels']:
                    idx_cg = filtered_data['channels'][ch]
                    ax[1].plot(data_cg[i, :], label=ch)
            ax[0].set_ylabel('Amplitude (uV)')
            ax[1].set_ylabel('Amplitude (uV)')
            ax[1].set_xlabel('Samples (after decimation)')
            ax[0].legend(loc='upper right')
            ax[1].legend(loc='upper right')
            plt.tight_layout()

            DTF = DTF_multivariate(data_ch, f, filtered_data['Fs_EEG'], p_opt=9, comment='child')
            S = multivariate_spectra(data_ch, f, filtered_data['Fs_EEG'], p_opt=9)
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ',selected_channels_ch ,  'DTF ch '+ event ,'sqrt')

            DTF = DTF_multivariate(data_cg, f, filtered_data['Fs_EEG'], p_opt=9, comment='caregiver')
            S = multivariate_spectra(data_cg, f, filtered_data['Fs_EEG'], p_opt=9)
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ', selected_channels_cg,  'DTF cg '+ event ,'sqrt')  
            plt.show()


    if EEG_HRV_DTF:
    # Something interesting seems to happen in the theta band in the Fz electrode of both child and caregiver
    # Let's filter the channel Fz in the theta band, get the instantaneous amplitude of the activity and evaluate DTF for the system consisting of
    # HRV and Fz theta instantaneous amplitude of both members
        f = np.arange(0.01,1,0.01) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fz'] 
        selected_channels_cg = ['Fz_cg']
        selected_events = ['Movie_1', 'Movie_2', 'Movie_3']

        for event in selected_events:
            DTF_data = eeg_hrv_dtf_analyze_event(filtered_data, selected_channels_ch, selected_channels_cg, events, event)

            # estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
            DTF = DTF_multivariate(DTF_data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
            S = multivariate_spectra(DTF_data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
            """Let's  plot the results in the table form."""
            mvar_plot(S, DTF,   f, 'From ', 'To ',['Child IBI', 'Caregiver IBI', 'Child Fz theta amp', 'Caregiver Fz_cg theta amp'],  'DTF '+ event ,'sqrt')
            plt.show()
