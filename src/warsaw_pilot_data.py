import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore  # type: ignore
from mtmvar import mvar_criterion, AR_coeff, mvar_H, mvar_plot, mvar_plot_dense, DTF_multivariate, multivariate_spectra, graph_plot # type: ignore
from utils import load_warsaw_pilot_data, scan_for_events, filter_warsaw_pilot_data, get_IBI_signal_from_ECG_for_selected_event, get_data_for_selected_channel_and_event
from utils import plot_EEG_channels_pl, overlay_EEG_channels_hyperscanning_pl, overlay_EEG_channels_hyperscanning
from utils import eeg_hrv_dtf_analyze_event


if __name__ == "__main__":
    folder = '../DATA/W_010/' 
    file  =  'W_010.obci'
    selected_events = ['Movie_1']# # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'

    debug_PLOT = True
    HRV_DTF = True # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True

    data = load_warsaw_pilot_data(folder, file, plot=False)
    events = scan_for_events(data, plot = True) #indexes of events in the data, this is done before filtering to avoid artifacts in the diode signal
    filtered_data = filter_warsaw_pilot_data(data)

    # First lets examine the data
    if debug_PLOT:
        print("Filtered data shape:", filtered_data['data'].shape)
        print("Filtered EEG channels:", filtered_data['EEG_channels_ch'])
        print("Filtered ECG channels:", filtered_data['EEG_channels_cg'])
        print("Events detected:", events)

        # separately (in subplots) for child and caregiver, plot the filtered ECG and overlay it with the interpolated IBI signals, highlithing the events

        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
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


        # # plot the filtered EEG channels
        # plot the filtered EEG channels for child
        #plot_EEG_channels(filtered_data, events, filtered_data['EEG_channels_ch'], title='Filtered Child EEG Channels')
        #plot_EEG_channels(filtered_data, events, filtered_data['EEG_channels_cg'], title='Filtered Caregiver EEG Channels')
        
        # Also plot using Plotly for interactive visualization
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_ch'], 
                            title='Filtered Child EEG Channels (Plotly)')
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_cg'], 
                            title='Filtered Caregiver EEG Channels (Plotly)')
        
    if HRV_DTF:
        # for each event extract the IBI signals from the ECG amplifier
        # of child and of the caregiver
        # costruct a numpy data array with the shape (N_samples, 2) 
        # and estimate DTF for each event
        
        f = np.arange(0.01, 1, 0.01) # frequency vector for the DTF estimation
        for event in selected_events:
            if event in events:
                # extract 60 seconds after the event
                data = np.zeros((2, 60*filtered_data['Fs_IBI']))
                IBI_ch_interp, IBI_cg_interp, t_IBI = get_IBI_signal_from_ECG_for_selected_event(filtered_data, events, event, plot=False, label='IBI signals for ' + event )
                # zscore the IBI signals
                IBI_ch_interp = zscore(IBI_ch_interp) # normalize the IBI   signals
                IBI_cg_interp = zscore(IBI_cg_interp) # normalize the IBI   signals
                data[0, :] = IBI_ch_interp #[start_idx:end_idx]
                data[1, :] = IBI_cg_interp #[start_idx:end_idx]

                S = multivariate_spectra(data, f, Fs = filtered_data['Fs_IBI'], max_p = 15, p_opt = None, crit_type='AIC')
                DTF = DTF_multivariate(data, f, Fs = filtered_data['Fs_IBI'], max_p = 15, p_opt = None, crit_type='AIC')
                """Let's  plot the results in the table form."""
                mvar_plot(S, DTF,   f, 'From ', 'To ',['Child', 'Caregiver'],  'DTF '+ event ,'sqrt')
        plt.show()

    
    if EEG_DTF:
        # for each event extract the EEG signals 
        # of child and of the caregiver
        # costruct a numpy data array with the shape (N_samples, 19) 
        # and estimate DTF for each event separately for child and caregiver EEG channels

        f = np.arange(1,30,0.5 ) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'] #, , 'T3','T4',  'T6',  'T5'
        selected_channels_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'O1_cg', 'O2_cg'] # , 'T3_cg', , 'T4_cg', , 'T5_cg', , 'T6_cg'
        
        for event in selected_events:
            data_ch = get_data_for_selected_channel_and_event(filtered_data, selected_channels_ch, events, event)
            data_cg = get_data_for_selected_channel_and_event(filtered_data, selected_channels_cg, events, event)

            # plot the data for the child and caregiver EEG channels
            overlay_EEG_channels_hyperscanning(data_ch, data_cg, filtered_data['channels'], event, selected_channels_ch, selected_channels_cg, title='Filtered EEG Channels - Hyperscanning')
            
            # Also plot using Plotly for interactive visualization
            overlay_EEG_channels_hyperscanning_pl(data_ch, data_cg, filtered_data['channels'], event, selected_channels_ch, selected_channels_cg, title='Filtered EEG Channels - Hyperscanning (Plotly)')

            p_opt =9 # force the model order to be 9, this is a good compromise between the model complexity and the estimation accuracy

            S = multivariate_spectra(data_ch, f, Fs = filtered_data['Fs_EEG'], max_p = 15, p_opt = p_opt, crit_type='AIC')
            DTF = DTF_multivariate(data_ch, f, Fs = filtered_data['Fs_EEG'], max_p = 15, p_opt = p_opt, crit_type='AIC')
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ',selected_channels_ch ,  'DTF ch '+ event ,'sqrt')

            S = multivariate_spectra(data_cg, f, Fs = filtered_data['Fs_EEG'], max_p = 15, p_opt = p_opt, crit_type='AIC')
            DTF = DTF_multivariate(data_cg, f, Fs = filtered_data['Fs_EEG'], max_p = 15, p_opt = p_opt, crit_type='AIC')
            mvar_plot_dense(S, DTF,   f, 'From ', 'To ', selected_channels_cg,  'DTF cg '+ event ,'sqrt')
            plt.show()


    if EEG_HRV_DTF:
    # Something interesting seems to happen in the theta band in the Fz electrode of both child and caregiver
    # Let's filter the channel Fz in the theta band, get the instantaneous amplitude of the activity and evaluate DTF for the system consisting of
    # HRV and Fz theta instantaneous amplitude of both members
        f = np.arange(0.01,1,0.01) # frequency vector for the DTF estimation
        selected_channels_ch  = ['Fz'] 
        selected_channels_cg = ['Fz_cg']

        for event in selected_events:
            DTF_data = eeg_hrv_dtf_analyze_event(filtered_data, selected_channels_ch, selected_channels_cg, events, event)

            # Now we have the data for the DTF estimation, let's estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
            # estimate DTF for the system consisting of HRV and Fz theta amplitude of both child and caregiver
            S = multivariate_spectra(DTF_data, f, Fs = filtered_data['Fs_IBI'], max_p = 15, p_opt = None, crit_type='AIC')
            DTF = DTF_multivariate(DTF_data, f, Fs = filtered_data['Fs_IBI'], max_p = 15, p_opt = None, crit_type='AIC')    
            """Let's  plot the results in the table form."""
            ChanNames = ['Ch IBI', 'Cg IBI', 'Ch Fz\n theta amp', 'Cg Fz_cg\n theta amp']
            mvar_plot(S, DTF,   f, 'From ', 'To ',ChanNames,  'DTF '+ event ,'sqrt')
            plt.show()

            # Finally let's plot the DTF results in the graph form using graph_plot  from mtmvar
            fig, ax = plt.subplots(figsize=(10, 8))
            graph_plot(connectivity_matrix = DTF, ax=ax, f=f, f_range=[0.2, 0.6], ChanNames=ChanNames, title='DTF ' + event)
            plt.show()

