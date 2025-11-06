import numpy as np
import matplotlib.pyplot as plt
from mtmvar import mvar_plot, mvar_plot_dense, DTF_multivariate, \
    multivariate_spectra, graph_plot
from utils import load_warsaw_pilot_data, scan_for_events, filter_warsaw_pilot_data, \
    get_data_for_selected_channel_and_event, debug_plot, hrv_dtf, eeg_dtf
from utils import plot_EEG_channels_pl, overlay_EEG_channels_hyperscanning_pl, overlay_EEG_channels_hyperscanning
from utils import eeg_hrv_dtf_analyze_event


if __name__ == "__main__":
    folder = '../DATA/W_010/' 
    file  =  'W_010.obci'
    selected_events = ['Movie_1', 'Movie_2', 'Movie_3']# # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'

    debug_PLOT = True
    HRV_DTF = True # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True

    data = load_warsaw_pilot_data(folder, file, plot=False)
    events = scan_for_events(data, plot = True) #indexes of events in the data, this is done before filtering to avoid artifacts in the diode signal
    filtered_data = filter_warsaw_pilot_data(data)

    # First lets examine the data
    if debug_PLOT:
        debug_plot(filtered_data, events)
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_ch'], 
                            title='Filtered Child EEG Channels (Plotly)')
        plot_EEG_channels_pl(filtered_data, events, filtered_data['EEG_channels_cg'], 
                            title='Filtered Caregiver EEG Channels (Plotly)')
        
    if HRV_DTF:
        hrv_dtf(filtered_data, events, selected_events)
    
    if EEG_DTF:
        eeg_dtf(filtered_data, events, selected_events)

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

