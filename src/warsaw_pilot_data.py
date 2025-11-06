from utils import load_warsaw_pilot_data, scan_for_events, filter_warsaw_pilot_data, \
    debug_plot, hrv_dtf, eeg_dtf, eeg_hrv_dtf, plot_EEG_channels_pl

if __name__ == "__main__":
    folder = '../DATA/W_010/'
    file = 'W_010.obci'
    selected_events = ['Movie_1', 'Movie_2',
                       'Movie_3']  # # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'

    debug_PLOT = True
    HRV_DTF = True  # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True  # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True

    data = load_warsaw_pilot_data(folder, file, plot=False)
    events = scan_for_events(data,
                             plot=True)  # indexes of events in the data, this is done before filtering to avoid artifacts in the diode signal
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
        eeg_dtf(filtered_data, events, selected_events)

    if EEG_HRV_DTF:
        eeg_hrv_dtf(filtered_data, events, selected_events)
