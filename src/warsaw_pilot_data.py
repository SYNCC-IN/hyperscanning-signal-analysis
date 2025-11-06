from src.DataLoader import DataLoader
from utils import debug_plot, hrv_dtf, eeg_dtf, eeg_hrv_dtf, plot_EEG_channels_pl

if __name__ == "__main__":
    folder = '../DATA/W_010/'
    file = 'W_010.obci'
    selected_events = ['Movie_1', 'Movie_2',
                       'Movie_3']  # # events to extract data for ; #, 'Movie_2', 'Movie_3', 'Talk_1', 'Talk_2'

    debug_PLOT = True
    HRV_DTF = True  # if True, the DTF will be estimated for the IBI signals from the ECG amplifier
    EEG_DTF = True  # if True, the DTF will be estimated for the EEG signals from child and caregiver separately
    EEG_HRV_DTF = True


    data = DataLoader("W_010", False)
    data.set_eeg_data("../DATA/W_010/")
    events = data.events

    if debug_PLOT:
        debug_plot(data, events)
        plot_EEG_channels_pl(data, events, data.channel_names['EEG']['ch'],
                             title='Filtered Child EEG Channels (offset for clarity)')
        plot_EEG_channels_pl(data, events, data.channel_names['EEG']['cg'],
                             title='Filtered Caregiver EEG Channels (offset for clarity)')

    if HRV_DTF:
        hrv_dtf(data, events, selected_events)

    if EEG_DTF:
        eeg_dtf(data, events, selected_events)

    if EEG_HRV_DTF:
        eeg_hrv_dtf(data, events, selected_events)
