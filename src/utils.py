import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, decimate, hilbert, welch
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots  # TODO czy trzeba mieszać plotly i matplotlib?
from scipy.stats import zscore

from src.mtmvar import DTF_multivariate, mvar_plot, multivariate_spectra, mvar_plot_dense, graph_plot


def get_ibi_signal_from_ecg_for_selected_event(filtered_data, events, selected_event):
    '''Get IBI signal from ECG data for a specific event.
    Args:
        filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract IBI signal for
        Fs_IBI (int): Sampling frequency for the IBI signals.
        plot (bool): Whether to plot the IBI signal.
        label (str): Label for the plot.
    Returns:
        IBI_ch_interp (np.ndarray): Interpolated IBI signal for the child.
        IBI_cg_interp (np.ndarray): Interpolated IBI signal for the caregiver.
        t_ECG (np.ndarray): Time vector for the interpolated IBI signal.
    '''
    if selected_event not in events:
        raise ValueError(f"Event '{selected_event}' not found in events dictionary.")
        # extract the ECG signal for the selected event
    ibi_ch_interp = filtered_data['IBI_ch_interp']
    ibi_cg_interp = filtered_data['IBI_cg_interp']
    t_ibi = filtered_data['t_IBI']
    t_idx = events[selected_event]  # get the time of the event in the data
    if t_idx is not None:
        # extract 60 seconds after the event
        # find the index in t_IBI
        start_idx = np.searchsorted(t_ibi, t_idx)  # find the index in t_IBI
        end_idx = start_idx + int(60 * filtered_data['Fs_IBI'])  # extract 60 seconds after the event
        # check if the start and end indices are within the bounds of the data
        if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
            raise ValueError(f"Event '{selected_event}' is out of bounds.")
    else:
        raise ValueError(f"Event '{selected_event}' is None.")

    # cut the IBI signal of the selected event
    ibi_ch_interp = ibi_ch_interp[start_idx:end_idx]
    ibi_cg_interp = ibi_cg_interp[start_idx:end_idx]
    t_ibi = t_ibi[start_idx:end_idx]

    return ibi_ch_interp, ibi_cg_interp, t_ibi


def get_data_for_selected_channel_and_event(filtered_data, selected_channels, events, selected_event):
    '''Get data for selected channels and event from the filtered data.
    Args:
        data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        selected_channels (list): List of channel names to extract data for.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract data for.
    Returns:    
        data_selected (np.ndarray): Data array with the shape (N_samples, N_channels) for the selected channels and event.
    '''
    if selected_event not in events:
        raise ValueError(f"Event '{selected_event}' not found in events dictionary.")
    idx = events[selected_event]
    if idx is not None:
        # extract 60 seconds after the event
        data_selected = np.zeros((len(selected_channels), int(60 * filtered_data['Fs_EEG'])))
        start_idx = int(idx * filtered_data['Fs_EEG'])  # convert the event time to the index in the filtered data
        end_idx = start_idx + int(60 * filtered_data['Fs_EEG'])  # extract 60 seconds after the event
        # check if the start and end indices are within the bounds of the data
        if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
            raise ValueError(f"Event '{selected_event}' is out of bounds.")
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx_ch = filtered_data['channels'][ch]
            data_selected[i, :] = filtered_data['data'][idx_ch, start_idx:end_idx]
    return data_selected


def clean_data_with_ICA(data, selected_channels, event):
    '''Clean data with ICA to remove artifacts.
    Args:
        data (np.ndarray): Data array with the shape (N_channels, N_samples) for the selected channels and event.
        selected_channels (list): List of channel names to extract data for.
        event (str): Name of the event to extract data for.
        plot (bool): Whether to plot the data before and after ICA.
    Returns:
        data_cleaned (np.ndarray): Cleaned data array with the shape (N_channels, N_samples) for the selected channels and event.
    '''
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=len(selected_channels), max_iter=1000, whiten="unit-variance")
    S_ = ica.fit_transform(data.T)  # get components

    _, ax = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)
    for i, ch in enumerate(selected_channels):
        ax[i].plot(S_[:, i])
        ax[i].set_ylabel(ch)
    plt.tight_layout()
    plt.show()
    idx_to_remove = input(f'Event {event}: select components to remove and press Enter to continue...  ')
    if idx_to_remove != '':
        idx_to_remove = [int(i) for i in idx_to_remove.split(',')]
        S_[:, idx_to_remove] = 0  # set the selected components to zero
        print('Selected components to remove: ', idx_to_remove)
    data_cleaned = ica.inverse_transform(S_).T  # reconstruct the data from the components   
    return data_cleaned


### PLOTS ####

def plot_eeg_channels_pl(filtered_data, events, selected_channels, title='Filtered EEG Channels', renderer='auto'):
    """
    Plot the filtered EEG channels with events highlighted using Plotly.
    Replicates the matplotlib version with vertical offsets in a single plot.
    Features interactive hover information and zooming capabilities.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing EEG data, channels, and time vectors
    events : dict
        Dictionary containing event timings
    selected_channels : list
        List of channel names to plot
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # colors for different events

    # Create a single figure
    fig = go.Figure()

    offset = 0
    spacing = 200  # vertical spacing between channels
    y_ticks = []
    y_tick_labels = []

    # Plot each channel with vertical offset
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx = filtered_data['channels'][ch]
            x_ch = filtered_data['data'][idx, :]
            # clip the amplitudes
            x_ch = np.clip(x_ch, -100, 100)

            # Add trace for this channel with offset
            fig.add_trace(go.Scatter(
                x=filtered_data['t_EEG'],
                y=x_ch + offset,
                mode='lines',
                name=ch,
                line={'width': 1},
                showlegend=True
            ))

            y_ticks.append(offset)
            y_tick_labels.append(ch)
            offset += spacing

    # Add event highlights as vertical rectangles spanning all channels
    event_colors_used = []
    for i, event in enumerate(events):
        if events[event] is not None:
            color_idx = i % len(colors)
            fig.add_vrect(
                x0=events[event],
                x1=events[event] + 60,
                fillcolor=colors[color_idx],
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f'{event} (60s)',
                annotation_position="top left",
                annotation={'font': {'size': 10, 'color': colors[color_idx]}, 'bgcolor': "white",
                            'bordercolor': colors[color_idx], 'borderwidth': 1}
            )
            if color_idx not in event_colors_used:
                # Add invisible trace for legend
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker={'size': 10, 'color': colors[color_idx]},
                    name=f'{event} Events',
                    showlegend=True
                ))
                event_colors_used.append(color_idx)

    # Update layout to match matplotlib appearance with enhanced interactivity
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        xaxis_title="Time [s]",
        yaxis_title="EEG Channels",
        yaxis={'tickvals': y_ticks, 'ticktext': y_tick_labels, 'showgrid': False, 'zeroline': False},
        xaxis={'showgrid': True, 'gridwidth': 1, 'gridcolor': 'lightgray'},
        height=600,
        width=1200,
        showlegend=True,
        legend={'orientation': "v", 'yanchor': "top", 'y': 1, 'xanchor': "left", 'x': 1.01, 'font': {'size': 10}},
        plot_bgcolor='white'
    )

    # Add range selector for time navigation
    fig.update_layout(
        xaxis={'rangeselector': {'buttons': [
            {'count': 30, 'label': "30s", 'step': "second", 'stepmode': "backward"},
            {'count': 60, 'label': "1m", 'step': "second", 'stepmode': "backward"},
            {'count': 300, 'label': "5m", 'step': "second", 'stepmode': "backward"},
            {'step': "all"}
        ]}, 'rangeslider': {'visible': True, 'thickness': 0.05}, 'type': "linear"}
    )

    # Show the figure based on renderer preference
    if renderer == 'html':
        save_figure_to_html(fig, title)
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except Exception as e:
            print(f"Failed to display with renderer 'browser': {e}")
            try:
                fig.show(renderer="notebook")
            except Exception as e:
                print(f"Failed to display with renderer 'notebook': {e}")
                save_figure_to_html(fig, title)
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            save_figure_to_html(fig, title)


def overlay_eeg_channels_hyperscanning(data_ch, data_cg, all_channels, event, selected_channels_ch,
                                       selected_channels_cg, title='Filtered EEG Channels - Hyperscanning'):
    """
    Overlay EEG channels for child and caregiver during a specific event.
    """
    _, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f'Child EEG channels for {event}')
    ax[1].set_title(f'Caregiver EEG channels for {event}')
    for i, ch in enumerate(selected_channels_ch):
        if ch in all_channels:
            ax[0].plot(data_ch[i, :], label=ch)
    for i, ch in enumerate(selected_channels_cg):
        if ch in all_channels:
            ax[1].plot(data_cg[i, :], label=ch)
    ax[0].set_ylabel('Amplitude [uV]')
    ax[1].set_ylabel('Amplitude [uV]')
    ax[1].set_xlabel('Samples')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    plt.suptitle(title)
    plt.tight_layout()


def overlay_eeg_channels_hyperscanning_pl(data_ch, data_cg, all_channels, event, selected_channels_ch,
                                          selected_channels_cg, title='Filtered EEG Channels - Hyperscanning',
                                          renderer='auto'):
    """
    Plot child and caregiver EEG channels for hyperscanning analysis using Plotly.
    Creates two subplots: one for child channels and one for caregiver channels.
    
    Parameters:
    -----------
    data_ch : numpy.ndarray
        Child EEG data (channels x samples)
    data_cg : numpy.ndarray  
        Caregiver EEG data (channels x samples)
    all_channels : dict
        Dictionary of all available channels
    event : str
        Name of the event being plotted
    selected_channels_ch : list
        List of selected child channel names
    selected_channels_cg : list
        List of selected caregiver channel names
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels - Hyperscanning')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Child EEG channels for {event}', f'Caregiver EEG channels for {event}'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Color palette for channels
    colors = px.colors.qualitative.Set3

    # Plot child EEG channels
    for i, ch in enumerate(selected_channels_ch):
        if ch in all_channels:
            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=list(range(data_ch.shape[1])),
                    y=data_ch[i, :],
                    mode='lines',
                    name=ch,
                    line=dict(color=colors[color_idx], width=1.5),
                    legendgroup='child',
                    legendgrouptitle_text="Child Channels"
                ),
                row=1, col=1
            )

    # Plot caregiver EEG channels  
    for i, ch in enumerate(selected_channels_cg):
        if ch in all_channels:
            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=list(range(data_cg.shape[1])),
                    y=data_cg[i, :],
                    mode='lines',
                    name=ch,
                    line={'color': colors[color_idx], 'width': 1.5},
                    legendgroup='caregiver',
                    legendgrouptitle={'text': "Caregiver Channels"}
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        height=800,
        width=1200,
        showlegend=True,
        legend={'orientation': "v", 'yanchor': "top", 'y': 1, 'xanchor': "left", 'x': 1.01, 'font': {'size': 10},
                'groupclick': "toggleitem"},
        plot_bgcolor='white'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=2, col=1)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Show the figure based on renderer preference
    if renderer == 'html':
        save_figure_to_html(fig, title, event)
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except Exception as e:
            print(f"Failed to display with renderer 'browser': {e}")
            try:
                fig.show(renderer="notebook")
            except Exception as e:
                print(f"Failed to display with renderer 'notebook': {e}")
                save_figure_to_html(fig, title, event)
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            save_figure_to_html(fig, title, event)


# ==================================
# ==================================
# ==================================
# ==================================
# ==================================
# ==================================

def save_figure_to_html(fig, title, event=None):
    html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}{'' if event is None else '_' + event}.html"
    fig.write_html(html_file)
    print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
    print("Open this file in your web browser to view the interactive plot.")


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

            dtf = DTF_multivariate(data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
            spectra = multivariate_spectra(data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
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
            data_ch = clean_data_with_ICA(data_ch, selected_channels_ch, event)
            data_cg = clean_data_with_ICA(data_cg, selected_channels_cg, event)

        # plot the data for the child and caregiver EEG channels
        overlay_eeg_channels_hyperscanning(data_ch, data_cg, filtered_data['channels'], event, selected_channels_ch,
                                           selected_channels_cg, title='Filtered EEG Channels - Hyperscanning')

        # Also plot using Plotly for interactive visualization
        overlay_eeg_channels_hyperscanning_pl(data_ch, data_cg, filtered_data['channels'], event, selected_channels_ch,
                                              selected_channels_cg,
                                              title='Filtered EEG Channels - Hyperscanning (Plotly)')

        p_opt = 9  # force the model order to be 9, this is a good compromise between the model complexity and the estimation accuracy

        dtf = DTF_multivariate(data_ch, f, filtered_data['Fs_EEG'], p_opt=p_opt, comment='child')
        spectra = multivariate_spectra(data_ch, f, filtered_data['Fs_EEG'], p_opt=p_opt)
        mvar_plot_dense(spectra, dtf, f, 'From ', 'To ', selected_channels_ch, 'DTF ch ' + event, 'sqrt')

        dtf = DTF_multivariate(data_cg, f, filtered_data['Fs_EEG'], p_opt=p_opt, comment='caregiver')
        spectra = multivariate_spectra(data_cg, f, filtered_data['Fs_EEG'], p_opt=p_opt)
        mvar_plot_dense(spectra, dtf, f, 'From ', 'To ', selected_channels_cg, 'DTF cg ' + event, 'sqrt')
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
        dtf = DTF_multivariate(dtf_data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
        spectra = multivariate_spectra(dtf_data, f, filtered_data['Fs_IBI'], max_p=15, crit_type='AIC')
        """Let's  plot the results in the table form."""
        chan_names = ['Child IBI', 'Caregiver IBI', 'Child Fz theta amp', 'Caregiver Fz_cg theta amp']
        mvar_plot(spectra, dtf, f, 'From ', 'To ', chan_names, 'DTF ' + event, 'sqrt')
        plt.show()

        # Finally let's plot the DTF results in the graph form using graph_plot  from mtmvar
        _, ax = plt.subplots(figsize=(10, 8))
        graph_plot(connectivity_matrix=dtf, ax=ax, f=f, f_range=[0.2, 0.6], chan_names=chan_names,
                   title='DTF ' + event)
        plt.show()
