import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots
from sklearn.decomposition import FastICA
import scipy.signal as signal   

def get_ibi_signal_from_ecg_for_selected_event(filtered_data, events, selected_event):
    """Get IBI signal from ECG data for a specific event.
    Args:
        filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract IBI signal for
    Returns:
        IBI_ch_interp (np.ndarray): Interpolated IBI signal for the child.
        IBI_cg_interp (np.ndarray): Interpolated IBI signal for the caregiver.
        t_ECG (np.ndarray): Time vector for the interpolated IBI signal.
    """
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
    """Get data for selected channels and event from the filtered data.
    Args:
        filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
        selected_channels (list): List of channel names to extract data for.
        events (dict): Dictionary containing the start and end time of detected events.
        selected_event (str): Name of the event to extract data for.
    Returns:
        data_selected (np.ndarray): Data array with the shape (N_samples, N_channels) for the selected channels and event.
    """
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
    else:
        raise ValueError(f"Event '{selected_event}' is None.")
    for i, ch in enumerate(selected_channels):
        if ch in filtered_data['channels']:
            idx_ch = filtered_data['channels'][ch]
            data_selected[i, :] = filtered_data['data'][idx_ch, start_idx:end_idx]
    return data_selected


def clean_data_with_ica(data, selected_channels, event):
    """Clean data with ICA to remove artifacts.
    Args:
        data (np.ndarray): Data array with the shape (N_channels, N_samples) for the selected channels and event.
        selected_channels (list): List of channel names to extract data for.
        event (str): Name of the event to extract data for.
    Returns:
        data_cleaned (np.ndarray): Cleaned data array with the shape (N_channels, N_samples) for the selected channels and event.
    """
    ica = FastICA(n_components=len(selected_channels), max_iter=1000, whiten="unit-variance")
    ica_components = ica.fit_transform(data.T)  # get components

    _, ax = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)
    for i, ch in enumerate(selected_channels):
        ax[i].plot(ica_components[:, i])
        ax[i].set_ylabel(ch)
    plt.tight_layout()
    plt.show()
    idx_to_remove = input(f'Event {event}: select components to remove and press Enter to continue...  ')
    if idx_to_remove != '':
        idx_to_remove = [int(i) for i in idx_to_remove.split(',')]
        ica_components[:, idx_to_remove] = 0  # set the selected components to zero
        print('Selected components to remove: ', idx_to_remove)
    data_cleaned = ica.inverse_transform(ica_components).T  # reconstruct the data from the components
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
                    line={'color': colors[color_idx], 'width': 1.5},
                    legendgroup='child',
                    legendgrouptitle={'text': "Child Channels"}
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

def plot_filter_characteristics(b, a, f, T, Fs, f_lim=None, db_lim=None):
    """
    Plot comprehensive filter characteristics including magnitude response, 
    group delay, impulse response, and step response.
    
    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients of the filter
    a : array_like
        Denominator polynomial coefficients of the filter
    f : array_like
        Frequency vector for frequency response computation (Hz)
    T : float
        Time duration for impulse and step response plots (seconds)
    Fs : float
        Sampling frequency (Hz)
    f_lim : tuple of float, optional
        Frequency axis limits (f_min, f_max) in Hz. Default is (0, Fs/2)
    db_lim : tuple of float, optional
        Magnitude axis limits (db_min, db_max) in dB. Default is auto-scaled
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Set default frequency limits
    if f_lim is None:
        f_lim = (0, Fs / 2)
    else:
        f_lim = (max(f_lim[0], 0), min(f_lim[1], Fs / 2))   
        f = f[np.logical_and(f >= f_lim[0], f <= f_lim[1])]
    
    # Create time vector
    time_vector = np.arange(-T, T, 1 / Fs)

    # Compute frequency response
    frequencies, frequency_response = signal.freqz(b, a, f, fs=Fs)
    magnitude = np.abs(frequency_response)
    magnitude_db = 20 * np.log10(magnitude)

    # Compute group delay
    _, group_delay = signal.group_delay((b, a), f, fs=Fs)

    # Compute impulse response
    impulse_signal = np.zeros(len(time_vector))
    impulse_signal[len(time_vector) // 2] = 1
    impulse_response = signal.lfilter(b, a, impulse_signal)

    # Compute step response
    step_signal = np.zeros(len(time_vector))
    step_signal[len(time_vector) // 2:] = 1
    step_response = signal.lfilter(b, a, step_signal)

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot magnitude response (top left)
    _plot_magnitude_response(frequencies, magnitude_db, f_lim, db_lim)
    
    # Plot group delay (bottom left)
    _plot_group_delay(frequencies, group_delay, f_lim)
    
    # Plot impulse response (top right)
    _plot_impulse_response(time_vector, impulse_signal, impulse_response, T)
    
    # Plot step response (bottom right)
    _plot_step_response(time_vector, step_signal, step_response, T)
    
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    
    return fig


def _find_db_crossings(frequencies, magnitude_db, target_db):
    """
    Find frequencies where magnitude response crosses a target dB value.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    magnitude_db : array_like
        Magnitude response in dB
    target_db : float
        Target dB value to find crossings
    
    Returns
    -------
    list
        List of frequencies where crossings occur
    """
    crossings = []
    for i in range(len(magnitude_db) - 1):
        # Check if the line segment crosses the target
        if (magnitude_db[i] >= target_db and magnitude_db[i + 1] < target_db) or \
           (magnitude_db[i] < target_db and magnitude_db[i + 1] >= target_db):
            # Linear interpolation to find exact crossing frequency
            freq_crossing = frequencies[i] + (target_db - magnitude_db[i]) * \
                           (frequencies[i + 1] - frequencies[i]) / \
                           (magnitude_db[i + 1] - magnitude_db[i])
            crossings.append(freq_crossing)
    return crossings


def _plot_magnitude_response(frequencies, magnitude_db, f_lim, db_lim):
    """
    Plot magnitude response of the filter in dB.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    magnitude_db : array_like
        Magnitude response in dB
    f_lim : tuple of float
        Frequency axis limits (f_min, f_max)
    db_lim : tuple of float or None
        Magnitude axis limits (db_min, db_max)
    """
    plt.subplot(2, 2, 1)
    plt.title('Magnitude Response')
    plt.plot(frequencies, magnitude_db)
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.xlim(f_lim)
    
    # Find and mark -6dB crossing points
    crossing_freqs = _find_db_crossings(frequencies, magnitude_db, -6.0)
    for freq in crossing_freqs:
        plt.axvline(x=freq, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
    
    if db_lim is None:
        # Auto-scale to show only the zoomed frequency range
        magnitude_zoom = magnitude_db[np.logical_and(f_lim[0] < frequencies, frequencies < f_lim[1])]
        plt.ylim((np.min(magnitude_zoom), np.max(magnitude_zoom)))
    else:
        plt.ylim((db_lim[0], db_lim[1]))


def _plot_group_delay(frequencies, group_delay, f_lim):
    """
    Plot group delay of the filter.
    
    Parameters
    ----------
    frequencies : array_like
        Frequency vector (Hz)
    group_delay : array_like
        Group delay in samples
    f_lim : tuple of float
        Frequency axis limits (f_min, f_max)
    """
    plt.subplot(2, 2, 3)
    plt.title('Group Delay')
    plt.plot(frequencies, group_delay)
    plt.ylabel('Samples')
    plt.xlabel('Frequency [Hz]')
    plt.grid(True)
    plt.xlim(f_lim)
    plt.ylim([np.min(group_delay) - 1, np.max(group_delay) + 1])


def _plot_impulse_response(time_vector, impulse_signal, impulse_response, T):
    """
    Plot impulse response of the filter.
    
    Parameters
    ----------
    time_vector : array_like
        Time vector (seconds)
    impulse_signal : array_like
        Input impulse signal
    impulse_response : array_like
        Filter's impulse response
    T : float
        Time duration (seconds)
    """
    plt.subplot(2, 2, 2)
    plt.title('Impulse Response')
    plt.stem(time_vector, impulse_signal)
    plt.plot(time_vector, impulse_response, 'r')
    plt.xlim([-T / 4, T])
    plt.grid(True)


def _plot_step_response(time_vector, step_signal, step_response, T):
    """
    Plot step response of the filter.
    
    Parameters
    ----------
    time_vector : array_like
        Time vector (seconds)
    step_signal : array_like
        Input step signal
    step_response : array_like
        Filter's step response
    T : float
        Time duration (seconds)
    """
    plt.subplot(2, 2, 4)
    plt.title('Step Response')
    plt.plot(time_vector, step_signal)
    plt.plot(time_vector, step_response, 'r')
    plt.xlim([-T / 4, T])
    plt.xlabel('Time [s]')
    plt.grid(True)

# ==================================

def save_figure_to_html(fig, title, event=None):
    html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}{'' if event is None else '_' + event}.html"
    fig.write_html(html_file)
    print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
    print("Open this file in your web browser to view the interactive plot.")
