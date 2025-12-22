import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.signal import firwin, lfilter


def process_time_et(child_pos_df_0, caregiver_pos_df_0, child_pupil_df_0, caregiver_pupil_df_0, child_pupil_df_1,
                    caregiver_pupil_df_1, child_pupil_df_2,
                    caregiver_pupil_df_2, Fs=1024):
    '''
    Create common time vector based on min and max timestamps from gaze position and pupil dataframes for child and caregiver.
    The indexes 0, 1, 2 refer to: watching movies, talk 1 and talk2 parts of the experiment.
    :param child_pos_df: child gaze_positions_on_surface_Surface dataframe
    :param caregiver_pos_df: caregiver gaze_positions_on_surface_Surface dataframe
    :param child_pupil_df: child pupil_positions dataframe
    :param caregiver_pupil_df: caregiver pupil_positions dataframe
    :param Fs: sampling frequency
    :return: common time
    '''
    # pupil 000: watching movies
    min_child_pupil_0 = min(child_pupil_df_0['pupil_timestamp'])
    max_child_pupil_0 = max(child_pupil_df_0['pupil_timestamp'])
    min_caregiver_pupil_0 = min(caregiver_pupil_df_0['pupil_timestamp'])
    max_caregiver_pupil_0 = max(caregiver_pupil_df_0['pupil_timestamp'])
    # pupil 001: talk 1
    min_child_pupil_1 = min(child_pupil_df_1['pupil_timestamp'])
    max_child_pupil_1 = max(child_pupil_df_1['pupil_timestamp'])
    min_caregiver_pupil_1 = min(caregiver_pupil_df_1['pupil_timestamp'])
    max_caregiver_pupil_1 = max(caregiver_pupil_df_1['pupil_timestamp'])
    # pupil 002: talk 2
    min_child_pupil_2 = min(child_pupil_df_2['pupil_timestamp'])
    max_child_pupil_2 = max(child_pupil_df_2['pupil_timestamp'])
    min_caregiver_pupil_2 = min(caregiver_pupil_df_2['pupil_timestamp'])
    max_caregiver_pupil_2 = max(caregiver_pupil_df_2['pupil_timestamp'])
    # pos 000: watching movies - only during watching movies we have gaze positions on the screen
    min_caregiver_pos = min(caregiver_pos_df_0['gaze_timestamp'])
    min_child_pos = min(child_pos_df_0['gaze_timestamp'])
    max_child_pos = max(child_pos_df_0['gaze_timestamp'])
    max_caregiver_pos = max(caregiver_pos_df_0['gaze_timestamp'])

    min_time = min(min_caregiver_pos, min_child_pos, min_caregiver_pupil_0, min_child_pupil_0, min_caregiver_pupil_1,
                   min_child_pupil_1, min_caregiver_pupil_2,
                   min_child_pupil_2)
    max_ttime = max(max_caregiver_pos, max_child_pos, max_caregiver_pupil_0, max_child_pupil_0, max_caregiver_pupil_1,
                    max_child_pupil_1, max_caregiver_pupil_2,
                    max_child_pupil_2)
    print("min", min_time)
    print("max", max_ttime)
    times = np.arange(min_time, max_ttime, 1 / Fs)
    return times


def process_pos(pos_df, df, who):
    '''
    Process gaze position dataframe to get interpolated x and y gaze positions on common time vector.
    :param pos_df: gaze_positions_on_surface_Surface dataframe
    :param time: common time vector
    :return: interpolated x and y gaze positions
    '''
    pos_df = pos_df[pos_df['on_surf'] != False]
    x_result = pos_df.groupby('gaze_timestamp')['x_norm'].mean().reset_index()
    y_result = pos_df.groupby('gaze_timestamp')['y_norm'].mean().reset_index()
    # the camera samples with different frame rates during the experiments, and sometimes loses the eye,
    # so we need to interpolate the gaze positions to the common time vector
    x_interp = np.interp(df['time'], x_result['gaze_timestamp'], x_result['x_norm'])
    y_interp = np.interp(df['time'], y_result['gaze_timestamp'], y_result['y_norm'])
    col_name_x = f'ET_{who}_x'
    col_name_y = f'ET_{who}_y'
    if col_name_x not in df.columns:
        df[col_name_x] = None
    if col_name_y not in df.columns:
        df[col_name_y] = None

    df[col_name_x] = x_interp
    df[col_name_y] = y_interp


def process_pupil(pupil_df, df, who, model_confidence=0.9, median_size=10, order=351, cutoff=1, Fs=1000,
                  plot_flag=False):
    '''
    Process pupil dataframe to get filtered 3D pupil diameter on common time vector.
    :param df: common dataframe to store the results
    :param who: 'ch' or 'cg' to indicate child or caregiver
    :param pupil_df: pupil positions dataframe
    :param time: common time for all signals
    :param model_confidence: confidence level for 3D pupil model
    :param median_size: size of median filter
    :param order: order of low pass filter
    :param cutoff: frequency cutoff for low pass filter
    :param Fs: sampling frequency
    :param plot_flag: debug plot
    :return: array of filtered pupil diameters
    '''

    filtr_3d = pupil_df[pupil_df['model_confidence'] > model_confidence]
    filtr_3d = filtr_3d.copy()
    minimum = min(filtr_3d['pupil_timestamp'])
    maximum = max(filtr_3d['pupil_timestamp'])
    mask = (df['time'] >= minimum) & (df['time'] <= maximum)
    filtr_3d['diameter3d_median'] = ndimage.median_filter(filtr_3d['diameter_3d'], size=median_size)
    diameter3d_interp = np.interp(df['time'], filtr_3d['pupil_timestamp'], filtr_3d['diameter3d_median'])
    b = firwin(order, cutoff=cutoff, fs=Fs)
    # Remember the first sample
    miu = diameter3d_interp[0]
    diameter3d_interp_filtred = lfilter(b, a=[1], x=diameter3d_interp - miu)
    delay = (len(b) - 1) // 2

    # Delay correction
    diameter3d_interp_filtred_aligned = np.roll(diameter3d_interp_filtred, -delay)
    # Fix last samples
    diameter3d_interp_filtred_aligned[-delay:] = np.nan
    # Fix the level
    diameter3d_interp_filtred_aligned += miu

    # Debug plot
    if plot_flag:
        plt.figure()
        plt.plot(pupil_df['pupil_timestamp'], pupil_df['diameter_3d'], label='Raw')
        plt.plot(filtr_3d['pupil_timestamp'], filtr_3d['diameter3d_median'], label='Median Filtered')
        plt.plot(df['time'], diameter3d_interp, label='Interpolated')
        plt.plot(df['time'], diameter3d_interp_filtred_aligned)
        plt.show()
    col_name = f'ET_{who}_diameter3d'
    if col_name not in df.columns:
        df[col_name] = None

    df.loc[mask, col_name] = diameter3d_interp_filtred_aligned[mask]


def process_event_et(annotations, df, event_name=None):
    '''
    Process event annotations from eye-tracking to mark events in the common dataframe.
    Add column 'event' to df, if not present, with event names based on annotations.
    :param annotations: dataframe with event annotations
    :param df: common dataframe to store the results
    :param event_name: optional name to assign to all events
    :return: series with event names
    '''
    annotations['type'] = annotations['label'].str.split('_').str[0]
    annotations['event'] = annotations['label'].str.split('_').str[1]

    starts = annotations[annotations['type'] == 'start'][['event', 'timestamp']]
    stops = annotations[annotations['type'] == 'stop'][['event', 'timestamp']]

    starts = starts.rename(columns={'timestamp': 't_start'})
    stops = stops.rename(columns={'timestamp': 't_stop'})
    intervals = starts.merge(stops, on='event')

    if 'ET_event' not in df.columns:
        df['ET_event'] = None

    for _, row in intervals.iterrows():
        mask = (df['time'] >= row['t_start']) & (df['time'] <= row['t_stop'])
        if event_name is None:
            df.loc[mask, 'ET_event'] = row['event']
        else:
            df.loc[mask, 'ET_event'] = event_name


def process_blinks(blinks, df, who):
    '''
    Process blink annotations from eye-tracking to mark blinks in the common dataframe.
    Add column '{who}_blinks' to df, if not present, with blink confidence values based on annotations.
    :param blinks: dataframe with blink annotations
    :param df: common dataframe to store the results
    :param who: identifier for the subject (e.g., 'ch' or 'cg')
    :return: series with blink confidence values
    '''
    cols = ['start_timestamp', 'end_timestamp', 'confidence']
    blinks = blinks[cols]
    col_name = f'ET_{who}_blinks'
    if col_name not in df.columns:
        df[col_name] = 0.0

    for _, row in blinks.iterrows():
        mask = (df['time'] >= row['start_timestamp']) & (df['time'] <= row['end_timestamp'])
        df.loc[mask, col_name] = row['confidence']
