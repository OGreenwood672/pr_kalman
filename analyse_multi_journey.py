
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from sensor_fusion import kalman_filters, height
from collections import deque

from globals import *
from utils import *
from analyse_journey import analyse_journey


import configparser

# Initialize the parser
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

FREQUENCY = config['data-collection'].getint('frequency')

ACCELEROMETER_VARIANCE = config['measurements'].getfloat('accelerometer_variance')

STAGE_NO_CHANGE_THRESHOLD = config['data-analysis'].getfloat('stage_no_change_threshold')
STAGE_ROLLING_AVG_LEN = config['data-analysis'].getint('stage_rolling_avg_len')

VELOCITY_PULL_TO_ZERO = config['data-analysis'].getfloat('velocity_pull_to_zero')

ACCELEROMETER_DRIFT_CORRECTION_BOUND = config['data-analysis'].getfloat('accelerometer_drift_correction')


def get_floor_journey(delta_heights):

    # Assume calibration (0 -> top)
    i = 0
    floors = [0]
    while i < len(delta_heights) and delta_heights[i] > 0: 
        floors.append(delta_heights[i] + floors[-1])
        i += 1
        
    floors = np.array(floors)
    journey = [*range(len(floors))]
    for dh in delta_heights[i:]:
        new_height = floors[journey[-1]] + dh
        journey.append(np.argmin(np.abs(floors - new_height)))
        
    return journey


def linear_offset(df, label, offset, start, stop):
    """
    time-dependent correction or transformation of the label values,
    with the offset controlling how much change happens across the time range
    @params
    df: dataframe
    label: column name to edit
    offset: 
    start: 
    stop: 
    """
    dt = df[TIMESTAMPS].iloc[stop] - df[TIMESTAMPS].iloc[start]
    for i in range(start, stop + 1):
        df.at[i, label] = df[label].iloc[i] - offset * ((df[TIMESTAMPS].iloc[i] - df[TIMESTAMPS].iloc[start]) / dt)


def get_dist(df, col, start, end):
    """
    Gets mean and variance of sample of col in df from start to end
    @params
    df: dataframe
    col: column name
    start: beginning of calculation
    end: end of calculation
    """
    mean = df.loc[start:end, col].mean()
    var_sum = 0
    for i in range(start, end):
        var_sum += (df[col].iloc[i] - mean) ** 2
    return mean, var_sum / (end - start - 1)


def get_stages(df, maxlen=20, mean_threshold=0.05, variance_threshold=0.0001, buffer_seconds=5):
    """
    Updates the STAGES column based on changes in rolling mean and variance of ACCELERATION.

    Args:
        df (pd.DataFrame): The DataFrame containing the ACCELERATION and TIMESTAMP columns.
        maxlen (int): Window size for rolling mean and variance calculations.
        mean_threshold (float): Threshold for the rolling mean.
        variance_threshold (float): Threshold for the variance.
        buffer_seconds (float): Minimum time (in seconds) of inactivity before transitioning from stage 1 -> 0.

    Returns:
        pd.DataFrame: The updated DataFrame with the STAGES column.
    """

    peak = None
    peak_sign = None
    min_peak = 0.1
    cycle_complete = False

    dt = df[TIMESTAMPS].iloc[1] - df[TIMESTAMPS].iloc[0]
    dt_count = buffer_seconds // dt

    # Initialize rolling window
    rolling_window = deque(maxlen=maxlen)
    df[STAGES] = 0  # Initialize STAGES column

    # Initialize stage and time tracking
    curr_stage = 0
    last_activity = df[TIMESTAMPS].iloc[0]  # Initialize last activity timestamp

    for i in range(len(df)):
        # Add current acceleration to the rolling window
        rolling_window.append(df[ACCELERATION].iloc[i])
        
        if len(rolling_window) < maxlen:
            continue  # Skip until the rolling window is full
        
        # Calculate rolling mean and variance
        rolling_mean = np.mean(rolling_window)
        rolling_variance = np.var(rolling_window)
        
        # Handle transitions
        if curr_stage == 0:
            if abs(rolling_mean) > mean_threshold or rolling_variance > variance_threshold:
                curr_stage = 1
                df.loc[max(i - (dt_count // 2), 0):, STAGES] = curr_stage
                last_activity = df[TIMESTAMPS].iloc[i]  # Update the last activity timestamp

                peak = None
                peak_sign = None
                cycle_complete = False

        elif curr_stage == 1:

            if min_peak < abs(df[ACCELERATION].iloc[i]):
                if peak == None:
                    peak = abs(df[ACCELERATION].iloc[i])
                    peak_sign = sign(df[ACCELERATION].iloc[i])
                elif abs(peak) > abs(df[ACCELERATION].iloc[i]) and sign(df[ACCELERATION].iloc[i]) == peak_sign:
                    peak = abs(df[ACCELERATION].iloc[i])
                elif peak_sign != sign(df[ACCELERATION].iloc[i]):
                    cycle_complete = True


            if abs(rolling_mean) <= mean_threshold and rolling_variance <= variance_threshold and cycle_complete:

                time_since_last_activity = df[TIMESTAMPS].iloc[i] - last_activity
                if time_since_last_activity > buffer_seconds:
                    curr_stage = 0
                    df.loc[max(i - maxlen, 0):, STAGES] = curr_stage
            else:
                last_activity = df[TIMESTAMPS].iloc[i]  # Update the last activity timestamp
        
        df.loc[i, STAGES] = curr_stage

    return df



def split_dataframe_by_stages(df, buffer):
    """
    Splits a DataFrame into sections based on changes in the stages column,
    separating sections by 000000 and adding a buffer of zeros on each side.

    Args:
        df (pd.DataFrame): The DataFrame containing the stages column.
        buffer (int): The number of consecutive zeros to include as a buffer.
        stages_col (str): The name of the stages column in the DataFrame.

    Returns:
        list: A list of DataFrames, each representing a section of the original DataFrame.
    """
    sections = []
    indexes = []
    in_section = False
    start_index = None

    for i in range(len(df)):
        if df[STAGES].iloc[i] != 0 and not in_section:
            # Start of a new section
            in_section = True
            start_index = max(0, i - buffer)  # Add buffer before the section
        elif df[STAGES].iloc[i] == 0 and in_section:
            # End of a section
            in_section = False
            end_index = min(len(df), i + buffer)  # Add buffer after the section
            sections.append(df.iloc[start_index:end_index])
            indexes.append(end_index)

    return sections, indexes


# save_dir = "./analysed/"

# for item in os.listdir(save_dir):
#     item_path = os.path.join(save_dir, item)
#     if os.path.isfile(item_path) or os.path.islink(item_path):
#         os.remove(item_path)
#     elif os.path.isdir(item_path):
#         shutil.rmtree(item_path)

# datastreams = "./datastreams"
# folders = os.listdir(datastreams)
# dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]
# FREQUENCY = 10
# data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)
# data_shape = (data_size_from_mins(10),)
# increment = 5

# for folder in folders:
#     path = f"{datastreams}/{folder}"
#     curr_data = []
#     journey = 0
#     data_file = 0
#     while data_file < len(os.listdir(path)):
#         data_path = f"{path}/{data_file}.dat"
#         prev_data_file = data_file
#         data = np.memmap(data_path, dtype=dtype, mode='r', shape=data_shape)
#         start = 0
#         end = increment

#         while end <= data_shape[0]:
#             if data_file != prev_data_file:
#                 prev_data_file = data_file
#                 data_path = f"{path}/{data_file}.dat"
#                 data = np.memmap(data_file, dtype=dtype, mode='r', shape=data_shape)

#             curr_data.extend(data[start:end])
#             df = pd.DataFrame(np.array(curr_data, dtype=dtype), columns=['timestamp', 'acceleration'])
#             df['raw_acceleration'] = df['acceleration']
#             df['timestamp'] -= df['timestamp'].iloc[0]

#             # var_a = (acc_smooth * var_raw_acc) / (2 + acc_smooth)
#             smooth(df, 'acceleration', ACCELERATION_SMOOTHING)
#             drift_correction(df, bound=0.02)

#             add_stages(df, LOWER_ACC_THRESHOLD, UPPER_ACC_THRESHOLD, 'acceleration', 'a')

#             first_one_index = df[df['aStages'] == 1].index.min()
#             if df['aStages'].iloc[-1] != 1 or not pd.notna(first_one_index):
#                 end += increment
#                 start += increment
#                 continue

#             df_slice = df.iloc[first_one_index + 1:]
#             zeros_after = df_slice[df_slice['aStages'] == 0]
#             if not zeros_after.empty:
#                 zero_after_one_index = zeros_after.index.min()

#             if zeros_after.empty or not pd.notna(zero_after_one_index):
#                 end += increment
#                 start += increment
#                 continue
            
#             print("Journey found!")
#             # Found the journey
#             half_stationary = (len(df) - zero_after_one_index) // 2

#             df = df[:-half_stationary]
#             if half_stationary > end:
#                 data_file -= 1
            
#             end = (end - half_stationary) % data_shape[0]
#             start = (start - half_stationary) % data_shape[0]

#             curr_data.clear()
#             start = end
#             end += min(increment, data_shape[0] - end)

#             mean, acc_var = get_dist(df, 'raw_acceleration', 0, df[df['aStages'] == 1].index[0])

#             differentiate(df, 'acceleration', 'timestamp', 'jerk')

#             #velocity variance = Sum 0...n: ((dt ** 2) / 2) * acc_var
#             integrate(df, 'acceleration', 'timestamp', 'model_velocity')

#             # Velocity variance resets to 0 at each stationary location
#             correct_vel_at_stationary(df, 'model_velocity', 0.0005)

#             # Height variance = Sum 0...n: ((dt ** 2) / 2) * velocity variance
#             integrate(df, 'model_velocity', 'timestamp', 'model_height')
            
#             # get_true_values(df, acc_var, 0)
                
#             save_df(df, save_dir + folder, journey)
#             journey += 1
        
#         data_file += 1



def main():

    datastreams = "./datastreams"
    folder = os.listdir(datastreams)[2]
    dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]
    data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)
    data_shape = (data_size_from_mins(10),)

    data_path = f"{datastreams}/{folder}/0.dat"
    data = np.memmap(data_path, dtype=dtype, mode='r', shape=data_shape)[:1500]

    df = pd.DataFrame(data, columns=['timestamp', 'acceleration', 'pressure'])

    # Correct timestamps
    df = df[df[TIMESTAMPS] != 0]
    df.reset_index(drop=True, inplace=True)
    df[TIMESTAMPS] -= df[TIMESTAMPS].iloc[0]
    df[TIMESTAMPS] /= 1000

    df.loc[0:len(df), [MODEL_DISPLACEMENT]] = 0
    df.loc[0:len(df), [MODEL_VELOCITY]] = 0
    df.loc[0:len(df), [BAROMETER_HEIGHT]] = 0
    df.loc[0:len(df), [TRUE_DISPLACEMENT]] = 0
    df.loc[0:len(df), [TRUE_VELOCITY]] = 0

    
    # Get initial stages
    df[ACCELERATION] -= df[ACCELERATION].mode().mean()
    get_stages(df, maxlen=15, mean_threshold=0.02, variance_threshold=0.00005, buffer_seconds=5)

    first_journey = df[df[STAGES] == 1].index[0]
    _, acc_var = get_dist(df, ACCELERATION, 0, first_journey)

    df[STAGES] /= 8
    chart(df, ACCELERATION, STAGES)
    df[STAGES] *= 8

    journeys, journey_end_index = split_dataframe_by_stages(df, 0)
    for journey in journeys:
        journey.reset_index(drop=True, inplace=True)

        analyse_journey(journey)
        
        df.update(journey.set_index(TIMESTAMPS))

    print("Based on Barometer:")
    delta_heights = []
    for i in journey_end_index:
        delta_heights.append(df[BAROMETER_HEIGHT].iloc[i - 1])

    floor_journey = get_floor_journey(delta_heights)
    print(' -> '.join([str(i) for i in floor_journey]))

    print("Based on Accelerometer:")
    delta_heights = []
    for i in journey_end_index:
        delta_heights.append(df[MODEL_DISPLACEMENT].iloc[i - 1])

    floor_journey = get_floor_journey(delta_heights)
    print(' -> '.join([str(i) for i in floor_journey]))

    print("Based on sensor fusion of Barometer and Accelerometer:")
    delta_heights = []
    for i in journey_end_index:
        delta_heights.append(df[TRUE_DISPLACEMENT].iloc[i - 1])

    floor_journey = get_floor_journey(delta_heights)
    print(' -> '.join([str(i) for i in floor_journey]))

    # for i in journey_end_index[::-1]:
    #     df.loc[i:, [MODEL_DISPLACEMENT]] += df[MODEL_DISPLACEMENT].iloc[i - 1]
    #     df.loc[i:, [BAROMETER_HEIGHT]] += df[BAROMETER_HEIGHT].iloc[i - 1]

    chart(df, MODEL_DISPLACEMENT)
    chart(df, BAROMETER_HEIGHT)
    chart(df, TRUE_DISPLACEMENT)
            
main()
