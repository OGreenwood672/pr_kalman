
import pandas as pd
import numpy as np
import os
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
    """
    Gets the journey in floors ie  1 -> 2 -> 15 -> 0

    @params
    delta_heights: List of change in heights
    """
    
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


def get_stages(df, maxlen=20, mean_threshold=0.05, variance_threshold=0.0001, buffer_seconds=5):
    """
    Updates the STAGES column based on changes in rolling mean and variance of ACCELERATION.
    0: Lift is stationary
    1: Lift is not stationary :)

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

    df[STAGES] /= 8
    chart(df, ACCELERATION, STAGES)
    df[STAGES] *= 8
    
    delta_heights_model_displacement = []
    delta_heights_barometer_height = []
    delta_heights_true_displacement = []

    journeys, journey_end_index = split_dataframe_by_stages(df, 0)
    for journey in journeys:
        journey.reset_index(drop=True, inplace=True)

        analyse_journey(journey)

        if journey[MODEL_DISPLACEMENT].iloc[-1] > 0:
            delta_heights_model_displacement.append(journey[MODEL_DISPLACEMENT].max())
            delta_heights_barometer_height.append(journey[BAROMETER_HEIGHT].max())
            delta_heights_true_displacement.append(journey[TRUE_DISPLACEMENT].max())
        else:
            delta_heights_model_displacement.append(journey[MODEL_DISPLACEMENT].min())
            delta_heights_barometer_height.append(journey[BAROMETER_HEIGHT].min())
            delta_heights_true_displacement.append(journey[TRUE_DISPLACEMENT].min())
        
        df.update(journey.set_index(TIMESTAMPS))

    print("Based on Barometer:")
    floor_journey = get_floor_journey(delta_heights_model_displacement)
    print(' -> '.join([str(i) for i in floor_journey]))

    print("Based on Accelerometer:")
    floor_journey = get_floor_journey(delta_heights_barometer_height)
    print(' -> '.join([str(i) for i in floor_journey]))

    print("Based on sensor fusion of Barometer and Accelerometer:")
    floor_journey = get_floor_journey(delta_heights_true_displacement)
    print(' -> '.join([str(i) for i in floor_journey]))

    # for i in journey_end_index[::-1]:
    #     df.loc[i:, [MODEL_DISPLACEMENT]] += df[MODEL_DISPLACEMENT].iloc[i - 1]
    #     df.loc[i:, [BAROMETER_HEIGHT]] += df[BAROMETER_HEIGHT].iloc[i - 1]

    chart(df, MODEL_DISPLACEMENT)
    chart(df, BAROMETER_HEIGHT)
    chart(df, TRUE_DISPLACEMENT)
            
main()
