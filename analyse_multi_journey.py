
import pandas as pd
import numpy as np
from collections import deque

from globals import *
from utils import *
from analyse_journey import analyse_journey

from sensor_fusion.height import calculate_height

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
    Computes the journey of floors based on changes in height.

    The function tracks the floors visited based on a list of height changes (`delta_heights`),
    starting from an assumed ground level (floor 0). It computes the sequence of floors the 
    journey passes through, accounting for both ascending and descending movements. 

    For example, if the `delta_heights` list is [1, 1, 13, -15], the output would be a journey 
    from floor 0 -> 1 -> 2 -> 15 -> 0.

    @params:
    delta_heights (list of float): A list of height changes between floors. Positive values 
                                    represent an increase in height (going up), while negative 
                                    values represent a decrease in height (going down).

    @returns:
    journey (list of int): A list representing the sequence of floors visited during the journey, 
                            starting from floor 0.

    @example:
    get_floor_journey([1, 1, 13, -15])  # Output: [0, 1, 2, 15, 0]
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


def get_stages_from_accelerometer(df, maxlen=20, mean_threshold=0.05, variance_threshold=0.0001, buffer_seconds=5):
    """
    Updates the 'STAGES' column in the DataFrame based on the rolling mean and variance of acceleration
    (ACCELERATION) to classify whether the lift is stationary or moving.

    The function identifies two stages:
    - Stage 0: The lift is stationary (i.e., little to no movement).
    - Stage 1: The lift is in motion (i.e., acceleration exceeds certain thresholds).

    It uses rolling statistics (mean and variance) to detect significant changes in acceleration. Additionally,
    it ensures that the lift stays stationary for a minimum time (defined by `buffer_seconds`) before transitioning
    from Stage 1 (moving) to Stage 0 (stationary).

    Args:
        df (pd.DataFrame): The DataFrame containing the following columns:
            - `ACCELERATION`: The acceleration data used to determine motion.
            - `TIMESTAMPS`: The timestamps corresponding to each acceleration reading.
        maxlen (int): The window size for calculating the rolling mean and variance. Larger values consider more 
                      data points for smoothing, making the detection of motion more stable but less responsive.
        mean_threshold (float): The threshold for the rolling mean of acceleration. When the absolute rolling mean 
                                exceeds this value, the lift is considered in motion.
        variance_threshold (float): The threshold for the variance of acceleration. If the variance exceeds this 
                                     value, it indicates a significant fluctuation, suggesting motion.
        buffer_seconds (float): The minimum time (in seconds) of inactivity required for transitioning from Stage 1 
                                 (moving) to Stage 0 (stationary). This helps avoid frequent stage toggling due to small
                                 fluctuations in acceleration.

    Returns:
        pd.DataFrame: The original DataFrame with an additional column `STAGES` that contains the stage (0 or 1) for 
                      each timestamp, indicating whether the lift is stationary or moving.

    Example:
        df = get_stages(df, maxlen=30, mean_threshold=0.1, variance_threshold=0.0005, buffer_seconds=10)
        # Updates `df` with the `STAGES` column based on acceleration data.
    """

    assert ACCELERATION in df.columns, f"{ACCELERATION} must be contained in the DataFrame"
    assert TIMESTAMPS in df.columns, f"{TIMESTAMPS} must be contained in the DataFrame"

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

        elif curr_stage == 1:

            if abs(rolling_mean) <= mean_threshold and rolling_variance <= variance_threshold:

                time_since_last_activity = df[TIMESTAMPS].iloc[i] - last_activity
                if time_since_last_activity > buffer_seconds:
                    curr_stage = 0
                    df.loc[max(i - maxlen, 0):, STAGES] = curr_stage
            else:
                last_activity = df[TIMESTAMPS].iloc[i]  # Update the last activity timestamp
        
        df.loc[i, STAGES] = curr_stage

    return df

def get_stages_from_barometer(df, maxlen=20, variance_threshold=0.0001, buffer_seconds=2):

    """
    Updates the 'STAGES' column in the DataFrame based on the rolling variance of barometer height (BAROMETER_HEIGHT).
    The function identifies two stages:
    - Stage 0: Lift is stationary (no significant change in pressure).
    - Stage 1: Lift is in motion (significant fluctuation in pressure).

    The function uses a rolling window to calculate the variance of the barometer height and detects changes 
    in the variance above a defined threshold. It also ensures that the system remains stationary for a specified 
    period (`buffer_seconds`) before transitioning from Stage 1 to Stage 0.

    Args:
        df (pd.DataFrame): The DataFrame containing the following columns:
            - `BAROMETER_HEIGHT`: The barometer height data (pressure or altitude).
            - `TIMESTAMPS`: The timestamps corresponding to each barometer height reading.
        maxlen (int): The size of the rolling window used to calculate the variance. A larger value smooths the variance 
                      over a longer time period, making the detection less sensitive to short-term fluctuations.
        variance_threshold (float): The threshold for the rolling variance. When the variance exceeds this value, 
                                     the lift is considered in motion (Stage 1).
        buffer_seconds (float): The minimum time (in seconds) of inactivity required to transition from Stage 1 
                                 (moving) to Stage 0 (stationary). This helps avoid frequent transitions due to 
                                 minor fluctuations in the barometer height.

    Returns:
        pd.DataFrame: The original DataFrame with an added `STAGES` column, where each entry is either 0 (stationary) 
                      or 1 (moving), representing the state of the lift based on the barometer height changes.

    Example:
        df = get_stages_from_barometer(df, maxlen=30, variance_threshold=0.0005, buffer_seconds=5)
        # Updates `df` with the `STAGES` column based on the variance of the barometer height.
    """

    assert BAROMETER_HEIGHT in df.columns, f"{BAROMETER_HEIGHT} must be contained in the DataFrame"
    assert TIMESTAMPS in df.columns, f"{TIMESTAMPS} must be contained in the DataFrame"

    dt = df[TIMESTAMPS].iloc[1] - df[TIMESTAMPS].iloc[0]
    dt_count = buffer_seconds // dt

    # Initialize rolling window
    rolling_window = deque(maxlen=maxlen)
    df[STAGES] = 0  # Initialize STAGES column

    # Initialize stage and time tracking
    curr_stage = 0

    for i in range(len(df)):
        # Add current acceleration to the rolling window
        rolling_window.append(df[BAROMETER_HEIGHT].iloc[i])
        
        if len(rolling_window) < maxlen:
            continue  # Skip until the rolling window is full
        
        # Calculate rolling mean and variance
        rolling_variance = np.var(rolling_window)
        condition = rolling_variance >= variance_threshold
        
        # Handle transitions
        if curr_stage == 0:
            if condition:
                curr_stage = 1
                df.loc[max(i - int(dt_count / 2), 0):, STAGES] = curr_stage
                last_activity = df[TIMESTAMPS].iloc[i]  # Update the last activity timestamp

        elif curr_stage == 1:

            if not condition:

                time_since_last_activity = df[TIMESTAMPS].iloc[i] - last_activity
                if time_since_last_activity > buffer_seconds / 2:
                    curr_stage = 0
                    df.loc[max(i - maxlen, 0):, STAGES] = curr_stage
            else:
                last_activity = df[TIMESTAMPS].iloc[i]  # Update the last activity timestamp
        
        df.loc[i, STAGES] = curr_stage

    return df

def priority_or(df, col1, col2, new_col):
    """
    Perform a priority-based logical OR operation on two binary columns of a DataFrame.

    This function combines the values of `col1` and `col2` into a new column (`new_col`) 
    based on specific priority rules:
      - `col1` has higher priority and immediately sets `new_col` to `1` for that row and 
        subsequent rows until neither column contains a `1`.
      - A `1` in `col2` is included in `new_col` **only if** it is followed by a `1` 
        in `col1` within the subsequent rows.

    The function iterates row by row, tracking the state (`in_stage`) to handle sequences of `1`s
    in `col1` or `col2`.

    Args:
        df (pd.DataFrame): The input DataFrame containing the binary columns.
        col1 (str): The name of the first column, representing high-priority events.
        col2 (str): The name of the second column, representing low-priority events.
        new_col (str): The name of the new column to store the result.

    Returns:
        pd.DataFrame: The updated DataFrame with the `new_col` column added.

    Notes:
        - Assumes `col1` and `col2` contain only binary values (0 or 1).
        - If `col1` and `col2` contain invalid values or the DataFrame is improperly structured, 
          unexpected behavior may occur.
    """

    in_stage = False
    df.loc[0:len(df), [new_col]] = 0

    i = 0
    while i < len(df):

        if in_stage:
            if df[col1].iloc[i] == 1 or df[col2].iloc[i] == 1:
                df.at[i, new_col] = 1
            else:
                in_stage = False
        
        elif df[col1].iloc[i] == 1:
            in_stage = True
            df.at[i, new_col] = 1
        
        elif df[col2].iloc[i] == 1:

            j = i
            while j < len(df) - 1 and df[col1].iloc[j] != 1 and df[col2].iloc[j] == 1:
                j += 1
            
            if df[col1].iloc[j] == 1:
                in_stage = True
                df.at[i, new_col] = 1
        
        i += 1
    
    return df




def split_dataframe_by_stages(df, buffer):
    """
    Splits a DataFrame into sections based on changes in the 'STAGES' column.
    Sections are defined as continuous periods where the 'STAGES' column has values different from 0.
    Each section is separated by zeros in the 'STAGES' column, and a buffer of zeros is added before and after each section.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'STAGES' column, which is used to identify boundaries between sections.
        buffer (int): The number of consecutive zero entries to include as a buffer before and after each section.

    Returns:
        tuple: A tuple containing:
            - A list of DataFrames, each representing a section of the original DataFrame.
            - A list of integers representing the index at the end of each section in the original DataFrame.
            
    Example:
        df = pd.DataFrame({'STAGES': [0, 0, 1, 1, 0, 1, 1, 0, 0]})
        sections, indexes = split_dataframe_by_stages(df, buffer=1)
        # sections will contain sub-DataFrames where the 'STAGES' column is non-zero.
        # indexes will contain the indices marking the end of each section in the original DataFrame.
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

def get_data():
    """
    Loads the sensor data from a specified folder within the './datastreams' directory.

    This function prompts the user to input the name of the folder containing the sensor data,
    reads the data from a `.dat` file, and returns the data as a numpy memmap array. The data is 
    expected to contain three fields: timestamp, acceleration, and pressure.

    Args:
        None: This function prompts the user to input a folder name containing the datastream to be loaded.

    Returns:
        np.memmap: A memory-mapped numpy array containing the sensor data with the following fields:
                   'timestamp' (int32), 'acceleration' (float32), 'pressure' (float32).
    
    Example:
        data = get_data()
    """

    datastreams = "./datastreams"
    folder = input("Enter folder name to analyse: ")
    dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]
    data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)
    data_shape = (data_size_from_mins(15),)

    data_path = f"{datastreams}/{folder}/0.dat"
    data = np.memmap(data_path, dtype=dtype, mode='r', shape=data_shape)

    return data

def initilise_columns(df):
    """
    Initializes the required columns in the provided DataFrame with default values.

    This function creates and initializes the following columns in the input DataFrame with default values:
    - MODEL_DISPLACEMENT
    - MODEL_VELOCITY
    - BAROMETER_HEIGHT
    - TRUE_DISPLACEMENT
    - TRUE_VELOCITY

    All columns are initialized to zero.

    Args:
        df (pd.DataFrame): The DataFrame in which the columns will be initialized.

    Returns:
        None: The function modifies the DataFrame in place, so no value is returned.
    
    Example:
        initilise_columns(df)
    """

    df.loc[0:len(df), [MODEL_DISPLACEMENT]] = 0
    df.loc[0:len(df), [MODEL_VELOCITY]] = 0
    df.loc[0:len(df), [BAROMETER_HEIGHT]] = 0
    df.loc[0:len(df), [TRUE_DISPLACEMENT]] = 0
    df.loc[0:len(df), [TRUE_VELOCITY]] = 0


def adjust_timestamps(df):
    """
    Adjusts the timestamps in the provided DataFrame by removing zeros and normalizing the values.

    This function performs two main tasks on the DataFrame:
    1. Removes rows with zero timestamps.
    2. Normalizes the timestamps by subtracting the first timestamp and converting the result to seconds.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'TIMESTAMPS' column to be adjusted.

    Returns:
        df (pd.DataFrame): The DataFrame.
    
    Example:
        adjust_timestamps(df)
    """

    df = df[df[TIMESTAMPS] != 0]
    df.reset_index(drop=True, inplace=True)
    df.loc[0:len(df), [TIMESTAMPS]] -= df[TIMESTAMPS].iloc[0]
    df.loc[0:len(df), [TIMESTAMPS]] /= 1000
    return df


def main():
    """
    Main function for processing and analyzing sensor data from a specific folder within the datastreams directory.

    The function:
    1. Prompts the user for a folder name to load sensor data from the datastreams directory.
    2. Initializes the DataFrame with sensor data (timestamp, acceleration, pressure).
    3. Initializes necessary columns and adjusts timestamps.
    4. Applies sensor fusion algorithms (e.g., using the barometer and accelerometer).
    5. Extracts and visualizes sensor data using charts.
    6. Identifies and analyzes distinct journeys (sections of data corresponding to stages of lift movement).
    7. Computes floor journeys based on different sensor readings (barometer, accelerometer, and sensor fusion).
    8. Displays floor journey results based on the calculated heights from different sources.

    The following steps are performed in the function:
    - Loading the sensor data with `get_data()`.
    - Initializing required columns in the DataFrame (e.g., displacement, velocity).
    - Adjusting timestamps for the sensor data.
    - Using the barometer and accelerometer to compute derived measurements such as height and displacement.
    - Correcting data drift and smoothing using Kalman filtering.
    - Identifying stages of movement based on the barometer height and accelerometer data.
    - Dividing the data into individual journeys based on stage transitions and analyzing each journey's characteristics.
    - Computing the maximum or minimum displacement for each journey depending on the direction of movement.
    - Displaying the computed floor journey results in a readable format for analysis.
    - Visualizing various sensor measurements and results through charts.

    Args:
        None: The function prompts the user for input and performs all necessary processing steps within the function.

    Returns:
        None: The function does not return any value; it directly modifies the data and displays results through prints and charts.
    
    Example:
        main()  # Call the function to process and analyze the data from the specified folder
    """

    data = get_data()

    df = pd.DataFrame(data, columns=['timestamp', 'acceleration', 'pressure'])

    initilise_columns(df)

    df = adjust_timestamps(df)

    
    df[BAROMETER_HEIGHT] = df[PRESSURE].apply(calculate_height)
    drift_correction(df, BAROMETER_HEIGHT, BAROMETER_DRIFT_CORRECTION_BOUND)
    smooth_kalman(df, BAROMETER_HEIGHT, BAROMETER_VARIANCE)

    get_stages_from_barometer(df, 20, 0.1, 8)
    df['barometer_stages'] = df[STAGES]

    df[ACCELERATION] -= df[ACCELERATION].round(3).mode().mean()
    get_stages_from_accelerometer(df, maxlen=15, mean_threshold=0.02, variance_threshold=0.0001, buffer_seconds=3)
    df['accelerometer_stages'] = df[STAGES]

    df = priority_or(df, 'barometer_stages', 'accelerometer_stages', STAGES)

    df[STAGES] *= 15
    chart(df, BAROMETER_HEIGHT, STAGES)
    df.loc[0:len(df), [BAROMETER_HEIGHT]] = 0
    df[STAGES] /= 15

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

    print("Based on Accelerometer:")
    floor_journey = get_floor_journey(delta_heights_model_displacement)
    print(' -> '.join([str(i) for i in floor_journey]))

    print("Based on Barometer:")
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

if __name__ == "__main__":
    main()