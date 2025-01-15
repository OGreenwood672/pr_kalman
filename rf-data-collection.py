
import PhidgetUtils
import numpy as np
import os
import pandas as pd
import threading
from sensor_fusion import height
from scipy.signal import find_peaks
import json

from utils import *
from globals import *

from time import sleep

FREQUENCY = 10
STOP_REQUEST = False


def attach_accelerometer():
    acc = PhidgetUtils.PhidgetAccelerometer(data_rate=FREQUENCY)

    while not acc.is_attached():
        sleep(0.1)

    print("Accelerometer is attached!")
    return acc

def attach_barometer():
    bar = PhidgetUtils.PhidgetBarometer(data_rate=FREQUENCY)

    while not bar.is_attached():
        sleep(0.1)

    print("Barometer is attached!")
    return bar

def modify_velocity_profile(df, threshold=0.01):
    """
    Modifies velocity profile by applying a linear offset between humps.
    Works for both positive and negative velocity profiles.
   
    Parameters:
    df (pd.DataFrame): DataFrame with velocity and timestamp columns
    threshold (float): Threshold for detecting velocity changes
   
    Returns:
    pd.DataFrame: Modified DataFrame with adjusted velocity profile
    """
    # Create a copy to avoid modifying original data
    modified_df = df.copy()
    velocities = df[MODEL_VELOCITY].values
    times = df[TIMESTAMPS].values
   
    # Calculate velocity changes to detect significant movements
    vel_changes = np.abs(np.gradient(velocities))
   
    # Find start of first movement (end of initial flat portion)
    first_hump_start_idx = np.where(vel_changes > threshold)[0][0]
    
    # Determine if the profile is primarily positive or negative
    is_positive_profile = np.max(velocities) > abs(np.min(velocities))
    
    # Find peaks or valleys based on profile direction
    if is_positive_profile:
        valleys, _ = find_peaks(-velocities, prominence=0.01)
    else:
        valleys, _ = find_peaks(velocities, prominence=0.01)
        
    if not valleys:
        # Find the start of second hump by looking for the major velocity change
        second_hump_mask = (times > times[first_hump_start_idx + 100])
        second_hump_velocities = velocities[second_hump_mask]
        
        # Adjust gradient direction based on profile
        if is_positive_profile:
            major_changes = np.where(np.gradient(second_hump_velocities) < -threshold)[0]
        else:
            major_changes = np.where(np.gradient(second_hump_velocities) > threshold)[0]
            
        second_hump_start_idx = np.where(second_hump_mask)[0][0] + major_changes[0] if len(major_changes) > 0 else len(times) - 1
    else:
        second_hump_start_idx = valleys[0]
    
    # Calculate offset at the transition point
    offset = velocities[second_hump_start_idx]
   
    # Apply linear offset
    linear_offset(modified_df, MODEL_VELOCITY, offset, first_hump_start_idx, second_hump_start_idx)
    modified_df.loc[second_hump_start_idx + 1:, [MODEL_VELOCITY]] -= offset
   
    return modified_df


def cut_df(df, threshold, buffer, col='value'):
    """
    Remove rows from the start and end of the DataFrame until the value in the specified column
    changes beyond the given threshold, but retain a buffer of additional rows.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        threshold (float): The threshold for determining a significant change.
        buffer (int): The number of additional rows to retain after trimming.
        col (str): The column to check for changes. Default is 'value'.

    Returns:
        pd.DataFrame: The trimmed DataFrame.
    """
    # Ensure the DataFrame is not empty
    if df.empty or col not in df.columns:
        return df

    # Trim from the start
    start_value = df[col].iloc[0]
    start_index = 0
    for i, val in enumerate(df[col]):
        if abs(val - start_value) > threshold:
            start_index = max(0, i - buffer)  # Include buffer
            break

    # Trim from the end
    end_value = df[col].iloc[-1]
    end_index = len(df) - 1
    for i, val in enumerate(reversed(df[col])):
        if abs(val - end_value) > threshold:
            end_index = min(len(df) - 1, len(df) - i + buffer - 1)  # Include buffer
            break

    # Return the sliced DataFrame
    return df.iloc[start_index:end_index + 1].reset_index(drop=True)


def stop_checker():
    global STOP_REQUEST
    input("Press enter to stop:")
    STOP_REQUEST = True

def get_journey_data(acc, bar):
    global STOP_REQUEST

    df = pd.DataFrame({"timestamp": [], "acceleration": [], "pressure": []})

    thread = threading.Thread(target=stop_checker, daemon=True)
    thread.start()

    while not STOP_REQUEST:

        row = pd.DataFrame({
            "timestamp": [acc.getTimestamp()],
            "acceleration": [magnitude(acc.getAcceleration()) * sign(acc.getAcceleration()[2])],
            "pressure": [bar.getPressure()]
        })
        df = pd.concat([df, row], ignore_index=True)
        
        sleep(0.17)
    
    STOP_REQUEST = False
    return df


def get_features(df):
        
    dx = df[MODEL_DISPLACEMENT].iloc[len(df) - 1]

    dbx = df[BAROMETER_HEIGHT].tail(15).mean()

    # Maximum absolute value from MODEL_VELOCITY column
    max_v = df[MODEL_VELOCITY].abs().max()
    
    # Determine if dv is positive by checking original value at max absolute position
    max_vel_idx = df[MODEL_VELOCITY].abs().idxmax()
    is_dv_positive = df.loc[max_vel_idx, MODEL_VELOCITY] > 0
    
    # Get da1 and da2 based on velocity direction
    if is_dv_positive:
        da1 = df[ACCELERATION].max()
        da2 = df[ACCELERATION].min()
    else:
        da1 = df[ACCELERATION].min()
        da2 = df[ACCELERATION].max()
    
    dt = df[TIMESTAMPS].iloc[len(df) - 1]

    return dx, dbx, max_v, da1, da2, dt

def save(lift_id, dx, dbx, max_v, da1, da2, dt, curr_floor, new_floor):

    file_path = f"./rf_data/{lift_id}.json"
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load existing data from the file
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        # Initialize a new list if the file does not exist
        data = []
    
    # Append the new record to the data list
    data.append({
        "dx": dx,
        "dbx": dbx,
        "max_v": max_v,
        "da1": da1,
        "da2": da2,
        "dt": dt,
        "curr_floor": curr_floor,
        "new_floor": new_floor
    })
    
    # Save the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():

    # acc = attach_accelerometer()
    # bar = attach_barometer()

    # lift_id = input("Enter lift id: ")

    # rf_model = joblib.load('random_forest_model.pkl')

    while True:
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n********************")

        datastreams = "./datastreams"
        dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]
        data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)
        data_shape = (data_size_from_mins(10),)

        data_path = f"{datastreams}/{os.listdir(datastreams)[0]}/0.dat"
        data = np.memmap(data_path, dtype=dtype, mode='r', shape=data_shape)[:350]
        df = pd.DataFrame(data, columns=['timestamp', 'acceleration', 'pressure'])

        # df = pd.read_csv('./error.csv')


        # curr_floor = int(input("Enter the current floor: "))
        # input("Enter to start: ")

        # df = get_journey_data(acc, bar)


        df = cut_df(df, 0.02, 5, ACCELERATION)

        try:

            df[TIMESTAMPS] -= df[TIMESTAMPS].iloc[0]
            df[TIMESTAMPS] /= 1000

            # drift_correction(df, ACCELERATION, 0.02)
            df[ACCELERATION] -= df[ACCELERATION].mode().mean()
            smooth_kalman(df, ACCELERATION, 0.05)
            integrate(df, ACCELERATION, TIMESTAMPS, MODEL_VELOCITY)

            integrate(df, MODEL_VELOCITY, TIMESTAMPS, MODEL_DISPLACEMENT)

            df[BAROMETER_HEIGHT] = df[PRESSURE].apply(height.calculate_height)
            drift_correction(df, BAROMETER_HEIGHT, 0.5)
            smooth_kalman(df, BAROMETER_HEIGHT, 0.5)

        except:
            chart(df, ACCELERATION)
            df.to_csv('./error.csv')
            print("[ERROR]")
            continue

        chart(df, ACCELERATION)
        chart(df, MODEL_VELOCITY)
        chart(df, MODEL_DISPLACEMENT)
        chart(df, BAROMETER_HEIGHT)

        # features = get_features(df)


        # df_to_predict = pd.DataFrame({
        #     "dx": [features[0]],
        #     "dbx": [features[1]],
        #     "max_v": [features[2]],
        #     "da1": [features[3]],
        #     "da2": [features[4]],
        #     "dt": [features[5]],
        #     "curr_floor": [curr_floor],
        # })
        # predictions = rf_model.predict(df_to_predict)
        # new_floor = int(input(f"What floor are we on now, I think we are on {predictions[0]}: "))

        # save(lift_id, *features, curr_floor, new_floor)
        # input("Enter to continue: ")

if __name__ == "__main__":
    main()