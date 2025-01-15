from sensor_fusion.kalman_filters import SingleValueKalmanFilter

# import sys
# from PyQt6 import QtWidgets
# QtWidgets.QApplication(sys.argv)

import matplotlib.pyplot as plt

from globals import *

def sign(v):
    return 1 if v > 0 else -1

def magnitude(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

def drift_correction(df, col, bound):
    """
    Removes the offset on the acceleration
    @params
    df: dataframe
    bound: how much the function will accept a lack of significant movement
    """
    rolling_sum = 0
    num_of_pnts = 0
    for acc in df[col]:
        if acc == df[col].iloc[0] or abs(rolling_sum / num_of_pnts - acc) < bound:
            rolling_sum += acc
            num_of_pnts += 1
    df.loc[0:len(df), [col]] -= rolling_sum / num_of_pnts


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


def differentiate(df, u, v, new_name):
    """
    Differentiation of u by v
    @params:
    u, v: du/dv
    new_name: resulting column name
    """
    assert u in df.columns, f"DataFrame must include {u}"
    assert v in df.columns, f"DataFrame must include {v}"
    df.loc[0:len(df), [new_name]] = 0.0

    for i in range(1, len(df)):
        du = df[u].iloc[i] - df[u].iloc[i - 1]
        dv = df[v].iloc[i] - df[v].iloc[i - 1]

        if dv != 0:
            df.at[i, new_name] = du / dv

def integrate(df, u, v, new_name):
    """
    Integration of u by v
    @params:
    df: dataframe
    u, v: ∫ u dv
    new_name: resulting columns name
    """
    assert u in df.columns, f"DataFrame must include {u}"
    assert v in df.columns, f"DataFrame must include {v}"
    df.loc[0:len(df), [new_name]] = 0.0

    for i in range(1, len(df)):
        avg_u = (df[u].iloc[i] + df[u].iloc[i - 1]) / 2
        dv = df[v].iloc[i] - df[v].iloc[i - 1]
        
        df.at[i, new_name] = df[new_name].iloc[i - 1] + avg_u * dv


def smooth_kalman(df, col, variance):
    """
    Smooths column of dataframe using single kalman filter
    @params
    df: Python Pandas DataFrame
    col: Column to be smoothed
    variance: The variance of the noise on the column
    """

    kf = SingleValueKalmanFilter(0, variance)
    smoothed = []
    for i in range(len(df)):
        v = df[col].iloc[i]

        if not i:
            dt = df[TIMESTAMPS].iloc[0]
        else:
            dt = (df[TIMESTAMPS].iloc[i] - df[TIMESTAMPS].iloc[i - 1])

        smoothed.append(kf.update(v, dt))
    
    df.loc[0:len(df), [col]] = smoothed


def chart(df, col1, col2=None):
    """
    Chart function to plot col1 by time
    If col2 is supplied, they are plotted against each other

    @params
    df: Python Pandas DataFrame
    col1: The column on the y-axis
    col2: (Optional) Column to be plotted on the other y-axis
    """

    fig, ax1 = plt.subplots()

    labels = {
        ACCELERATION: 'Acceleration (m/s^2)',
        MODEL_DISPLACEMENT: 'Height (m)',
        MODEL_VELOCITY: 'Velocity (m/s)',
        STAGES: "stages",
        TRUE_DISPLACEMENT: "True Height (m)",
        BAROMETER_HEIGHT: "Barometer Height (m)",
        TRUE_VELOCITY: "True Velocity (m/s)"
    }
    col1_label = labels.get(col1)
    if col1_label == None: col1_label = col1

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(col1_label, color='tab:blue')

    ax1.plot(df['timestamp'], df[col1], color='tab:blue', label=col1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')   
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')

    if col2 != None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(col2, color='tab:red')

        ax2.plot(df['timestamp'], df[col2], color='tab:red', label=col2)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        minimum = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        maximum = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(minimum, maximum)
        ax2.set_ylim(minimum, maximum)

        plt.title(f'{col1_label} and {col2} vs. Time')
    
    else:
        plt.title(f'{col1_label} vs. Time') 

    fig.tight_layout()

    plt.show()