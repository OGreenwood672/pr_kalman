from sensor_fusion.kalman_filters import SingleValueKalmanFilter

import matplotlib.pyplot as plt

from globals import *

def sign(v):
    """
    Given value v, returns the sign of the value
    @params
    v: value to calculate sign of
    """
    return 1 if v > 0 else -1

def magnitude(v):
    """
    Calculates the magnitude of a 3D vector
    @params
    v: vector to caluclate magnitude for
    """
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]

def drift_correction(df, col, bound):
    """
    Applies drift correction to a specified column in the DataFrame by removing a calculated offset.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str): The column name to apply drift correction to.
        bound (float): Threshold to determine significant movement; values within this range 
                       are considered stationary.

    Modifies:
        Adjusts the specified column (`col`) in-place by subtracting the calculated drift offset.

    Example:
        >>> import pandas as pd
        >>> data = {'ACC': [1.01, 1.02, 1.03, 0.98, 0.99]}
        >>> df = pd.DataFrame(data)
        >>> drift_correction(df, 'ACC', bound=0.05)
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
    Applies a time-dependent linear correction or transformation to the values of a specified column 
    in a DataFrame. The offset controls the total change applied across the specified time range.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be modified.
        label (str): The column name in the DataFrame to apply the transformation to.
        offset (float): The total amount by which the values will be adjusted linearly over the range.
        start (int): The starting index of the range within the DataFrame.
        stop (int): The ending index of the range within the DataFrame.

    Modifies:
        The specified column (`label`) of the DataFrame in-place, applying a linear adjustment 
        between the start and stop indices based on the offset.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'TIMESTAMPS': [0, 1, 2, 3, 4],
        ...     'VALUES': [10, 20, 30, 40, 50]
        ... }
        >>> df = pd.DataFrame(data)
        >>> linear_offset(df, 'VALUES', offset=10, start=1, stop=3)
        >>> print(df)
           TIMESTAMPS  VALUES
        0           0    10.0
        1           1    17.5
        2           2    25.0
        3           3    32.5
        4           4    50.0
    """
    dt = df[TIMESTAMPS].iloc[stop] - df[TIMESTAMPS].iloc[start]
    for i in range(start, stop + 1):
        df.at[i, label] = df[label].iloc[i] - offset * ((df[TIMESTAMPS].iloc[i] - df[TIMESTAMPS].iloc[start]) / dt)



def differentiate(df, u, v, new_name):
    """
    Computes the numerical derivative of column `u` with respect to column `v` in a DataFrame.

    The derivative is calculated as the difference of consecutive values in `u` divided by 
    the difference of consecutive values in `v`. The result is stored in a new column specified 
    by `new_name`.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        u (str): The name of the column to differentiate.
        v (str): The name of the column with respect to which the differentiation is performed.
        new_name (str): The name of the new column to store the result of the differentiation.

    Raises:
        AssertionError: If `u` or `v` are not present in the DataFrame columns.
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
    Computes the numerical integral of column `u` with respect to column `v` in a DataFrame.

    The integral is approximated using the trapezoidal rule, where the average of consecutive 
    values of `u` is multiplied by the change in `v` to estimate the integral at each point.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        u (str): The name of the column to integrate.
        v (str): The name of the column with respect to which the integration is performed.
        new_name (str): The name of the new column to store the result of the integration.

    Raises:
        AssertionError: If `u` or `v` are not present in the DataFrame columns.
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
    Applies a Kalman filter to smooth a specified column of data in the DataFrame.

    The Kalman filter is applied to each value in the specified column to estimate the 
    true value from noisy observations. The filter uses the given `variance` to 
    model the noise in the data. The smoothed results replace the original values in the column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str): The name of the column to apply the Kalman filter to.
        variance (float): The variance of the noise in the data, which affects the filter's smoothing behavior.

    Raises:
        AssertionError: If the specified column `col` is not found in the DataFrame.
    """

    assert col in df.columns, f"DataFrame must include {col}" 

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
    Plots a time-series chart of `col1` versus time. If `col2` is provided, it plots both `col1` 
    and `col2` on dual y-axes against time.

    The function creates a plot using matplotlib, where:
    - `col1` is plotted on the left y-axis against time.
    - If `col2` is provided, it is plotted on the right y-axis.
    - The x-axis represents the time series (from the `timestamp` column).

    The function also handles common labels for certain columns based on a predefined mapping, 
    and it displays the chart with proper labels and axes.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col1 (str): The name of the column to plot on the left y-axis.
        col2 (str, optional): The name of the second column to plot on the right y-axis. Default is None.

    Raises:
        AssertionError: If either `col1` or `col2` (if provided) is not found in the DataFrame.
    """
    
    assert col1 in df.columns, f"DataFrame must include {col1}" 
    if col2:
        assert col2 in df.columns, f"DataFrame must include {col2}" 

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

        col2_label = labels[col2] if labels.get(col2) else col2
        ax2.plot(df['timestamp'], df[col2], color='tab:red', label=col2_label)
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