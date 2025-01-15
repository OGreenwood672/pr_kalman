from sensor_fusion.kalman_filters import SingleValueKalmanFilter, KalmanFilter
from sensor_fusion.height import calculate_height
from globals import *
from utils import *

def get_true_values(df, accelerometer_variance, barometer_variance):

    """
    Uses a Kalman filter to combine accelerometer and barometer data to estimate true displacement and velocity.

    The function integrates the Kalman filter to provide a more accurate estimate of the true displacement
    and velocity by fusing data from the accelerometer and barometer sensors. The Kalman filter is updated 
    with each new measurement in the DataFrame, considering the given variances for both sensors.

    Args:
        df (pd.DataFrame): The DataFrame containing sensor data. It must have the following columns:
                           - ACCELERATION: Acceleration data from the accelerometer.
                           - TIMESTAMPS: Timestamps of sensor readings.
                           - BAROMETER_HEIGHT: Barometer height data.
        accelerometer_variance (float): The variance of the accelerometer sensor, used in the Kalman filter.
        barometer_variance (float): The variance of the barometer sensor, used in the Kalman filter.

    Returns:
        None: The function updates the input DataFrame in place, adding the estimated true displacement and velocity 
              in the columns TRUE_DISPLACEMENT and TRUE_VELOCITY.

    Example:
        get_true_values(df, accelerometer_variance=0.01, barometer_variance=0.1)

    Notes:
        The Kalman filter predicts the next state using accelerometer data and then corrects the prediction 
        using the barometer data. The updated true displacement and velocity estimates are stored in the DataFrame.
    """

    kf = KalmanFilter(0, 0, accelerometer_variance, barometer_variance)
    for i in df.index:

        if not i:
            dt = df[TIMESTAMPS].iloc[0]
        else:
            dt = (df[TIMESTAMPS].iloc[i] - df[TIMESTAMPS].iloc[i - 1])

        kf.predict_with_accelerometer(df[ACCELERATION].iloc[i], dt)
        # kf.predict_with_state(df[MODEL_DISPLACEMENT].iloc[i], df[MODEL_VELOCITY].iloc[i], dt)

        # kf.update_with_model(df['model_height'], df['model_velocity'])
        kf.update_with_barometer(df[BAROMETER_HEIGHT].iloc[i])

        height, velocity = kf.get_estimate()
        df.loc[i, TRUE_DISPLACEMENT] = height
        df.loc[i, TRUE_VELOCITY] = velocity


def analyse_journey(journey):

    """
    Analyzes a single journey by applying various data processing steps, including smoothing, integration,
    drift correction, and sensor fusion.

    The function processes the given journey's sensor data by:
    1. Smoothing accelerometer data using a Kalman filter.
    2. Removing the mean from the accelerometer data for normalization.
    3. Integrating acceleration to compute velocity and then integrating velocity to compute displacement.
    4. Applying barometer height calculation, drift correction, and smoothing on the barometer data.
    5. Using a sensor fusion approach to estimate true displacement and velocity.

    Args:
        journey (pd.DataFrame): A DataFrame representing a single journey's sensor data. It must have the following columns:
            - ACCELERATION: Acceleration data from the accelerometer.
            - TIMESTAMPS: Timestamps of sensor readings.
            - PRESSURE: Pressure data from the barometer.
            - MODEL_VELOCITY: The model velocity (calculated from the accelerometer).
            - MODEL_DISPLACEMENT: The model displacement (calculated from the velocity).
            - BAROMETER_HEIGHT: The barometer height, which will be calculated and corrected.
        
    Returns:
        None: The function processes and updates the `journey` DataFrame in place, adding new columns for processed
              data such as smoothed values, velocity, displacement, and corrected barometer height.
    
    Example:
        analyse_journey(journey)

    Notes:
        - The function relies on Kalman filtering to smooth accelerometer and barometer data.
        - Integration is performed on acceleration to calculate velocity and on velocity to calculate displacement.
        - The barometer height is corrected using a drift correction method and normalized.
        - True displacement and velocity estimates are derived by combining the accelerometer and barometer data.
    """

    smooth_kalman(journey, ACCELERATION, ACCELEROMETER_VARIANCE)
    journey.loc[0:len(journey), [ACCELERATION]] -= journey[ACCELERATION].mode().mean()

    integrate(journey, ACCELERATION, TIMESTAMPS, MODEL_VELOCITY)

    linear_offset(journey, MODEL_VELOCITY, journey[MODEL_VELOCITY].iloc[len(journey) - 1], 0, len(journey) - 1)

    integrate(journey, MODEL_VELOCITY, TIMESTAMPS, MODEL_DISPLACEMENT)

    journey.loc[0:len(journey), [BAROMETER_HEIGHT]] = journey[PRESSURE].apply(calculate_height)
    drift_correction(journey, BAROMETER_HEIGHT, bound=BAROMETER_DRIFT_CORRECTION_BOUND)
    smooth_kalman(journey, BAROMETER_HEIGHT, BAROMETER_VARIANCE)

    journey.loc[0:len(journey), [BAROMETER_HEIGHT]] /= 2.6

    get_true_values(journey, ACCELEROMETER_VARIANCE, BAROMETER_VARIANCE)

