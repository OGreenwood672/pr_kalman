from sensor_fusion.kalman_filters import SingleValueKalmanFilter, KalmanFilter
from sensor_fusion.height import calculate_height
from globals import *
from utils import *

def get_true_values(df, accelerometer_variance, barometer_variance):

    """
    Integrates the kalman filter to predict better values by combining the sensors
    @params
    df: dataframe with data
    accelerometer_variance: The variance of the accelerometer
    barometer_variance: The variance of the barometer
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

    smooth_kalman(journey, ACCELERATION, ACCELEROMETER_VARIANCE)
    journey.loc[0:len(journey), [ACCELERATION]] -= journey[ACCELERATION].mode().mean()

    integrate(journey, ACCELERATION, TIMESTAMPS, MODEL_VELOCITY)

    integrate(journey, MODEL_VELOCITY, TIMESTAMPS, MODEL_DISPLACEMENT)

    journey.loc[0:len(journey), [BAROMETER_HEIGHT]] = journey[PRESSURE].apply(calculate_height)
    drift_correction(journey, BAROMETER_HEIGHT, bound=BAROMETER_DRIFT_CORRECTION_BOUND)
    smooth_kalman(journey, BAROMETER_HEIGHT, BAROMETER_VARIANCE)

    journey.loc[0:len(journey), [BAROMETER_HEIGHT]] /= 2.6

    get_true_values(journey, ACCELEROMETER_VARIANCE, BAROMETER_VARIANCE)

