import numpy as np

# Touch at your own risk

class KalmanFilter:

    """
    A Kalman filter for estimating the height and velocity of an object based on accelerometer and barometer measurements.
    
    The filter combines the noisy measurements from both sensors (accelerometer and barometer) and updates its state
    (height and velocity) over time. It uses the Kalman filter's prediction and update steps to improve the accuracy of
    the estimates.

    Attributes:
        x (numpy.ndarray): The state vector, consisting of height and velocity.
        P (numpy.ndarray): The covariance matrix, representing uncertainty in the state.
        accelerometer_variance (float): The variance (uncertainty) of the accelerometer measurements.
        barometer_variance (float): The variance (uncertainty) of the barometer measurements.
        Q (numpy.ndarray): The process noise covariance matrix, which varies based on the time step (dt).
        H (numpy.ndarray): The measurement matrix for the barometer, which only measures height.
        R_barometer (numpy.ndarray): The measurement noise covariance matrix for the barometer.

    Methods:
        __init__(initial_height=0, initial_velocity=0, accelerometer_variance=0.2, barometer_variance=0.2):
            Initializes the Kalman filter with given initial height, velocity, and measurement variances.
        
        predict_with_accelerometer(acceleration, dt):
            Predicts the next state (height and velocity) using the accelerometer measurements and time step.
        
        update_with_barometer(barometer_height):
            Updates the filter's state using the barometer's height measurement.
        
        predict_with_state(height, velocity, dt):
            Predicts the next state using direct measurements of height and velocity.
        
        get_estimate():
            Returns the current estimates of height and velocity.
    """

    def __init__(
            self,
            initial_height=0,
            initial_velocity=0,
            accelerometer_variance=0.2,
            barometer_variance=0.2
        ):
        """
        Initializes the Kalman filter with initial state (height and velocity) and measurement variances.

        Args:
            initial_height (float): The initial height estimate (default is 0).
            initial_velocity (float): The initial velocity estimate (default is 0).
            accelerometer_variance (float): The variance (uncertainty) of the accelerometer measurements (default is 0.2).
            barometer_variance (float): The variance (uncertainty) of the barometer measurements (default is 0.2).
        """
        # Initial state (height, velocity)
        self.x = np.array([[initial_height], [initial_velocity]])
        
        # Initial state covariance matrix
        self.P = np.eye(2) * 1000  # High uncertainty at the start
        
        self.accelerometer_variance = accelerometer_variance
        self.barometer_variance = barometer_variance
        
        # Process noise covariance (Q)
        self.Q = np.zeros((2, 2))  # Calculated each timestep due to varied dt
        
        # Measurement matrix for barometer (only measures height)
        self.H = np.array([[1, 0]])
        
        # Barometer measurement noise
        self.R_barometer = np.array([[barometer_variance]])

    def predict_with_accelerometer(self, acceleration, dt):
        """
        Predicts the next state (height and velocity) using the accelerometer measurement and time step.

        Args:
            acceleration (float): The measured acceleration (from accelerometer).
            dt (float): The time step (in seconds) since the last prediction.
        """
        # State transition matrix
        A = np.array([[1, dt],
                     [0, 1]])
        
        # Control matrix
        B = np.array([[0.5 * (dt**2)],
                     [dt]])
        
        # Predict state
        self.x = np.dot(A, self.x) + np.dot(B, np.array([[acceleration]]))
        
        # Update process noise covariance
        self.Q = np.array([[0.25 * self.accelerometer_variance * (dt**4), 0.5 * self.accelerometer_variance * (dt**3)],
                          [0.5 * self.accelerometer_variance * (dt**3), self.accelerometer_variance * (dt**2)]])
        
        # Predict covariance
        self.P = np.dot(np.dot(A, self.P), A.T) + self.Q

    def update_with_barometer(self, barometer_height):
        """
        Updates the Kalman filter's state using the barometer's height measurement.

        Args:
            barometer_height (float): The height measurement from the barometer.
        """
        # Measurement
        z = np.array([[barometer_height]])
        
        # Innovation
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R_barometer
        
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state
        self.x = self.x + np.dot(K, y)
        
        # Update covariance
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
    
    # def update_with_model(self, corrected_height, corrected_velocity):
    #     # Measurement update using the corrected height and velocity
    #     Z = np.array([[corrected_height], [corrected_velocity]])  # Corrected measurements
    #     Y = Z - self.x  # Innovation (measurement residual)
        
    #     # Compute the Kalman Gain
    #     S = self.P + self.R  # Innovation covariance
    #     K = np.dot(self.P, np.linalg.inv(S))  # Kalman gain
        
    #     # Update the estimate with the corrected measurements
    #     self.x = self.x + np.dot(K, Y)
        
    #     # Update the error covariance
    #     self.P = self.P - np.dot(K, self.P)

    def predict_with_state(self, height, velocity, dt):
        """
        Predicts the next state using direct measurements of height and velocity.

        Args:
            height (float): The measured height.
            velocity (float): The measured velocity.
            dt (float): The time step (in seconds) since the last prediction.
        """
        # State transition matrix
        A = np.array([[1, dt],
                     [0, 1]])
        
        # Current measurement
        z = np.array([[height],
                     [velocity]])
        
        # Predict state using state transition
        x_pred = np.dot(A, self.x)
        
        # Update process noise covariance for position and velocity
        pos_variance = 0.5 * self.accelerometer_variance * (dt**4)  # Position uncertainty
        vel_variance = self.accelerometer_variance * (dt**2)        # Velocity uncertainty
        pos_vel_covariance = 0.5 * self.accelerometer_variance * (dt**3)  # Position-velocity covariance
        
        self.Q = np.array([[pos_variance, pos_vel_covariance],
                          [pos_vel_covariance, vel_variance]])
        
        # Predict covariance
        P_pred = np.dot(np.dot(A, self.P), A.T) + self.Q
        
        # Measurement noise covariance
        R = np.array([[self.barometer_variance, 0],
                     [0, self.accelerometer_variance]])  # Assuming velocity has similar noise to acceleration
        
        # Kalman gain
        H = np.eye(2)  # Full state measurement
        S = np.dot(np.dot(H, P_pred), H.T) + R
        K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
        
        # Update state and covariance
        self.x = x_pred + np.dot(K, (z - np.dot(H, x_pred)))
        self.P = np.dot((np.eye(2) - np.dot(K, H)), P_pred)

    def get_estimate(self):
        """
        Returns the current estimates of height and velocity.

        Returns:
            tuple: The current estimates of height and velocity (height, velocity).
        """
        return self.x[0, 0], self.x[1, 0]  # Height and velocity

class SingleValueKalmanFilter:
    """
    A Kalman filter implementation that estimates a value and its rate of change over time using sensor measurements.

    The filter estimates the value based on noisy measurements and updates its estimate over time by taking into account
    the rate of change (velocity) of the value. It uses a state vector that includes both the value and its rate of change
    and applies the Kalman filter's prediction and update steps.

    Attributes:
        x (numpy.ndarray): State vector containing the value and its rate of change (velocity).
        P (numpy.ndarray): Covariance matrix representing the uncertainty in the state vector.
        R (numpy.ndarray): Measurement noise covariance matrix.
        H (numpy.ndarray): Measurement matrix that defines how the state vector relates to the measurements.

    Methods:
        __init__(initial=0, variance=0.1): Initializes the Kalman filter with an initial value and measurement variance.
        update(measured_value, dt): Updates the filter with a new value measurement and the time step since the last update.
    """

    def __init__(self, initial=0, variance=0.1):
        """
        Initializes the Kalman filter with the given initial value and measurement variance.

        Args:
            initial (float): The initial estimate of the value to be tracked. Default is 0.
            variance (float): The variance (uncertainty) in the measurements. Default is 0.1.
        """
        # State vector [value,value_rate_of_change]
        self.x = np.array([[initial],
                          [0.0]])  # assume initial rate of change is 0
        
        # Initial uncertainty covariance matrix
        self.P = np.array([[10.0, 0],
                          [0, 10.0]])  # start with high uncertainty
        
        # Measurement noise (R)
        self.R = np.array([[variance]])
        
        # Measurement matrix (H)
        self.H = np.array([[1.0, 0]])  # we only measure value
        
    def update(self, measured_value, dt):
        """
        Updates the filter's estimate based on a new measurement and time step.

        This method performs the prediction and update steps of the Kalman filter algorithm:
        1. Predict the new state based on the previous state.
        2. Update the state estimate using the new measurement.

        Args:
            measured_value (float): The raw measurement of the value (e.g., from a sensor).
            dt (float): The time step in seconds since the last update (delta time).

        Returns:
            float: The filtered estimate of the value after applying the Kalman filter.
        """
        # State transition matrix
        F = np.array([[1, dt],
                     [0, 1]])
        
        # Process noise covariance matrix
        # Using a simplified continuous white noise model
        q = 0.1  # process noise parameter (can be tuned)
        Q = np.array([[q * dt**3 / 3, q * dt**2 / 2],
                     [q * dt**2 / 2, q * dt]])
        
        # Predict
        self.x = np.dot(F, self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + Q
        
        # Update
        z = np.array([[measured_value]])
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return float(self.x[0])  # return the filtered value
