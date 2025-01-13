import numpy as np

# Touch at your own risk

class KalmanFilter:
    def __init__(
            self,
            initial_height=0,
            initial_velocity=0,
            accelerometer_variance=0.2,
            barometer_variance=0.2
        ):
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
        Predict next state using direct height and velocity measurements
        
        Args:
            height (float): Measured height
            velocity (float): Measured velocity
            dt (float): Time step in seconds
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
        return self.x[0, 0], self.x[1, 0]  # Height and velocity

class SingleValueKalmanFilter:
    def __init__(self, initial=0, variance=0.1):
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
        Update the filter with a new value measurement
        
        Args:
            measured_value (float): Raw value measurement
            dt (float): Time step in seconds
        
        Returns:
            float: Filtered value value
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
