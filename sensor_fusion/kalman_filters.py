import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_height=0, initial_velocity=0):
        # Initial state (height, velocity)
        self.x = np.array([[initial_height], [initial_velocity]])  # state: [height, velocity]
        
        # Initial state covariance matrix
        self.P = np.eye(2) * 1000  # High uncertainty at the start
        
        # Process noise covariance (Q) (uncertainty in motion model)
        self.Q = np.array([[0.1, 0], 
                           [0, 0.1]])  # Assume some process noise (tuning parameter)
        
        # Measurement noise covariance (R) (uncertainty in corrected measurements)
        self.R = np.array([[1, 0],  # Uncertainty in the height measurement (barometer)
                           [0, 1]])  # Uncertainty in the velocity measurement (from your correction)
        
        # Barometer measurement noise (this can be adjusted based on your sensor specs)
        self.R_barometer = np.array([[5]])  # Barometer noise in height measurement
    
    def predict(self, acceleration, dt):
        # State transition matrix (A), where dt is the variable time step
        A = np.array([[1, dt],  # height = height + velocity * dt
                      [0, 1]])  # velocity = velocity (no change in dt)
        
        # Control matrix (B) (acceleration affects the velocity)
        B = np.array([[0.5 * dt**2],  # displacement change from acceleration
                      [dt]])         # velocity change from acceleration
        
        # Predict the next state (height and velocity) using accelerometer data
        self.x = np.dot(A, self.x) + np.dot(B, np.array([[acceleration]]))
        
        # Predict the next state covariance
        self.P = np.dot(np.dot(A, self.P), A.T) + self.Q
    
    def update(self, corrected_height, corrected_velocity):
        # Measurement update using the corrected height and velocity
        Z = np.array([[corrected_height], [corrected_velocity]])  # Corrected measurements
        Y = Z - self.x  # Innovation (measurement residual)
        
        # Compute the Kalman Gain
        S = self.P + self.R  # Innovation covariance
        K = np.dot(self.P, np.linalg.inv(S))  # Kalman gain
        
        # Update the estimate with the corrected measurements
        self.x = self.x + np.dot(K, Y)
        
        # Update the error covariance
        self.P = self.P - np.dot(K, self.P)

    def update_with_barometer(self, barometer_height):
        # Measurement update using the barometer height measurement
        Z = np.array([[barometer_height]])  # Barometer measurement (height)
        Y = Z - self.x[0, 0]  # Innovation (measurement residual)
        
        # Compute the Kalman Gain
        S = self.P[0, 0] + self.R_barometer  # Innovation covariance (only height component)
        K = self.P[0, 0] / S  # Kalman gain (for height update)
        
        # Update the height estimate with the barometer measurement
        self.x[0, 0] = self.x[0, 0] + K * Y
        
        # Update the error covariance (only for the height part)
        self.P[0, 0] = self.P[0, 0] - K * self.P[0, 0]

    def get_estimate(self):
        return self.x[0, 0], self.x[1, 0]  # Height and velocity


# Simulated Data (for illustration)
time_steps = 100  # Number of time steps (e.g., seconds)
time_intervals = np.random.uniform(0.5, 2.0, time_steps)  # Random intervals between 0.5 to 2.0 seconds
timestamps = np.cumsum(time_intervals)  # Cumulative timestamps

# Corrected height and velocity (these could come from your other correction process)
corrected_heights = np.linspace(0, 100, time_steps) + np.random.normal(0, 2, time_steps)  # Noisy corrected height data
corrected_velocities = np.random.normal(5, 0.5, time_steps)  # Corrected velocity data

# Barometer measurements (also noisy)
barometer_heights = np.linspace(0, 100, time_steps) + np.random.normal(0, 5, time_steps)  # Noisy barometer data

# Initialize Kalman filter
kf = KalmanFilter(initial_height=0, initial_velocity=0)

# Lists to store the results for plotting
estimated_heights = []
estimated_velocities = []

# Simulating the process of correcting the velocity and height based on lift behavior:
for t in range(time_steps):
    # Get the current time interval (dt)
    dt = time_intervals[t]
    
    # Here, you would apply your method for detecting when the lift is level (e.g., using accelerometer)
    # and then apply linear corrections to the height and velocity.
    if corrected_velocities[t] < 0.5:  # Assume the lift is level if velocity is nearly zero (a simple heuristic)
        # Apply linear correction for height and velocity to match expected lift pattern
        corrected_heights[t] = 10 * np.sin(t / 10)  # Example pattern correction, you can modify this.
        corrected_velocities[t] = 0  # Assume velocity should be near zero when lift is level
    
    # Predict the next state based on the corrected velocity and height
    kf.predict(dt)
    
    # Update the Kalman filter with the corrected height and velocity measurements
    kf.update(corrected_heights[t], corrected_velocities[t])
    
    # Update the Kalman filter with the barometer height measurement
    kf.update_with_barometer(barometer_heights[t])
    
    # Store the results for plotting
    height, velocity = kf.get_estimate()
    estimated_heights.append(height)
    estimated_velocities.append(velocity)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot estimated vs corrected height
plt.subplot(2, 1, 1)
plt.plot(corrected_heights, label='Corrected Height', color='blue', linestyle='--')
plt.plot(estimated_heights, label='Estimated Height (Kalman Filter)', color='red')
plt.xlabel('Time (seconds)')
plt.ylabel('Height (meters)')
plt.legend()
plt.title('Height Estimation with Corrected Data, Barometer, and Kalman Filter')

# Plot estimated velocity
plt.subplot(2, 1, 2)
plt.plot(estimated_velocities, label='Estimated Velocity', color='green')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.title('Estimated Velocity')

plt.tight_layout()
plt.show()
