from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *
from time import sleep

from scipy.constants import g


class PhidgetAccelerometer:
    """
    A wrapper class for interfacing with the Phidget Accelerometer device.
    
    This class allows for the collection of acceleration data from the Phidget accelerometer, with additional
    calculations for velocity and displacement. The accelerometer readings are adjusted for drift and gravity. 
    It also supports calibration of the accelerometer to correct for drift.

    Attributes:
        _acceleration (list): Current acceleration values in the x, y, and z directions (m/s^2).
        _velocity (list): Current velocity values in the x, y, and z directions (m/s).
        _displacement (list): Current displacement values in the x, y, and z directions (m).
        _drift (list): Drift values for the accelerometer to adjust for biases.
        _gravity (list): Gravity vector, used for removing gravitational acceleration (assumes gravity in z-direction).
        _timestamp (int): Timestamp of the most recent acceleration reading.
        _data_rate (int): The data rate (in Hz) at which the accelerometer should collect data.
        _accelerometer (Accelerometer): Phidget Accelerometer object used for data acquisition.
    """

    def __init__(self, data_rate):
        """
        Initializes the PhidgetAccelerometer with a specified data rate.

        Args:
            data_rate (int): The desired data rate for the accelerometer (in Hz).
        """
        self._acceleration = [0, 0, 0]
        self._velocity = [0, 0, 0]
        self._displacement = [0, 0, 0]

        self._drift = [0, 0, 0]
        self._gravity = [0, 0, g]  # Gravity in the z-direction
        self._timestamp = -1
        self._data_rate = data_rate

        self._accelerometer = Accelerometer()

        # Set event handlers
        self._accelerometer.setOnAttachHandler(self.on_attach)
        self._accelerometer.setOnAccelerationChangeHandler(self.updateAcceleration)

        # Open the accelerometer and wait for attachment
        self._accelerometer.openWaitForAttachment(5000)

    def stop(self):
        """
        Stops the accelerometer by closing the connection.

        This will terminate the data collection and release the resources.
        """
        self._accelerometer.close()

    def updateAcceleration(self, accelerometer, acceleration, timestamp):
        """
        Updates the acceleration values, velocity, and displacement based on the latest accelerometer readings.

        This method is called every time the accelerometer sends updated data. It calculates the current acceleration,
        velocity, and displacement, accounting for drift and gravity.

        Args:
            accelerometer (Accelerometer): The accelerometer object triggering the update.
            acceleration (list): The new acceleration values in the x, y, and z directions.
            timestamp (int): The timestamp of the new acceleration measurement.
        """
        # Adjust the acceleration values by removing drift and gravity
        self._acceleration = [
            acc * g - drift - gravity
            for acc, drift, gravity in zip(acceleration, self._drift, self._gravity)
        ]

        # If the timestamp is earlier than the previous one, ignore it
        if timestamp < self._timestamp:
            return
        
        # Calculate velocity and displacement based on the time delta
        dt = timestamp - self._timestamp
        self._velocity = [v + dt * a for a, v in zip(self._acceleration, self._velocity)]
        self._displacement = [x + dt * v for v, x in zip(self._velocity, self._displacement)]

        # Update the timestamp for the next reading
        self._timestamp = timestamp

    def on_attach(self, accelerometer):
        """
        Event handler for when the accelerometer is attached.

        This method is called when the accelerometer device is connected. It resets the timestamp and sets the data rate
        to the desired value, ensuring it falls within the supported range.

        Args:
            accelerometer (Accelerometer): The accelerometer object that was attached.
        """
        self._timestamp = 0  # Reset timestamp when the device is attached
        
        # Validate and set the data rate within the allowable range
        min_data_rate = self._accelerometer.getMinDataRate()
        max_data_rate = self._accelerometer.getMaxDataRate()
        if self._data_rate < min_data_rate or self._data_rate > max_data_rate:
            self.stop()
            raise Exception(f"Invalid Data Rate ({self._data_rate}). Data Rate must be between {min_data_rate} and {max_data_rate}.")
        self._accelerometer.setDataRate(self._data_rate)

    def is_attached(self):
        """
        Checks if the accelerometer is currently attached.

        Returns:
            bool: True if the accelerometer is attached, False otherwise.
        """
        return self._accelerometer.getAttached()

    def calibrate(self, calibration_time=5):
        """
        Calibrates the accelerometer to correct for drift by averaging acceleration readings over the given calibration time.

        This method collects acceleration data for a specified period and adjusts the drift values accordingly.
        It also resets velocity and displacement to zero after calibration.

        Args:
            calibration_time (int): The time duration (in seconds) to collect data for calibration. Default is 5 seconds.
        """
        data_points = calibration_time * self._data_rate

        self._drift = [0, 0, 0]
        temp_drift = [0, 0, 0]
        for i in range(data_points):
            temp_drift = [drift + acc / data_points for drift, acc in zip(temp_drift, self._acceleration)]
            sleep(1 / self._data_rate)
        
        self._drift = temp_drift
        self._velocity = [0, 0, 0]
        self._displacement = [0, 0, 0]

    def getAcceleration(self):
        """
        Retrieves the current acceleration values.

        Returns:
            list: The current acceleration values in the x, y, and z directions (m/s^2).
        """
        return self._acceleration

    def getTimestamp(self):
        """
        Retrieves the timestamp of the latest acceleration reading.

        Returns:
            int: The timestamp of the latest acceleration reading.
        """
        return self._timestamp
