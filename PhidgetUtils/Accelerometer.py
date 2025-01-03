from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *
from time import sleep

from scipy.constants import g


class PhidgetAccelerometer:
    
    def __init__(self, data_rate):

        self._acceleration = [0, 0, 0]
        self._velocity = [0, 0, 0]
        self._displacement = [0, 0, 0]

        self._drift = [0, 0, 0]
        self._gravity = [0, 0, g]
        self._timestamp = -1
        self._data_rate = data_rate

        self._accelerometer = Accelerometer()

        # Set event handlers
        self._accelerometer.setOnAttachHandler(self.on_attach)
        self._accelerometer.setOnAccelerationChangeHandler(self.updateAcceleration)

        # Open the accelerometer
        self._accelerometer.openWaitForAttachment(5000)

    def stop(self):
        """Stop the accelerometer by closing the connection."""
        self._accelerometer.close()

    def updateAcceleration(self, accelerometer, acceleration, timestamp):
        """
        Update the acceleration values and timestamp.
        :param acceleration: List of acceleration values [x, y, z].
        :param timestamp: Timestamp of the reading.
        """
        self._acceleration = [
            acc * g - drift - gravity
            for acc, drift, gravity in zip(acceleration, self._drift, self._gravity)
        ]

        if timestamp < self._timestamp:
            return
        
        dt = timestamp - self._timestamp
        self._velocity = [v + dt * a for a, v in zip(self._acceleration, self._velocity)]
        self._displacement = [x + dt * v for v, x in zip(self._velocity, self._displacement)]

        self._timestamp = timestamp


    def on_attach(self, accelerometer):
        """
        Event handler for when the accelerometer is attached.
        Resets the timestamp and sets the data rate.
        """
        self._timestamp = 0  # Reset timestamp
        
        # Validate and set the data rate
        min_data_rate = self._accelerometer.getMinDataRate()
        max_data_rate = self._accelerometer.getMaxDataRate()
        if self._data_rate < min_data_rate or self._data_rate > max_data_rate:
            self.stop()
            raise Exception(f"""
                Invalid Data Rate ({self._data_rate})
                Data Rate must be between {min_data_rate} and {max_data_rate}
            """)
        self._accelerometer.setDataRate(self._data_rate)

    def is_attached(self):
        """
        Check if the accelerometer is attached.
        :return: True if attached, False otherwise.
        """
        return self._accelerometer.getAttached()

    def calibrate(self, calibration_time=5):

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
        """Get the current acceleration values."""
        return self._acceleration

    def getTimestamp(self):
        """Get the timestamp of the latest acceleration reading."""
        return self._timestamp
