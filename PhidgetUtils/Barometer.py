from Phidget22.Phidget import *
from Phidget22.Devices.PressureSensor import *


class PhidgetBarometer:
    
    def __init__(self, data_rate):
        self._pressure = 0.0  # Pressure in Pascals
        self._timestamp = -1  # Timestamp of the latest reading
        self._data_rate = data_rate  # Desired data rate

        self._barometer = PressureSensor()

        # Set event handlers
        self._barometer.setOnAttachHandler(self.on_attach)
        self._barometer.setOnPressureChangeHandler(self.updatePressure)

        # Open the barometer
        self._barometer.openWaitForAttachment(5000)

    def stop(self):
        """Stop the barometer by closing the connection."""
        self._barometer.close()

    def updatePressure(self, pressure, timestamp):
        """
        Update the pressure value and timestamp.
        :param pressure: The pressure reading in Pascals.
        :param timestamp: The timestamp of the reading.
        """
        self._timestamp = timestamp
        self._pressure = pressure

    def on_attach(self):
        """
        Event handler for when the barometer is attached.
        Resets the timestamp and sets the data rate.
        """
        self._timestamp = 0  # Reset timestamp
        
        # Validate and set the data rate
        min_data_rate = self._barometer.getMinDataRate()
        max_data_rate = self._barometer.getMaxDataRate()
        if self._data_rate < min_data_rate or self._data_rate > max_data_rate:
            self.stop()
            raise Exception(f"""
                Invalid Data Rate ({self._data_rate})
                Data Rate must be between {min_data_rate} and {max_data_rate}
            """)
        self._barometer.setDataRate(self._data_rate)

    def is_attached(self):
        """
        Check if the barometer is attached.
        :return: True if attached, False otherwise.
        """
        return self._barometer.getAttached()

    def getPressure(self):
        """
        Get the current pressure reading.
        :return: Pressure in Pascals.
        """
        return self._pressure

    def getTimestamp(self):
        """
        Get the timestamp of the latest pressure reading.
        :return: Timestamp as an integer.
        """
        return self._timestamp
