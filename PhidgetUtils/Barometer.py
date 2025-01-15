from Phidget22.Phidget import *
from Phidget22.Devices.PressureSensor import *

from scipy.constants import R as R_universal
from scipy.constants import g as g0
import numpy as np


class PhidgetBarometer:
    """
    A wrapper class for interfacing with the Phidget Barometer (Pressure Sensor) device.
    
    This class allows for the collection of pressure data from the Phidget barometer and can convert pressure readings
    into height using the barometric formula. It supports setting the data rate, calibrating the sensor, and managing
    its attachment status.

    Attributes:
        _pressure (float): The current pressure reading in Pascals.
        _data_rate (int): The desired data rate for the barometer (in Hz).
        _barometer (PressureSensor): Phidget PressureSensor object used for data acquisition.
    """

    def __init__(self, data_rate):
        """
        Initializes the PhidgetBarometer with a specified data rate.

        Args:
            data_rate (int): The desired data rate for the barometer (in Hz).
        """
        self._pressure = 0.0  # Pressure in Pascals
        self._data_rate = data_rate  # Desired data rate

        self._barometer = PressureSensor()

        # Set event handlers
        self._barometer.setOnAttachHandler(self.on_attach)
        self._barometer.setOnPressureChangeHandler(self.updatePressure)

        # Open the barometer and wait for attachment
        self._barometer.openWaitForAttachment(5000)

    def stop(self):
        """
        Stops the barometer by closing the connection.

        This method terminates the data collection and releases the resources.
        """
        self._barometer.close()

    def updatePressure(self, barometer, pressure):
        """
        Updates the pressure reading from the barometer.

        This method is triggered whenever a new pressure reading is available. It stores the latest pressure measurement.

        Args:
            barometer (PressureSensor): The barometer object triggering the update.
            pressure (float): The pressure reading in Pascals.
        """
        self._pressure = pressure

    def on_attach(self, barometer):
        """
        Event handler for when the barometer is attached.

        This method is called when the barometer device is connected. It resets the timestamp and sets the data rate
        to the desired value, ensuring it falls within the supported range.

        Args:
            barometer (PressureSensor): The barometer object that was attached.
        """
        self._timestamp = 0  # Reset timestamp when the device is attached
        
        # Validate and set the data rate within the allowable range
        min_data_rate = self._barometer.getMinDataRate()
        max_data_rate = self._barometer.getMaxDataRate()
        if self._data_rate < min_data_rate or self._data_rate > max_data_rate:
            self.stop()
            raise Exception(f"Invalid Data Rate ({self._data_rate}). Data Rate must be between {min_data_rate} and {max_data_rate}.")
        self._barometer.setDataRate(self._data_rate)

    def is_attached(self):
        """
        Checks if the barometer is currently attached.

        Returns:
            bool: True if the barometer is attached, False otherwise.
        """
        return self._barometer.getAttached()

    def getPressure(self):
        """
        Retrieves the current pressure reading.

        Returns:
            float: The current pressure in Pascals.
        """
        return self._pressure


    @staticmethod
    def convert_to_height(pressure, temperature=None):
        """
        Convert given pressure to height,
        @params
        pressure: given pressure
        """
        # Constants
        P_b = 101325  # Reference pressure at sea level (Pa)
        T_Mb = 288.15  # Reference temperature (K)
        L_Mb = -0.0065  # Temperature lapse rate (K/m)
        M = 0.0289644  # Molar mass of Earth's air (kg/mol)
        h_b = 0  # Reference height (m) - using sea level as reference
        
        if temperature is not None:
            # Use first equation (with lapse rate)
            # P = Pb[1 - (L_M,b/T_M,b)(h - hb)]^(g0*M0)/(R*L_M,b)
            exponent = (g0 * M) / (R_universal * L_Mb)
            term = 1 - (pressure / P_b) ** (1 / exponent)
            height = (T_Mb * term) / L_Mb + h_b
            
        else:
            # Use second equation (isothermal case)
            # P = Pb * exp[(-g0*M(h - hb))/(R*T_M,b)]
            height = -(R_universal * T_Mb) / (g0 * M) * np.log(pressure / P_b) + h_b
        
        return height