from scipy.constants import R as R_universal
from scipy.constants import g as g0
import numpy as np

def calculate_height(pressure, temperature=None):
    """
    Calculate geopotential height based on pressure using either constant lapse rate
    or isothermal atmospheric equations.
    
    Args:
        pressure (float): Pressure at the height to be calculated (Pa)
        temperature (float, optional): Temperature if using lapse rate equation (K)
    
    Returns:
        float: Geopotential height in meters
        
    Reference values based on International Standard Atmosphere (ISA):
    - Reference pressure (Pb) = 101325 Pa at sea level
    - Reference temperature (T_M,b) = 288.15 K
    - Lapse rate (L_M,b) = -0.0065 K/m in troposphere
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

if __name__ == "__main__":
    # Test with standard atmospheric pressure at sea level
    test_pressure = 101325  # Pa
    
    # Test both equations
    height1 = calculate_height(test_pressure, temperature=288.15)
    height2 = calculate_height(test_pressure)
    
    print(f"Height (with temperature): {height1:.2f} m")
    print(f"Height (isothermal): {height2:.2f} m")

    # Max Temp Change (Burj Khalifa) = 7.5 degrees celcius
    # def get_pressure_height(df):
    #     T0 = 288.15  # Standard temperature at sea level in Kelvin
    #     L = 0.0065   # Temperature lapse rate in K/m
    #     P0 = 101325  # Standard atmospheric pressure at sea level in Pascals
    #     R = 8.314    # Universal gas constant in J/(mol·K)
    #     g = 9.80665  # Acceleration due to gravity in m/s²
    #     # M = 0.0289644  # Molar mass of dry air in kg/mol
    #     humidity = 0.25
    #     M = humidity * 0.0180 + (1 - humidity) * 0.0289644

    #     assert 'pressure' in df.columns, "DataFrame must contain a 'pressure' column."
        
    #     df['pHeight'] = 0.0

    #     # Calculate height from pressure using the barometric formula
    #     for i in range(len(df)):
    #         P = df['pressure'].iloc[i] * 100  # Pressure in Pascals
    #         if P <= 0:
    #             raise ValueError("Pressure must be positive.")
            
    #         height = (T0 / L) * (1 - (P / P0)**(R * L / (g * M)))
    #         df.at[i, 'pHeight'] = height

    #     df["pHeight"] = df["pHeight"] - df["pHeight"].iloc[0];
