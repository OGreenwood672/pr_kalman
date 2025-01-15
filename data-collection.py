import PhidgetUtils
import numpy as np
import tkinter as tk
from tkinter import ttk
from math import ceil
import os
from utils import *
from time import sleep, time
import configparser

# Initialize the configparser to handle configuration files
config = configparser.ConfigParser()

# Read configuration from 'config.ini' file
config.read('config.ini')

# Frequency of data collection (samples per second)
FREQUENCY = config['data-collection'].getint('frequency')

# Initialize accelerometer using PhidgetUtils with specified data rate
acc = PhidgetUtils.PhidgetAccelerometer(data_rate=FREQUENCY)

# Wait until the accelerometer is attached to the system
while not acc.is_attached():
    sleep(0.1)

print("Accelerometer is attached!")

# Initialize barometer using PhidgetUtils with specified data rate
bar = PhidgetUtils.PhidgetBarometer(data_rate=FREQUENCY)

# Wait until the barometer is attached to the system
while not bar.is_attached():
    sleep(0.1)

print("Barometer is attached!")

# Flag to control whether data collection is in progress
RECORDING = False

# Create a folder to store the recorded data, named with the current timestamp
folder = f"./datastreams/{int(time())}"
os.makedirs(folder, exist_ok=True)

# Define a structured data type for storing timestamp, acceleration, and pressure
dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]

# Calculate the required number of data points from the given minutes
data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)

# Set the data size for 15 minutes of data collection
data_size = data_size_from_mins(15)

# Create a memory-mapped array to store the collected data
data = np.memmap(f"{folder}/0.dat", dtype=dtype, mode='w+', shape=(data_size,))

# Initialize data index and file count for storing data in multiple files
data_index = 0
file_count = 1

# Variable to store the previous timestamp of the accelerometer for time comparison
prev_time = -1

# Initialize the Tkinter GUI window
root = tk.Tk()
root.title("Accelerometer + Barometer Measurements")
root.geometry("400x250")
root.resizable(False, False)

# Button widgets for controlling recording
start_button = stop_button = None

def start():
    """
    Starts the data recording process by setting the RECORDING flag to True.
    Disables the start button and enables the stop button.
    """
    global RECORDING
    RECORDING = True
    stop_button.config(state=tk.NORMAL)
    start_button.config(state=tk.DISABLED)
    record()

def stop():
    """
    Stops the data recording process by setting the RECORDING flag to False.
    Disables the stop button and enables the start button.
    """
    global RECORDING
    RECORDING = False
    stop_button.config(state=tk.DISABLED)
    start_button.config(state=tk.NORMAL)

def record():
    """
    Continuously records accelerometer and barometer data if the RECORDING flag is True.
    Data is stored in a memory-mapped array and written to disk. The recording continues 
    recursively using the root.after method at the defined frequency.
    """
    global prev_time
    global data_index
    global data
    global file_count
    global folder
    global data_size

    if RECORDING:
        # Check if a new timestamp is available from the accelerometer
        if acc.getTimestamp() > prev_time:
            prev_time = acc.getTimestamp()

            # If memory buffer is full, allocate a new file for data storage
            if data_size <= data_index:
                data = np.memmap(f"{folder}/{file_count}.dat", dtype=dtype, mode='w+', shape=(data_size,))
                file_count += 1
                data_index = 0

            # Store the accelerometer and barometer data in memory
            data[data_index] = (
                acc.getTimestamp(),
                magnitude(acc.getAcceleration()) * sign(acc.getAcceleration()[2]),
                bar.getPressure()
            )
            data_index += 1
            
        # Recursively call the record function after the interval based on the frequency
        root.after(ceil(1000 / FREQUENCY), record)

# Styling with ttk (themed Tkinter widgets)
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 14))
style.configure("TFrame", background="#f0f0f0")

# Main frame for the GUI
frame = ttk.Frame(root, padding=20, style="TFrame")
frame.pack(expand=True, fill=tk.BOTH)

# Title label in the GUI
title_label = ttk.Label(
    frame, text="Accelerometer + Barometer", font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

# Status label showing the current status of the system
status_label = ttk.Label(frame, text="Ready to Start", foreground="blue")
status_label.pack(pady=10)

# Frame for holding the buttons
button_frame = ttk.Frame(frame, padding=10, style="TFrame")
button_frame.pack(pady=20)

# Start button to initiate data recording
start_button = ttk.Button(button_frame, text="Start", command=start)
start_button.grid(row=0, column=0, padx=10)

# Stop button to stop data recording, initially disabled
stop_button = ttk.Button(button_frame, text="Stop", command=stop, state=tk.DISABLED)
stop_button.grid(row=0, column=1, padx=10)

# Run the Tkinter application
root.mainloop()

# Stop the accelerometer and barometer when the application exits
acc.stop()
bar.stop()
