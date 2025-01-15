
import PhidgetUtils
import numpy as np
import tkinter as tk
from tkinter import ttk
from math import ceil
import os
from utils import *

from time import sleep, time

import configparser

# Initialize the parser
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

FREQUENCY = config['data-collection'].getint('frequency')


acc = PhidgetUtils.PhidgetAccelerometer(data_rate=FREQUENCY)

while not acc.is_attached():
    sleep(0.1)

print("Accelerometer is attached!")


bar = PhidgetUtils.PhidgetBarometer(data_rate=FREQUENCY)

while not bar.is_attached():
    sleep(0.1)

print("Barometer is attached!")


RECORDING = False

folder = f"./datastreams/{int(time())}"
os.makedirs(folder, exist_ok=True)

dtype = [('timestamp', 'int32'), ('acceleration', 'float32'), ('pressure', 'float32')]

data_size_from_mins = lambda mins : int(FREQUENCY * 60 * mins)

data_size = data_size_from_mins(15)
data = np.memmap(f"{folder}/0.dat", dtype=dtype, mode='w+', shape=(data_size,))
data_index = 0
file_count = 1

prev_time = -1

root = tk.Tk()
root.title("Accelerometer + Barometer Measurements")
root.geometry("400x250")
root.resizable(False, False)


start_button = stop_button = None

def start():
    global RECORDING
    RECORDING = True
    stop_button.config(state=tk.NORMAL)
    start_button.config(state=tk.DISABLED)
    record()

def stop():
    global RECORDING
    RECORDING = False
    stop_button.config(state=tk.DISABLED)
    start_button.config(state=tk.NORMAL)

def record():

    global prev_time
    global data_index
    global data
    global file_count
    global folder
    global data_size

    if RECORDING:
        if acc.getTimestamp() > prev_time:
            prev_time = acc.getTimestamp()

            if data_size <= data_index:
                data = np.memmap(f"{folder}/{file_count}.dat", dtype=dtype, mode='w+', shape=(data_size,))
                file_count += 1
                data_index = 0

            data[data_index] = (
                acc.getTimestamp(),
                magnitude(acc.getAcceleration()) * sign(acc.getAcceleration()[2]),
                bar.getPressure()
            )
            data_index += 1
            
        root.after(ceil(1000 / FREQUENCY), record)


# Styling with ttk
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 14))
style.configure("TFrame", background="#f0f0f0")

# Main frame
frame = ttk.Frame(root, padding=20, style="TFrame")
frame.pack(expand=True, fill=tk.BOTH)

# Title Label
title_label = ttk.Label(
    frame, text="Accelerometer + Barometer", font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

# Status Label
status_label = ttk.Label(frame, text="Ready to Start", foreground="blue")
status_label.pack(pady=10)

# Buttons
button_frame = ttk.Frame(frame, padding=10, style="TFrame")
button_frame.pack(pady=20)

start_button = ttk.Button(button_frame, text="Start", command=start)
start_button.grid(row=0, column=0, padx=10)

stop_button = ttk.Button(button_frame, text="Stop", command=stop, state=tk.DISABLED)
stop_button.grid(row=0, column=1, padx=10)

# Run the application
root.mainloop()



acc.stop()
bar.stop()