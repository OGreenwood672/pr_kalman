
# Data Collection Instructions

Head to the lowest level the lift gets to
Connect Accelerometer and Barometer to laptop via usb
Click Run All on the juypter notebook - data_stream_collection.ipynb.
Click start on the TKinter application
Press all the floor buttons on the lift (go up one floor at a time to the top)
Once at the top, you can go anywhere from then on
Data collected will be stored in a .dat file in ./datastreams folder
It will be stored in a folder with the unix time stamp from when you began the program -> recommend changing to location name

# Data Analysis
Run the analyse_multi_journey.py file
Enter name of folder saved in -> (Either will be location or unix time stamp)

You will initially see acceleration profile from the accelerometer compared with the stages (ie when we think the lift is in a journey)
The terminal will then show the floors you travelled to (Or the program believes you have travelled to)
Then the program will display the height changes throughout the journey from accelerometer data, barometer data and finally a fusion of both.