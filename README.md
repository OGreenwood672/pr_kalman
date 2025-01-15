# **Lift Data Collection and Analysis Guide**

## **First Time Setup Instructions**

1. **Download the Repository**

   - Clone or download the repository onto your device.

2. **Set Up a Virtual Environment**

   ### Windows:

   - Open a terminal in the repository directory.
   - Run the following commands:
     ```
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     ```

   ### Linux/Mac:

   - Open a terminal in the repository directory.
   - Run the following commands:
     ```
     python3 -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     sudo apt-get install python3-tk
     ```

3. **Deactivate the Virtual Environment:**

   - To deactivate the virtual environment, simply run:
     ```
     deactivate
     ```

4. **Reactivating the Virtual Environment:**

   - Every time you return to the project, you need to activate the virtual environment before running any scripts:
     
     ### Windows:
     ```
     venv\Scripts\activate
     ```

     ### Linux/Mac:
     ```
     source venv/bin/activate
     ```

---

## **Data Collection Instructions**

1. **Prepare the Setup**

   - Head to the **lowest level** the lift can access.
   - Connect the **Accelerometer** and **Barometer** to your laptop via USB.

2. **Start Data Collection**

   - Run: `data-collection.py`.
   - A TKinter application will appear — click **Start** when still on the ground floor.

   **Note for Linux Users:** You must run the data collection process with `sudo -E python data-collection.py` to ensure proper access to the USB devices.

3. **Record Lift Movements**

   - Press **all the floor buttons** on the lift, ascending **one floor at a time** to the top.
   - Once at the top, you can freely move between floors.

4. **Save Collected Data**

   - The collected data will automatically be saved as a `.dat` file in the `./datastreams` folder.
   - By default, the folder name will be a **Unix timestamp** marking when you started the program.
   - **Recommendation:** Rename the folder to the **location name** for clarity.

---

## **Data Analysis Instructions**

1. **Run the Analysis**

   - Execute the `analyse_multi_journey.py` file.
   - Enter the **name of the folder** where the data is stored (either the location name or the Unix timestamp).

2. **Understand the Outputs**

   - **Acceleration Profile:**
     - The program will display the lift's acceleration profile alongside detected journey stages (when the lift is moving).
   - **Floor Predictions:**
     - In the terminal, the program will output the floors you traveled to (or its best estimation).
   - **Height Changes:**
     - Graphs will show the lift's height changes during the journey, derived from:
       - Accelerometer data
       - Barometer data
       - A **fusion** of both for improved accuracy.

---

### **Notes**

- For consistent results, follow the recommended movement pattern during data collection.
- **Always activate the virtual environment before running any scripts** using the instructions provided in the "First Time Setup Instructions."
- Remember to deactivate the virtual environment when finished by running `deactivate`.


## Documentation

- Run `pdoc PhidgetUtils sensor_fusion analyse_journey.py analyse_multi_journey.py globals.py utils.py`
- If that fails, use the docstrings inside the files - they are the same thing

## Future Improvement

My final day of work I have found barometer can be affected by acceleration, leading to lag. >:[
This may lead to error when accelerating larger amount, however eventual consistency is better.