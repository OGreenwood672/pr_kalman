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
     ```

3. **Deactivate the Virtual Environment:**

   - To deactivate, simply run:
     ```
     deactivate
     ```

## **Data Collection Instructions**

1. **Prepare the Setup**

   - Head to the **lowest level** the lift can access.
   - Connect the **Accelerometer** and **Barometer** to your laptop via USB.

2. **Start Data Collection**

   - Open the Jupyter Notebook: `data_stream_collection.ipynb`.
   - Click **Run All** to initialize the program.
   - A TKinter application will appear — click **Start**.

   **Note for Linux Users:** You must run the data collection process with `sudo` to ensure proper access to the USB devices.

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
- Remember to deactivate the virtual environment when finished by running `deactivate`.

