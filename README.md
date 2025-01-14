# **Lift Data Collection and Analysis Guide**

## **Data Collection Instructions**

1. **Prepare the Setup**  
   - Head to the **lowest level** the lift can access.  
   - Connect the **Accelerometer** and **Barometer** to your laptop via USB.

2. **Start Data Collection**  
   - Open the Jupyter Notebook: `data_stream_collection.ipynb`.  
   - Click **Run All** to initialize the program.  
   - A TKinter application will appear — click **Start**.

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
- Ensure the sensors are properly calibrated before starting the program.  
- For consistent results, follow the recommended movement pattern during data collection.
