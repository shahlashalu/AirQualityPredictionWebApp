#app.py

import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('aqi_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Apply custom CSS to style the page
st.markdown("""
    <style>
        /* Body background and general styling */
        .reportview-container {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
            color: #333;
        }

        /* Header Styling */
        h1 {
            text-align: center;
            color: #4caf50;
            font-size: 3em;
            font-weight: bold;
        }

        h2 {
            text-align: center;
            color: #2196f3;
            font-size: 2em;
            font-weight: normal;
        }

        /* Input box styling */
        .stNumberInput, .stTextInput {
            font-size: 1.2em;
            padding: 10px;
            border-radius: 5px;
            border: 2px solid #4caf50;
        }

        .stButton {
            background-color: #4caf50;
            color: white;
            font-size: 1.2em;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton:hover {
            background-color: #45a049;
        }

        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #2e7d32;
            color: white;
        }

        /* Prediction result */
        .result {
            font-size: 1.5em;
            font-weight: bold;
            color: #f44336;
            text-align: center;
        }

        /* Section Styling */
        .section {
            margin: 20px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Title of the app
st.title("Air Quality Index Prediction")

# Input fields for user input
so2 = st.number_input('Enter SO2 (sulphur dioxide) concentration:', min_value=0.0)
no2 = st.number_input('Enter NO2 (nitrogen dioxide) concentration:', min_value=0.0)
rspm = st.number_input('Enter RSPM (respirable suspended particulate matter) concentration:', min_value=0.0)
spm = st.number_input('Enter SPM (suspended particulate matter) concentration:', min_value=0.0)

# Button to trigger prediction
if st.button('Predict AQI'):
    # Functions to calculate individual pollutant indices
    def cal_SOi(so2):
        if so2 <= 40:
            return so2 * (50/40)
        elif so2 <= 80:
            return 50 + (so2-40)*(50/40)
        elif so2 <= 380:
            return 100 + (so2-80)*(100/300)
        elif so2 <= 800:
            return 200 + (so2-380)*(100/420)
        elif so2 <= 1600:
            return 300 + (so2-800)*(100/800)
        else:
            return 400 + (so2-1600)*(100/800)

    def cal_Noi(no2):
        if no2 <= 40:
            return no2*50/40
        elif no2 <= 80:
            return 50+(no2-40)*(50/40)
        elif no2 <= 180:
            return 100+(no2-80)*(100/100)
        elif no2 <= 280:
            return 200+(no2-180)*(100/100)
        elif no2 <= 400:
            return 300+(no2-280)*(100/120)
        else:
            return 400+(no2-400)*(100/120)

    def cal_RSPMT(rspm):
        if rspm <= 30:
            return rspm*50/30
        elif rspm <= 60:
            return 50+(rspm-30)*50/30
        elif rspm <= 90:
            return 100+(rspm-60)*100/30
        elif rspm <= 120:
            return 200+(rspm-90)*100/30
        elif rspm <= 250:
            return 300+(rspm-120)*(100/130)
        else:
            return 400+(rspm-250)*(100/130)

    def cal_SPMi(spm):
        if spm <= 50:
            return spm*50/50
        elif spm <= 100:
            return 50+(spm-50)*(50/50)
        elif spm <= 250:
            return 100+(spm-100)*(100/150)
        elif spm <= 350:
            return 200+(spm-250)*(100/100)
        elif spm <= 430:
            return 300+(spm-350)*(100/80)
        else:
            return 400+(spm-430)*(100/430)

    # Calculate pollutant indices
    soi = cal_SOi(so2)
    noi = cal_Noi(no2)
    rpi = cal_RSPMT(rspm)
    spi = cal_SPMi(spm)

    # Prepare input data for prediction
    input_data = np.array([[soi, noi, rpi, spi]])

    # Get prediction from the model
    prediction = model.predict(input_data)

    # Display the predicted AQI
    st.success(f"Predicted AQI: {round(prediction[0], 2)}")

    # Categorize the AQI into the respective range
    if prediction[0] <= 50:
        st.info("Air Quality: Good (0-50) - The air is fresh and free from toxins.")
    elif prediction[0] <= 100:
        st.info("Air Quality: Satisfactory/Moderate (51-100) - Acceptable air quality, but sensitive individuals might experience minor discomfort.")
    elif prediction[0] <= 200:
        st.warning("Air Quality: Unhealthy/Moderately polluted (101-200) - Breathing may become uncomfortable, especially for those with respiratory issues.")
    elif prediction[0] <= 300:
        st.error("Air Quality: Poor (201-300) - Prolonged exposure can cause chronic health issues or organ damage.")
    elif prediction[0] <= 400:
        st.error("Air Quality: Very Poor (301-400) - Dangerously high pollution levels.")
    else:
        st.error("Air Quality: Severe/Hazardous (401-500) - Very high pollution levels, requiring immediate action.")
