# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:48:58 2024

@author: hp
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/hp/air_quality_prediction/trained_model.sav', 'rb'))

# Function to map predictions to labels
def map_prediction_to_label(prediction):
    if prediction == 0:
        return 'Good'
    elif prediction == 1:
        return 'Moderate'
    elif prediction == 2:
        return 'Satisfactory'
    elif prediction == 3:
        return 'Poor'
    elif prediction == 4:
        return 'Severe'
    elif prediction == 5:
        return 'Hazardous'

# Streamlit app
def main():
    # Title of the web app
    st.title("Air Quality Prediction Web App")

    # User inputs for air quality indices
    st.write("Enter the values for the following air quality indices:")
    so = st.number_input("SOi (Sulfur Oxide)", min_value=0.0, format="%.2f")
    no = st.number_input("Noi (Nitrogen Oxide)", min_value=0.0, format="%.2f")
    rp = st.number_input("Rpi (Respirable Particulates)", min_value=0.0, format="%.2f")
    spm = st.number_input("SPMi (Suspended Particulate Matter)", min_value=0.0, format="%.2f")

    # Prediction button
    if st.button("Predict Quality of Air"):
        # Prepare input data
        input_data = np.asarray([so, no, rp, spm]).reshape(1, -1)

        # Predict using the loaded model
        prediction = loaded_model.predict(input_data)[0]

        # Display the prediction
        st.success(f"The predicted air quality is: {map_prediction_to_label(prediction)}")

# Run the app
if __name__ == "__main__":
    main()
