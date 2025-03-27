import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model
model_path = "trained_model.sav"

# Check if model exists before loading
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.success("✅ Model loaded successfully!")
else:
    model = None
    st.error("❌ Model not found! Please check the file path.")

# Function to map predictions to labels
def map_prediction_to_label(prediction):
    labels = {
        0: 'Good',
        1: 'Moderate',
        2: 'Satisfactory',
        3: 'Poor',
        4: 'Severe',
        5: 'Hazardous'
    }
    return labels.get(prediction, "Unknown")

# Streamlit UI
st.title("🌍 Air Quality Prediction Web App")

# User Inputs
so = st.number_input("SO (Sulfur Oxide)", min_value=0.0, max_value=500.0, value=45.0)
no = st.number_input("NO (Nitrogen Oxide)", min_value=0.0, max_value=500.0, value=56.0)
rp = st.number_input("RP (Respirable Particulates)", min_value=0.0, max_value=500.0, value=45.89)
spm = st.number_input("SPM (Suspended Particulate Matter)", min_value=0.0, max_value=1000.0, value=700.0)

# Prediction Button
if st.button("Predict Air Quality"):
    if model:
        input_data = np.array([[so, no, rp, spm]])
        prediction = model.predict(input_data)[0]
        prediction_label = map_prediction_to_label(prediction)

        st.success(f"🌱 Predicted Air Quality: **{prediction_label}**")
    else:
        st.error("Model not found! Please check the file.")
