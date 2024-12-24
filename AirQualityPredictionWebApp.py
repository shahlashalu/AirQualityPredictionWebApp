# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:19:52 2024

@author: hp
"""

import numpy as np
import os
import pickle
import streamlit as st


# Determine the path to the model file
current_dir = os.path.dirname(__file__)  # Get the directory of the current script
model_path = os.path.join(current_dir, 'C:/Users/hp/air_quality_prediction/trained_model.sav')

# Check if the file exists before attempting to load it
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")



#  creating a function for prediction

def air_quality(input_data):
    input_data = (4.7,7.5,10.9,45.9)

    # changing the input data to numpy array
    input_data_as_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
      return 'Good'
    elif (prediction[0] == 1):
      return 'Moderate'
    elif (prediction[0] == 2):
      return 'Satisfacory'
    elif (prediction[0] == 3):
      return 'Poor'
    elif (prediction[0] == 4):
      return 'Severe'
    elif (prediction[0] == 5):
      return 'Hazardous'


def main():
    
    # giving a title
    st.title('Air Quality Prediction Web App')
   
   
    # getting the input data from the user

   
    SOi = st.text_input('Value of SOi')
    Noi = st.text_input('Value of Noi')
    Rpi = st.text_input('Value of Rpi')
    SPMi = st.text_input('Value of SPMi')
   
   
   
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Quality of Air'):
        diagnosis = air_quality([SOi, Noi, Rpi, SPMi])
       
       
    st.success(diagnosis)


if __name__ == '__main__':
    main()


