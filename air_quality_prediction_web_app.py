# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:34:33 2024

@author: hp
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/hp/OneDrive/ドキュメント/trained_model.sav', 'rb'))

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


















    