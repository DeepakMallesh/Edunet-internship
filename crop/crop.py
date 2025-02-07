import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load the pre-trained model and scaler
model_path = 'crop_dtc_model.pkl'
scaler_path = 'crop_scaler.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    dtc = joblib.load(model_path)

if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
else:
    scaler = joblib.load(scaler_path)

# Define crop_rec function
def crop_rec(N, P, K, temp, hum, ph, rain):
    features = pd.DataFrame([[N, P, K, temp, hum, ph, rain]])
    
    try:
        transformed_features = scaler.transform(features)
        prediction = dtc.predict(transformed_features).reshape(1, -1)
        
        crop_dict = {
            0: 'Rice',
            1: 'Maize',
            2: 'Chickpea',
            3: 'Kidney Beans',
            4: 'Pigeon Peas',
            5: 'Moth Beans',
            6: 'Mung Bean',
            7: 'Black Gram',
            8: 'Lentil',
            9: 'Pomegranate',
            10: 'Banana',
            11: 'Mango',
            12: 'Grapes',
            13: 'Watermelon',
            14: 'Muskmelon',
            15: 'Apple',
            16: 'Orange',
            17: 'Papaya',
            18: 'Coconut',
            19: 'Cotton',
            20: 'Jute',
            21: 'Coffee'
        }
        
        crop = [crop_dict[i] for i in prediction[0]]
        return f"{crop[0]} is the best crop to grow on the farm"
    except Exception as e:
        return f"Error during prediction: {e}"

# Streamlit app
st.markdown("""
    <style>
    .stButton button {
        margin: auto;
        display: block;
        font-size: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
st.title('Crop Recommendation System')

N = st.text_input('Enter Nitrogen (N)', placeholder="Enter the value")
P = st.text_input('Enter Phosphorous (P)', placeholder="Enter the value")
K = st.text_input('Enter Potassium (K)', placeholder="Enter the value")
temp = st.text_input('Enter Temperature (°C)', placeholder="Enter the value")
hum = st.text_input('Enter Humidity (%)', placeholder="Enter the value")
ph = st.text_input('Enter pH', placeholder="Enter the value")
rain = st.text_input('Enter Rainfall (mm)', placeholder="Enter the value")

if st.button('Predict'):
    try:
        result = crop_rec(float(N), float(P), float(K), float(temp), float(hum), float(ph), float(rain))
        st.write(result)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
