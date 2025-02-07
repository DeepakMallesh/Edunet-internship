# fertilizer_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load the pre-trained model, scaler, and feature names
model_path = 'fertilizer_model.pkl'
scaler_path = 'scaler.pkl'
feature_names_path = 'feature_names.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    dtc = joblib.load(model_path)

if not os.path.exists(scaler_path):
    st.error(f"Scaler file not found: {scaler_path}")
else:
    scaler = joblib.load(scaler_path)

if not os.path.exists(feature_names_path):
    st.error(f"Feature names file not found: {feature_names_path}")
else:
    feature_names = joblib.load(feature_names_path)

# Categorical feature dictionaries
soil_dict = {
    'Sandy': 0,
    'Loamy': 1,
    'Black': 2,
    'Red': 3,
    'Clayey': 4
}
crop_dict = {
    'Rice': 0,
    'Maize': 1,
    'Sugarcane': 2,
    'Cotton': 3,
    'Tobacco': 4,
    'Paddy': 5,
    'Barley': 6,
    'Wheat': 7,
    'Millets': 8,
    'Oil seeds': 9,
    'Pulses': 10,
    'Ground Nuts': 11
}

# Define fert_rec function
def fert_rec(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    # Encode categorical features
    if Soil_Type not in soil_dict or Crop_Type not in crop_dict:
        return "Invalid Soil Type or Crop Type"
    
    Soil_Type_encoded = soil_dict[Soil_Type]
    Crop_Type_encoded = crop_dict[Crop_Type]
    
    features = pd.DataFrame({
        'Temparature': [Temparature],
        'Humidity': [Humidity],
        'Moisture': [Moisture],
        'Soil_Type': [Soil_Type_encoded],
        'Crop_Type': [Crop_Type_encoded],
        'Nitrogen': [Nitrogen],
        'Potassium': [Potassium],
        'Phosphorous': [Phosphorous]
    })

    # Ensure feature columns align with those during training
    features = pd.get_dummies(features)
    for col in feature_names:
        if col not in features.columns:
            features[col] = 0
    features = features[feature_names]
    
    try:
        transformed_features = scaler.transform(features)
        prediction = dtc.predict(transformed_features).reshape(1, -1)
        
        fert_dict = {
            0: 'Urea',
            1: 'DAP',
            2: '14-35-14',
            3: '28-28',
            4: '17-17-17',
            5: '20-20',
            6: '10-26-26'
        }
        
        fert = [fert_dict[i] for i in prediction[0]]
        return f"{fert[0]} is the best fertilizer to use"
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
st.title('Fertilizer Recommendation System')

Temparature = st.text_input('Enter Temparature (°C)', placeholder="Enter the value")
Humidity = st.text_input('Enter Humidity (%)', placeholder="Enter the value")
Moisture = st.text_input('Enter Moisture (units)', placeholder="Enter the value")
Soil_Type = st.selectbox('Select Soil Type', ['Select', 'Clayey', 'Sandy', 'Loamy', 'Black', 'Red'])
Crop_Type = st.selectbox('Select Crop Type', ['Select', 'Rice', 'Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'])
Nitrogen = st.text_input('Enter Nitrogen (units)', placeholder="Enter the value")
Potassium = st.text_input('Enter Potassium (units)', placeholder="Enter the value")
Phosphorous = st.text_input('Enter Phosphorous (units)', placeholder="Enter the value")

if st.button('Predict'):
    if Soil_Type == 'Select' or Crop_Type == 'Select':
        st.error('Please select valid options for Soil Type and Crop Type.')
    else:
        try:
            result = fert_rec(float(Temparature), float(Humidity), float(Moisture), Soil_Type, Crop_Type, float(Nitrogen), float(Potassium), float(Phosphorous))
            st.write(result)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
