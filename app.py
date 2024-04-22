import streamlit as st
import pandas as pd
import numpy as np
from joblib import load


def load_model(model_name):
    # Ensure that the path includes the .joblib extension
    return load(model_name)

# Specify the correct path and filename when loading
models = {
    'Régression Linéaire': load_model(r'C:\Users\Tifa\Downloads\AIQ\Régression_Linéaire.joblib'),
    'Régression de Forêt Aléatoire': load_model(r'C:\Users\Tifa\Downloads\AIQ\Régression_de_Forêt_Aléatoire.joblib'),
    'Régression par Gradient Boosting': load_model(r'C:\Users\Tifa\Downloads\AIQ\Régression_par_Gradient_Boosting.joblib')
}


# Streamlit interface
st.title('Air Quality Prediction Interface')

# Model selector
model_option = st.selectbox('Select Model:', list(models.keys()))

# User input features
st.subheader('Enter the feature values:')
pm25 = st.number_input('PM2.5 Median', value=10.0)
o3 = st.number_input('O3 Median', value=30.0)
no2 = st.number_input('NO2 Median', value=20.0)
co = st.number_input('CO Median', value=0.5)
so2 = st.number_input('SO2 Median', value=3.0)
pm10 = st.number_input('PM10 Median', value=40.0)

# Make prediction
if st.button('Predict Air Quality'):
    features = np.array([[pm25, o3, no2, co, so2, pm10]])
    prediction = models[model_option].predict(features)
    st.write(f'Predicted Air Quality Index (AQI): {prediction[0]:.2f}')

# Optional: Display the code or model metrics
st.text_area('Code Snippet:', '''<Your model training and evaluation code here>''', height=300)
