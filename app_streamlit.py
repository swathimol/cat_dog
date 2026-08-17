"""
STREAMLIT APP — Weather Category Predictor
----------------------------------------------------------------------------
Teaching focus: how fast you can go from model to demo. No HTML required.

Run with:
    python -m pip install streamlit joblib numpy scikit-learn
    python -m streamlit run app_streamlit.py
"""

import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
CLASS_NAMES = list(model.classes_)

st.title("🌦️ Weather Category Predictor (Streamlit)")
st.write("Enter today's readings to predict the weather category.")

temperature = st.number_input("Temperature (°C)", value=20.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, value=10.0, step=1.0)
pressure = st.number_input("Pressure (hPa)", value=1013.0, step=1.0)

if st.button("Predict"):
    X = np.array([[temperature, humidity, wind_speed, pressure]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.success(f"Prediction: **{pred}**")

    st.write("Prediction confidence:")
    for name, p in sorted(zip(CLASS_NAMES, proba), key=lambda x: -x[1]):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))
