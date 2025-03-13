import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🌊 Capability Predictor App")
st.markdown("### Enter Minimum Temperature, Maximum Temperature, and Humidity")

# User input fields
mintempC = st.number_input("Min Temperature (°C)", min_value=-50, max_value=50, value=20)
maxtempC = st.number_input("Max Temperature (°C)", min_value=-50, max_value=60, value=30)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=45)

# Prediction function
def predict_capability(mintempC, maxtempC, humidity):
    input_data = np.array([[mintempC, maxtempC, humidity]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return "✅ Capable" if prediction[0] == 1 else "❌ Not Capable"

# Predict when button is clicked
if st.button("Check Capability"):
    result = predict_capability(mintempC, maxtempC, humidity)
    st.subheader(f"Result: {result}")


