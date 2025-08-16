import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load saved model and preprocessors
model = tf.keras.models.load_model("heart_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.title("❤️ Heart Failure Prediction App")
st.write("Enter patient details to predict the risk of heart disease.")

# --- User inputs ---
age = st.number_input("Age", value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_bp = st.number_input("Resting Blood Pressure (mmHg)", value=120)
cholesterol = st.number_input("Cholesterol (mg/dL)", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak (ST Depression)", value=1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Prepare input data ---
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "ChestPainType": [chest_pain],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "RestingECG": [resting_ecg],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Oldpeak": [oldpeak],
    "ST_Slope": [st_slope]
})

# Encode categorical features
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
num_cols = [col for col in input_df.columns if col not in cat_cols]

encoded_cat = encoder.transform(input_df[cat_cols])
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

final_input = pd.concat([input_df[num_cols], encoded_df], axis=1)

# Scale numeric features
scaled_input = scaler.transform(final_input)

# --- Prediction ---
if st.button("Predict"):
    prediction_prob = model.predict(scaled_input)[0][0]
    prediction = int(prediction_prob > 0.5)

    if prediction == 1:
        st.error(f"🚨 High risk of heart disease! (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ Low risk of heart disease. (Probability: {prediction_prob:.2f})")
