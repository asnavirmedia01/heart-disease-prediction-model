import streamlit as st
import pandas as pd
import joblib

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# -------------------- Load Model Artifacts --------------------
try:
    loaded_model = joblib.load('best_ada_model.joblib')
    loaded_feature_names = joblib.load('feature_names.joblib')
    loaded_label_encoders = joblib.load('label_encoders.joblib')
except FileNotFoundError as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# -------------------- Prediction Function --------------------
def predict_heart_disease(
    Smoking: str,
    Age: int,
    Family_Heart_Disease: str,
    BMI: float,
    Cholesterol_Level: int,
    Blood_Pressure: int,
    Stress_Level: str,
    Diabetes: str,
    Homocysteine_Level: float
) -> str:

    input_raw_data = {
        'Smoking': Smoking,
        'Age': Age,
        'Family Heart Disease': Family_Heart_Disease,
        'BMI': BMI,
        'Cholesterol Level': Cholesterol_Level,
        'Blood Pressure': Blood_Pressure,
        'Stress Level': Stress_Level,
        'Diabetes': Diabetes,
        'Homocysteine Level': Homocysteine_Level
    }

    input_df_processed = pd.DataFrame([input_raw_data])

    for col, encoder in loaded_label_encoders.items():
        if col in input_df_processed.columns:
            try:
                input_df_processed[col] = encoder.transform(input_df_processed[col])
            except ValueError:
                return "Error: Invalid input"

    input_df = input_df_processed[loaded_feature_names]
    prediction = loaded_model.predict(input_df)[0]

    return 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'

# -------------------- UI Layout --------------------
st.title("Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    age = st.slider("Age", 1, 90, 45)
    bmi = st.slider("BMI", 18.0, 35.0, 25.0, 0.1)
    cholesterol_level = st.slider("Cholesterol Level", 150, 319, 200)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])

with col2:
    family_heart_disease = st.selectbox("Family Heart Disease", ["No", "Yes"])
    blood_pressure = st.slider("Blood Pressure", 90, 179, 120)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    homocysteine_level = st.slider("Homocysteine Level", 5.0, 19.99, 10.0, 0.01)

st.markdown("---")

# -------------------- Prediction Button --------------------
if st.button("Predict Heart Disease"):
    result = predict_heart_disease(
        Smoking=smoking,
        Age=age,
        Family_Heart_Disease=family_heart_disease,
        BMI=bmi,
        Cholesterol_Level=cholesterol_level,
        Blood_Pressure=blood_pressure,
        Stress_Level=stress_level,
        Diabetes=diabetes,
        Homocysteine_Level=homocysteine_level
    )

    if "Error" in result:
        st.error(result)
    else:
        st.success(f"Prediction: {result}")

# -------------------- Debug Reference --------------------
st.subheader("Features used for Prediction")
st.write(loaded_feature_names)