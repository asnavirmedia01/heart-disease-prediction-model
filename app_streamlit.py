import streamlit as st
import pandas as pd
import joblib

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

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

# -------------------- Session State --------------------
if "records" not in st.session_state:
    st.session_state.records = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# -------------------- UI Layout --------------------
st.title("Heart Disease Prediction App")
st.write("This application predicts heart disease risk using a trained machine learning model.")

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    age = st.number_input("Age", 1, 120, 45)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
    cholesterol_level = st.number_input("Cholesterol Level", 100, 400, 200)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])

with col2:
    family_heart_disease = st.selectbox("Family Heart Disease", ["No", "Yes"])
    blood_pressure = st.number_input("Blood Pressure", 70, 250, 120)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    homocysteine_level = st.number_input("Homocysteine Level", 2.0, 50.0, 10.0, 0.1)

st.markdown("---")

btn1, btn2 = st.columns(2)

with btn1:
    if st.button("Predict"):
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

        st.session_state.last_result = result

        st.session_state.records.append({
            "Smoking": smoking,
            "Age": age,
            "Family Heart Disease": family_heart_disease,
            "BMI": bmi,
            "Cholesterol Level": cholesterol_level,
            "Blood Pressure": blood_pressure,
            "Stress Level": stress_level,
            "Diabetes": diabetes,
            "Homocysteine Level": homocysteine_level,
            "Prediction": result
        })

with btn2:
    if st.button("Reset"):
        st.experimental_rerun()

# -------------------- Result Display --------------------
if st.session_state.last_result:
    st.subheader("Prediction Result")
    st.success(st.session_state.last_result)

# -------------------- Table Display & Export --------------------
if st.session_state.records:
    st.subheader("Patient Records")

    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, use_container_width=True)

    excel_file = "patient_records.xlsx"
    df.to_excel(excel_file, index=False)

    with open(excel_file, "rb") as f:
        st.download_button(
            "Download Records (Excel)",
            f,
            excel_file,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
