import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# -------------------- Page Config --------------------
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")

# -------------------- Load Model --------------------
try:
    loaded_model = joblib.load("best_ada_model.joblib")
    loaded_feature_names = joblib.load("feature_names.joblib")
    loaded_label_encoders = joblib.load("label_encoders.joblib")
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
        "Smoking": Smoking,
        "Age": Age,
        "Family Heart Disease": Family_Heart_Disease,
        "BMI": BMI,
        "Cholesterol Level": Cholesterol_Level,
        "Blood Pressure": Blood_Pressure,
        "Stress Level": Stress_Level,
        "Diabetes": Diabetes,
        "Homocysteine Level": Homocysteine_Level
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

    return "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

# -------------------- Session State --------------------
# Records and last prediction
if "records" not in st.session_state:
    st.session_state.records = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

# -------------------- Input Defaults --------------------
input_defaults = {
    "smoking": "No",
    "age": 45,
    "bmi": 25.0,
    "cholesterol_level": 200,
    "stress_level": "Low",
    "family_heart_disease": "No",
    "blood_pressure": 120,
    "diabetes": "No",
    "homocysteine_level": 10.0
}

# -------------------- UI Layout --------------------
st.title("Heart Disease Prediction App")
st.write("Predict heart disease risk using a trained machine learning model.")

st.header("Patient Information")

col1, col2 = st.columns(2)

# --- Column 1 ---
with col1:
    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"],
        index=0 if st.session_state.reset_trigger else ["No", "Yes"].index(st.session_state.get("smoking", input_defaults["smoking"])),
        key="smoking"
    )
    age = st.number_input(
        "Age",
        1, 120,
        value=input_defaults["age"] if st.session_state.reset_trigger else st.session_state.get("age", input_defaults["age"]),
        key="age"
    )
    bmi = st.number_input(
        "BMI",
        10.0, 60.0,
        step=0.1,
        value=input_defaults["bmi"] if st.session_state.reset_trigger else st.session_state.get("bmi", input_defaults["bmi"]),
        key="bmi"
    )
    cholesterol_level = st.number_input(
        "Cholesterol Level",
        100, 400,
        step=1,
        value=input_defaults["cholesterol_level"] if st.session_state.reset_trigger else st.session_state.get("cholesterol_level", input_defaults["cholesterol_level"]),
        key="cholesterol_level"
    )
    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"],
        index=0 if st.session_state.reset_trigger else ["Low","Medium","High"].index(st.session_state.get("stress_level", input_defaults["stress_level"])),
        key="stress_level"
    )

# --- Column 2 ---
with col2:
    family_heart_disease = st.selectbox(
        "Family Heart Disease",
        ["No", "Yes"],
        index=0 if st.session_state.reset_trigger else ["No","Yes"].index(st.session_state.get("family_heart_disease", input_defaults["family_heart_disease"])),
        key="family_heart_disease"
    )
    blood_pressure = st.number_input(
        "Blood Pressure",
        70, 250,
        step=1,
        value=input_defaults["blood_pressure"] if st.session_state.reset_trigger else st.session_state.get("blood_pressure", input_defaults["blood_pressure"]),
        key="blood_pressure"
    )
    diabetes = st.selectbox(
        "Diabetes",
        ["No", "Yes"],
        index=0 if st.session_state.reset_trigger else ["No","Yes"].index(st.session_state.get("diabetes", input_defaults["diabetes"])),
        key="diabetes"
    )
    homocysteine_level = st.number_input(
        "Homocysteine Level",
        2.0, 50.0,
        step=0.1,
        value=input_defaults["homocysteine_level"] if st.session_state.reset_trigger else st.session_state.get("homocysteine_level", input_defaults["homocysteine_level"]),
        key="homocysteine_level"
    )

# Clear reset flag after rendering
if st.session_state.reset_trigger:
    st.session_state.reset_trigger = False

st.markdown("---")

# -------------------- Action Buttons --------------------
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

        # Append to records
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
        st.session_state.reset_trigger = True
        st.session_state.last_result = None
        st.rerun()

# -------------------- Result Display --------------------
if st.session_state.last_result:
    st.subheader("Prediction Result")
    st.success(st.session_state.last_result)

# -------------------- Table Display & Excel Export --------------------
if st.session_state.records:
    st.subheader("Patient Records")

    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, use_container_width=True)

    # In-memory Excel export
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    st.download_button(
        label="Download Records (Excel)",
        data=buffer,
        file_name="patient_records.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
