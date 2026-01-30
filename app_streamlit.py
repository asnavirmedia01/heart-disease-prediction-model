import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import numpy as np

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    layout="centered"
)

# -------------------- Load Model --------------------
try:
    model = joblib.load("best_ada_model.joblib")
    feature_names = joblib.load("feature_names.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
except FileNotFoundError as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# -------------------- Prediction Function --------------------
def predict_with_probability(input_data: dict):
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    df = df[feature_names]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # Probability of heart disease

    return prediction, probability

# -------------------- Session State --------------------
if "records" not in st.session_state:
    st.session_state.records = []

# -------------------- UI --------------------
st.title("Heart Disease Prediction App")
st.write(
    "This application estimates heart disease risk using a trained AdaBoost model. "
    "Results are probabilistic and intended for educational purposes only."
)

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    age = st.number_input("Age", 1, 120, value=45)

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        10.0, 60.0,
        step=0.1,
        value=25.0
    )
    st.caption("BMI = weight (kg) / height² (m²)")

    cholesterol = st.number_input("Cholesterol Level", 100, 400, value=200)
    stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

with col2:
    family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    blood_pressure = st.number_input("Blood Pressure", 70, 250, value=120)
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    homocysteine = st.number_input(
        "Homocysteine Level",
        2.0, 50.0,
        step=0.1,
        value=10.0
    )

st.markdown("---")

# -------------------- Predict Button --------------------
if st.button("Predict Risk"):
    input_data = {
        "Smoking": smoking,
        "Age": age,
        "Family Heart Disease": family_history,
        "BMI": bmi,
        "Cholesterol Level": cholesterol,
        "Blood Pressure": blood_pressure,
        "Stress Level": stress,
        "Diabetes": diabetes,
        "Homocysteine Level": homocysteine
    }

    pred, prob = predict_with_probability(input_data)
    risk_percent = prob * 100

    # -------------------- Risk Band Logic --------------------
    st.subheader("Prediction Result")

    if risk_percent < 40:
        st.success(f"Low Risk of Heart Disease ({risk_percent:.2f}% confidence)")
        risk_label = "Low Risk"
    elif 40 <= risk_percent <= 70:
        st.warning(f"Borderline Risk of Heart Disease ({risk_percent:.2f}% confidence)")
        risk_label = "Borderline Risk"
    else:
        st.error(f"High Risk of Heart Disease ({risk_percent:.2f}% confidence)")
        risk_label = "High Risk"

    # Save record
    st.session_state.records.append({
        **input_data,
        "Risk Level": risk_label,
        "Risk Probability (%)": round(risk_percent, 2)
    })

    # -------------------- Feature Importance --------------------
    st.markdown("---")
    st.subheader("Top Contributing Features (Model-Level)")

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(6)

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.caption(
        "Feature importance reflects the overall influence of features in the trained model, "
        "not individual patient-specific explanations."
    )

# -------------------- Records & Excel Export --------------------
if st.session_state.records:
    st.markdown("---")
    st.subheader("Patient Records")

    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, use_container_width=True)

    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    st.download_button(
        label="Download Records (Excel)",
        data=buffer,
        file_name="patient_records.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
