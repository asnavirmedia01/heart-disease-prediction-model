pip install streamlit pandas joblib
import streamlit as st
import pandas as pd
import joblib

# Load the best performing model and preprocessing artifacts
try:
    loaded_model = joblib.load('best_ada_model.joblib')
    loaded_feature_names = joblib.load('feature_names.joblib')
    loaded_label_encoders = joblib.load('label_encoders.joblib')
    # Note: loaded_scaler is not used here as the final AdaBoost model was trained on unscaled data.
except FileNotFoundError as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure that 'best_ada_model.joblib', 'feature_names.joblib', and 'label_encoders.joblib' are in the same directory as this script.")
    st.stop()

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
    """
    Predicts heart disease status based on input features using a pre-trained model.

    Args:
        Smoking (str): 'Yes' or 'No'
        Age (int): Patient's age
        Family_Heart_Disease (str): 'Yes' or 'No'
        BMI (float): Body Mass Index
        Cholesterol_Level (int): Cholesterol level
        Blood_Pressure (int): Blood pressure
        Stress_Level (str): 'High', 'Low', or 'Medium'
        Diabetes (str): 'Yes' or 'No'
        Homocysteine_Level (float): Homocysteine level

    Returns:
        str: A string indicating the prediction result.
    """

    # Prepare a dictionary for the input values, matching the expected column names
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

    # Create a DataFrame from the single row of input data
    input_df_processed = pd.DataFrame([input_raw_data])

    # Encode categorical features using the loaded encoders
    for col, encoder in loaded_label_encoders.items():
        if col in input_df_processed.columns:
            try:
                input_df_processed[col] = encoder.transform(input_df_processed[col])
            except ValueError:
                st.error(f"Invalid value for '{col}'. Please check the input.")
                return "Error: Invalid input"

    # Select and order features according to the `loaded_feature_names`
    input_df = input_df_processed[loaded_feature_names]

    # Make prediction
    prediction = loaded_model.predict(input_df)[0]

    # Convert prediction to user-friendly string
    if prediction == 1:
        result = 'Heart Disease Detected'
    else:
        result = 'No Heart Disease'

    return result

# Streamlit UI
st.title('Heart Disease Prediction App')
st.write('Enter patient details to predict the likelihood of heart disease.')

# Input widgets
with st.sidebar:
    st.header('Patient Information')
    smoking = st.selectbox('Smoking', ['No', 'Yes'])
    age = st.slider('Age', min_value=1, max_value=90, value=45)
    family_heart_disease = st.selectbox('Family Heart Disease', ['No', 'Yes'])
    bmi = st.slider('BMI', min_value=18.0, max_value=35.0, value=25.0, step=0.1)
    cholesterol_level = st.slider('Cholesterol Level', min_value=150, max_value=319, value=200)
    blood_pressure = st.slider('Blood Pressure', min_value=90, max_value=179, value=120)
    stress_level = st.selectbox('Stress Level', ['Low', 'Medium', 'High'])
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
    homocysteine_level = st.slider('Homocysteine Level', min_value=5.0, max_value=19.99, value=10.0, step=0.01)

if st.button('Predict Heart Disease'):
    prediction_result = predict_heart_disease(
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

    if "Error" in prediction_result:
        st.error(prediction_result)
    else:
        st.success(f'Prediction: {prediction_result}')

# Display the loaded feature names as a reference (optional)
st.subheader('Features used for Prediction (and their order):')
st.write(loaded_feature_names)
