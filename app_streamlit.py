import streamlit as st
import pandas as pd
import joblib

# Load the best performing model and preprocessing artifacts
try:
    loaded_model = joblib.load('best_ada_model.joblib')
    loaded_feature_names = joblib.load('feature_names.joblib')
    loaded_label_encoders = joblib.load('label_encoders.joblib')
    # Note: loaded_scaler is not used here as the final AdaBoost model was trained on unscaled data.
    # If the model was scaled, we would need to load the scaler here:
    # loaded_scaler = joblib.load('scaler.joblib')
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
        if col in input_df_processed.columns: # Only encode columns present in the input
            try:
                input_df_processed[col] = encoder.transform(input_df_processed[col])
            except ValueError:
                st.error(f"Invalid value for '{col}'. Please check the input. Expected values for {col}: {encoder.classes_}")
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

# --- Streamlit UI Design ---

# Custom CSS for a medical aesthetic
st.markdown(
    """
    <style>
    .reportview-container .main {
        background-color: #f8f9fa; /* Light grey background */
    }
    .stApp {
        max-width: 800px; /* Limit width for better mobile viewing */
        margin: auto;
        padding: 1rem; /* Generous spacing */
    }
    h1 {
        color: #007bff; /* Medical blue for title */
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stMarkdown p {
        text-align: center;
        font-size: 1.1em;
        color: #495057;
        margin-bottom: 2rem;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div>div, .stNumberInput>div>div>input {
        border-radius: 8px; /* Rounded input fields */
        border: 1px solid #ced4da; /* Soft border */
        padding: 0.5rem 1rem;
        box-shadow: none;
    }
    .stButton>button {
        background-color: #28a745; /* Green for predict button */
        color: white;
        border-radius: 10px;
        font-size: 1.2em;
        padding: 0.75rem 1.5rem;
        border: none;
        width: 100%; /* Make button span full width */
        margin-top: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Soft shadow */
    }
    .stButton>button:hover {
        background-color: #218838; /* Darker green on hover */
    }
    .stAlert {
        border-radius: 10px; /* Rounded alert cards */
        font-size: 1.1em;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Heart Disease Prediction App')
st.markdown(
    """
    <p>
    Enter the patient's information below to get an instant prediction on the likelihood of heart disease.
    This tool uses an advanced machine learning model to assist healthcare professionals.
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---") # Visual separator

# Input widgets organized in two columns
col1, col2 = st.columns(2)

with col1:
    smoking = st.selectbox('Smoking', ['No', 'Yes'], key='smoking')
    age = st.slider('Age', min_value=1, max_value=90, value=45, key='age')
    family_heart_disease = st.selectbox('Family Heart Disease', ['No', 'Yes'], key='family_heart_disease')
    bmi = st.slider('BMI', min_value=18.0, max_value=35.0, value=25.0, step=0.1, key='bmi')
    cholesterol_level = st.slider('Cholesterol Level', min_value=150, max_value=319, value=200, key='cholesterol_level')

with col2:
    blood_pressure = st.slider('Blood Pressure', min_value=90, max_value=179, value=120, key='blood_pressure')
    stress_level = st.selectbox('Stress Level', ['Low', 'Medium', 'High'], key='stress_level')
    diabetes = st.selectbox('Diabetes', ['No', 'Yes'], key='diabetes')
    homocysteine_level = st.slider('Homocysteine Level', min_value=5.0, max_value=19.99, value=10.0, step=0.01, key='homocysteine_level')

# Predict button
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

    # Display prediction result in a highlighted card
    if "Error" in prediction_result:
        st.error(prediction_result)
    else:
        if prediction_result == 'Heart Disease Detected':
            st.error(f'**Result:** {prediction_result} 🚨') # Use st.error for a "bad" result
        else:
            st.success(f'**Result:** {prediction_result} ✅') # Use st.success for a "good" result
