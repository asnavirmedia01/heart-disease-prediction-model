# heart-disease-prediction-model
https://heart-disease-prediction-model-fcguwhrgskihrigrjrpddx.streamlit.app/
 Heart Disease Prediction Model
Predicts the likelihood of heart disease using clinical and demographic features. Designed as a risk assessment prototype, not a diagnostic tool.
Overview
This project implements a machine learning pipeline for heart disease risk prediction, including:
Data preprocessing and cleaning
Feature engineering and encoding
Model training, evaluation, and comparison
Deployment via a web interface (Streamlit/Gradio)
The goal is reproducible, interpretable, and responsible ML for clinical research and educational purposes.
Problem Statement
Heart disease is a leading cause of mortality globally. Early risk identification allows preventive intervention and improved clinical decision-making.
This project frames heart disease prediction as a binary classification problem. The model predicts whether a patient is likely to have heart disease based on clinical and demographic features.
⚠️ Disclaimer: This model is not a medical diagnostic tool and should not be used to make real-world clinical decisions.
Dataset
The dataset includes structured clinical and demographic variables commonly used in cardiovascular research:
Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, electrocardiogram results, maximum heart rate, exercise-induced angina, and more.
Key preprocessing steps:
Missing value handling
Encoding of categorical variables
Feature scaling where applicable
Handling class imbalance
The dataset exhibits class imbalance, which influenced model selection and evaluation metrics.
Model Training & Evaluation
Several algorithms were evaluated, including linear and ensemble-based models. AdaBoost was selected as the final model due to its balanced performance across key metrics.
Metrics used for evaluation:
Metric
Value
Accuracy
XX%
Precision
XX%
Recall
XX%
ROC-AUC
XX%
Focus on recall: Minimizing false negatives is critical in heart disease risk prediction, as missing high-risk patients has serious implications.
Explainability
Feature importance analysis was performed to improve model interpretability. Key predictors were analyzed to ensure alignment with known clinical risk factors.
Model transparency is essential for building trust in health-related ML applications.
Deployment
The model is deployed via a web interface:
All model artifacts (.joblib, encoders, feature metadata) must reside in the same directory as the application script.
Dependencies are specified in requirements.txt.
Designed for demonstration and research purposes only.
Common issues documented: missing model files, path errors, or dependency conflicts.
Reproducibility
Random seeds fixed during training
All dependencies versioned in requirements.txt
Model artifacts included to guarantee consistent inference
Ethical Considerations
Dataset may not represent all populations equally
Predictions may reflect training data biases
Model does not account for all clinical variables used in professional diagnosis
Intended Use:
Academic research, educational demonstration, and ML experimentation
Misuse Risks:
Clinical diagnosis or self-diagnosis by patients
Any real-world treatment decisions