<img width="1365" height="647" alt="588237101-faed5048-7db2-4d80-927a-729182aebdc8" src="https://github.com/user-attachments/assets/7a1516ef-0f93-45b3-a872-3f6e0b0d5ce3" />

# Heart Disease Prediction Project

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis and Preprocessing](#exploratory-data-analysis-and-preprocessing)
4. [Feature Engineering and Selection](#feature-engineering-and-selection)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Best Model and Performance](#best-model-and-performance)
7. [Deployment with Streamlit](#deployment-with-streamlit)
8. [How to Run the Project](#how-to-run-the-project)
9. [Conclusion](#conclusion)

---

## Problem Statement
Heart disease remains a leading cause of mortality worldwide. Early and accurate prediction of heart disease is crucial for timely intervention and improved patient outcomes. This project aims to develop a machine learning model to predict the presence of heart disease based on various health indicators and lifestyle factors.

---

## Dataset
The dataset used in this project contains various health-related features and a target variable indicating the presence or absence of heart disease. It includes demographic information, vital signs, and laboratory results.
### Dataset Source

```text
https://github.com/asnavirmedia01/heart-disease-prediction-model/blob/a9cd2451031e17c08593e80fb440e5e13e5f1cf3/heart_disease_dataset.csv
```
Here's a glimpse of the data:

```python
   Age  Gender  Blood Pressure  Cholesterol Level Exercise Habits Smoking
0   69    Male             110                269        Moderate     Yes
1   76    Male              91                223        Moderate      No
2   48  Female             117                187            High      No
3   50  Female             137                304        Moderate      No
4   61  Female              91                276            High      No

  Family Heart Disease Diabetes   BMI Stress Level  Sleep Hours
0                  Yes       No  23.9          Low          8.9
1                   No       No  28.4         High          7.8
2                  Yes      Yes  33.9          Low          8.2
3                   No      Yes  28.1          Low          6.6
4                   No      Yes  18.9       Medium          6.6

   Triglyceride Level  Fasting Blood Sugar  CRP Level  Homocysteine Level
0                 293                  116       6.23               15.75
1                 134                   72       8.36               11.83
2                 124                  111       1.62               12.52
3                  82                  136       8.66               18.38
4                 296                  147       9.46               15.51

  Heart Disease Status
0                  Yes
1                   No
2                   No
3                   No
4                   No
```

---

## Exploratory Data Analysis and Preprocessing

- **Missing Values**: The dataset was checked for missing values, and none were found.
- **Duplicate Rows**: No duplicate rows were identified.
- **Outliers**: Outliers were analyzed using the IQR method and boxplots, revealing no significant outliers across numerical features.
- **Data Distribution**: Histograms were generated for numerical features to understand their distributions. Count plots were used for categorical features.
- **Correlation Analysis**: A correlation heatmap was generated to visualize relationships between numerical features and identify potential multicollinearity.
- **Target Distribution**: The distribution of `Heart Disease Status` was analyzed, showing a relatively balanced dataset.
- **Categorical Encoding**: All categorical columns were encoded using `LabelEncoder` to convert them into numerical representations suitable for machine learning models.
- **Train-Test Split**:
  - `80%` training data
  - `20%` testing data
  - `stratify=y`
  - `random_state=42`
- **Feature Scaling**:
  - `StandardScaler` was applied after the train-test split.
  - The scaler was fitted only on the training data to prevent data leakage.

---

## Feature Engineering and Selection

- **Feature Importance**:
  The Decision Tree Classifier identified the following important features:

  - `Smoking`
  - `Age`
  - `Family Heart Disease`
  - `BMI`
  - `Cholesterol Level`
  - `Blood Pressure`
  - `Stress Level`
  - `Diabetes`
  - `Homocysteine Level`

- **Dimensionality Reduction**:
  Only the most important features were selected for model training to reduce complexity and improve efficiency.

---

## Model Training and Evaluation

Several machine learning models were trained and evaluated:

- Decision Tree Classifier
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Gaussian Naive Bayes
- XGBoost Classifier
- LightGBM Classifier

### Cross Validation
- `GridSearchCV`
- `cv=5`

### Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score

---

## Best Model and Performance

The **AdaBoost Classifier** achieved the best overall performance on the test dataset.

### Classification Report

```python
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       154
           1       1.00      1.00      1.00       146

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300
```

> Note:
> Extremely high accuracy in medical prediction tasks may indicate possible overfitting or data leakage. Additional validation on external datasets is recommended.

---

## Deployment with Streamlit

A Streamlit application `app_streamlit.py` was developed to provide a user-friendly interface for heart disease prediction.

The application:
- Accepts patient health information
- Processes the data using saved preprocessing artifacts
- Uses the trained AdaBoost model for prediction
- Displays prediction results instantly

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <(https://github.com/asnavirmedia01/heart-disease-prediction-model.git)>
cd heart-disease-prediction-model

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Required Files

Ensure these files exist in the project directory:

- `best_ada_model.joblib`
- `feature_names.joblib`
- `label_encoders.joblib`

### 4. Run the Streamlit Application

```bash
streamlit run app_streamlit.py
```

### 5. Open the Application

After running the command, Streamlit will generate a local URL such as:

```bash
http://localhost:8501
```

Or use the deployed version:

```text
https://heart-disease-prediction-model-fcguwhrgskihrigrjrpddx.streamlit.app/
```

---

## Conclusion

This project developed a complete machine learning pipeline for heart disease prediction, including:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Model training
- Evaluation
- Streamlit deployment

The AdaBoost model achieved the strongest performance among all evaluated models and was integrated into a Streamlit application for interactive prediction.
