<img width="1365" height="647" alt="Screenshot 2026-05-06 040714" src="https://github.com/user-attachments/assets/faed5048-7db2-4d80-927a-729182aebdc8" />

# Heart Disease Prediction Project

## 📝 Project Overview

This project aims to develop a machine learning model capable of predicting the likelihood of heart disease in individuals based on various health and lifestyle factors. By leveraging a comprehensive dataset containing patient information, we perform exploratory data analysis, preprocess the data, and train several classification models to identify the most effective predictor. The ultimate goal is to provide a robust tool for early risk assessment, which can assist healthcare professionals in identifying at-risk individuals and guiding preventive measures.

## 📊 Dataset

The dataset used in this project contains detailed health records for patients, including demographic information, physiological measurements, and lifestyle habits. It comprises `1500` entries and `16` features, with `Heart Disease Status` as the target variable.

Here's a glimpse of the dataset:

```
   Age  Gender  Blood Pressure  Cholesterol Level Exercise Habits Smoking  \
0   69    Male             110                269        Moderate     Yes   
1   76    Male              91                223        Moderate      No   
2   48  Female             117                187            High      No   
3   50  Female             137                304        Moderate      No   
4   61  Female              91                276            High      No   

  Family Heart Disease Diabetes   BMI Stress Level  Sleep Hours  \
0                  Yes       No  23.9          Low          8.9   
1                   No       No  28.4         High          7.8   
2                  Yes      Yes  33.9          Low          8.2   
3                   No      Yes  28.1          Low          6.6   
4                   No      Yes  18.9       Medium          6.6   

   Triglyceride Level  Fasting Blood Sugar  CRP Level  Homocysteine Level  \
0                 293                  116       6.23               15.75   
1                 134                   72       8.36               11.83   
2                 124                  111       1.62               12.52   
3                  82                  136       8.66               18.38   n4                 296                  147       9.46               15.51   

  Heart Disease Status  
0                  Yes  
1                   No  
2                   No  
3                   No  
4                   No  
```

### Key Features

*   **Demographic**: `Age`, `Gender`
*   **Physiological**: `Blood Pressure`, `Cholesterol Level`, `BMI`, `Triglyceride Level`, `Fasting Blood Sugar`, `CRP Level`, `Homocysteine Level`
*   **Lifestyle**: `Exercise Habits`, `Smoking`, `Sleep Hours`, `Stress Level`
*   **Medical History**: `Family Heart Disease`, `Diabetes`
*   **Target Variable**: `Heart Disease Status`

## 🛠️ Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration**: The dataset was loaded and initial checks for missing values, duplicates, and data types were performed.
2.  **Exploratory Data Analysis (EDA)**: Visualizations (histograms, boxplots, count plots, heatmaps) were used to understand feature distributions, relationships, and potential outliers. Correlation analysis helped in identifying highly correlated features and feature importance.
3.  **Data Preprocessing**: Categorical features were encoded using `LabelEncoder`. The dataset was split into training and testing sets. Feature scaling was applied to numerical features using `StandardScaler` for some models, but ultimately, the model trained on unscaled, feature-reduced data performed best.
4.  **Model Training and Evaluation**: Several classification algorithms were trained and evaluated, including:
    *   Logistic Regression
    *   K-Nearest Neighbors (KNN)
    *   Support Vector Machine (SVM)
    *   Decision Tree Classifier (with hyperparameter tuning via GridSearchCV)
    *   Random Forest
    *   Gradient Boosting
    *   AdaBoost
    *   XGBoost
    *   LightGBM

    Models were evaluated based on metrics like Accuracy, Recall, Precision, F1-Score, and ROC AUC, with a particular focus on **Recall for the positive class (Heart Disease Detected)** due to the nature of the problem (minimizing false negatives).
5.  **Feature Selection**: Based on feature importance analysis from a preliminary Decision Tree model, a subset of the most impactful features was selected to reduce model complexity and potentially improve performance.
6. **Model Selection**: AdaBoost was identified as the best performing model on the feature-reduced dataset.

## 🚀 Results

After comprehensive evaluation, the **AdaBoost Classifier** emerged as the top-performing model, achieving outstanding metrics on the test set when trained on the reduced feature set (features with importance > 0.01). The model was able to achieve **100% Accuracy and Recall for the 'Heart Disease Detected' class**.

### Best Model Performance (AdaBoost)

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       154
           1       1.00      1.00      1.00       146

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300
```


## 📦 Deployment

A Streamlit application `app_streamlit.py` has been created to provide an interactive interface for predicting heart disease. This application allows users to input patient details and receive an immediate risk assessment.

### How to Run the Streamlit App

1.  **Download necessary files**: Download `app_streamlit.py`, `best_ada_model.joblib`, `feature_names.joblib`, and `label_encoders.joblib` from the Colab environment.
2.  **Install Streamlit**: If you don't have Streamlit installed, open your terminal or command prompt and run:
    ```bash
    pip install streamlit
    ```
3.  **Place files**: Ensure all downloaded files are in the same directory.
4.  **Run the app**: Navigate to that directory in your terminal and execute:
    ```bash
    https://heart-disease-prediction-model-fcguwhrgskihrigrjrpddx.streamlit.app/
    ```

This will open the application in your web browser, allowing you to interact with the Heart Disease Prediction model.
