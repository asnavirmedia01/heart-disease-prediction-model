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

## Problem Statement
Heart disease remains a leading cause of mortality worldwide. Early and accurate prediction of heart disease is crucial for timely intervention and improved patient outcomes. This project aims to develop a machine learning model to predict the presence of heart disease based on various health indicators and lifestyle factors.

## Dataset
The dataset used in this project contains various health-related features and a target variable indicating the presence or absence of heart disease. It includes demographic information, vital signs, and laboratory results.

Here's a glimpse of the data:

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
3                  82                  136       8.66               18.38   
4                 296                  147       9.46               15.51   

  Heart Disease Status  
0                  Yes  
1                   No  
2                   No  
3                   No  
4                   No  
```

## Exploratory Data Analysis (EDA) & Preprocessing
-   **Missing Values**: The dataset was checked for missing values, and none were found.
-   **Duplicate Rows**: No duplicate rows were identified.
-   **Outliers**: Outliers were analyzed using IQR method and boxplots, revealing no significant outliers across numerical features.
-   **Data Distribution**: Histograms were generated for numerical features to understand their distributions. Count plots were used for categorical features.
-   **Correlation Analysis**: A correlation heatmap was generated to visualize relationships between numerical features, identifying potential multicollinearity and feature importance.
-   **Target Distribution**: The distribution of 'Heart Disease Status' was analyzed, showing a relatively balanced dataset.
-   **Categorical Encoding**: All categorical columns were encoded using `LabelEncoder` to convert them into numerical representations suitable for machine learning models. The mapping for each encoding is explicitly shown.

## Feature Engineering & Selection
-   **Feature Importance**: Decision Tree Classifier's feature importances were calculated to identify the most influential features. The following features were identified as important:
    -   `Smoking`
    -   `Age`
    -   `Family Heart Disease`
    -   `BMI`
    -   `Cholesterol Level`
    -   `Blood Pressure`
    -   `Stress Level`
    -   `Diabetes`
    -   `Homocysteine Level`
-   **Dimensionality Reduction**: The dataset was reduced to include only these important features for model training, aiming to improve model performance and reduce complexity.

## Model Training and Evaluation
Several classification models were trained and evaluated on the preprocessed and scaled dataset to predict heart disease. The models included:
-   Decision Tree Classifier (with hyperparameter tuning using GridSearchCV)
-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Support Vector Machine (SVM)
-   Random Forest Classifier
-   Gradient Boosting Classifier
-   AdaBoost Classifier
-   Gaussian Naive Bayes
-   XGBoost Classifier
-   LightGBM Classifier

Models were evaluated using metrics such as accuracy, recall (especially for class 1 - 'Heart Disease Detected'), precision, and F1-score. ROC curves and AUC scores were also generated for each model to assess their discriminative power.

## Best Model & Performance
Based on the evaluation, **AdaBoost Classifier** emerged as the best-performing model, achieving a perfect score across accuracy and recall for class 1 on the test set. This indicates its strong capability in correctly identifying individuals with heart disease.

**Confusion Matrix for AdaBoost (Best Model):**

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       154
           1       1.00      1.00      1.00       146

    accuracy                           1.00       300
   macro avg       1.00      1.00      1.00       300
weighted avg       1.00      1.00      1.00       300
```

This high performance was achieved after feature selection, which focused the models on the most impactful features.

## Deployment with Streamlit
A Streamlit application `app_streamlit.py` has been developed to provide a user-friendly interface for predicting heart disease. This app allows users to input patient information and receive instant predictions. The application utilizes the best-performing AdaBoost model and the saved preprocessing artifacts (label encoders and feature names) to ensure consistent predictions.

## How to Run the Project
To run this project and the Streamlit application, follow these steps:

1.  **Clone the Repository (or download the notebook and files)**:
    ```bash
    git clone <repository_url>
    cd heart-disease-prediction
    ```

2.  **Ensure you have the necessary files**: Make sure you have the trained model (`best_ada_model.joblib`), feature names (`feature_names.joblib`), and label encoders (`label_encoders.joblib`) in the same directory as `app_streamlit.py`. These files are generated by running the corresponding cells in the Jupyter notebook.

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (A `requirements.txt` file should contain: `pandas`, `scikit-learn`, `streamlit`, `joblib`, `openpyxl`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`)

4.  **Run the Streamlit Application**:
    ```bash
    https://heart-disease-prediction-model-fcguwhrgskihrigrjrpddx.streamlit.app/
    ```
    This will open the application in your web browser.

## Conclusion
This project successfully developed a robust machine learning pipeline for heart disease prediction, from data loading and comprehensive EDA to model training, evaluation, and deployment. The AdaBoost model demonstrated exceptional performance, making it a valuable tool for assisting in early risk assessment. The Streamlit application provides an accessible way for potential users to interact with the model.
