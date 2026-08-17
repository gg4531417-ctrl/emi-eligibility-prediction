# EMI Eligibility Prediction System

A machine learning--based web application that predicts an applicant's
**EMI eligibility** and estimates their **maximum monthly EMI** using
trained classification and regression models.

The project includes model evaluation, MLflow experiment tracking,
serialized model artifacts, a Streamlit application, GitHub version
control, and public Streamlit deployment.

## 🚀 Live Application

**Live Demo:** https://emi-eligibility-4531417.streamlit.app/

The deployed application is available through HTTPS and can be used to
enter applicant information and view the prediction result.

------------------------------------------------------------------------

## 📌 Project Overview

The **EMI Eligibility Prediction System** is designed as an end-to-end
machine learning project with two prediction tasks:

1.  **Classification** -- predicts the applicant's EMI eligibility
    category.
2.  **Regression** -- predicts the applicant's maximum monthly EMI
    amount.

The classification model produces the following classes:

-   `Eligible`
-   `High_Risk`
-   `Not_Eligible`

The application presents the prediction in a simple Streamlit interface.

### Example Prediction

For a tested applicant, the deployed application produced:

-   **EMI Eligibility:** Eligible
-   **Predicted Maximum Monthly EMI:** ₹24,424.48

> The displayed prediction depends on the applicant inputs supplied to
> the model.

------------------------------------------------------------------------

## 🏗️ System Architecture

``` text
                    Applicant Input
                           │
                           ▼
                  Streamlit Web App
                           │
                           ▼
                     Preprocessor
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
       Classification Model   Regression Model
                 │                   │
                 ▼                   ▼
        EMI Eligibility       Maximum Monthly EMI
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Prediction Result
```

------------------------------------------------------------------------

## 🤖 Machine Learning Models

Three classification models and three regression models were tracked
using MLflow.

### Classification Models

  Model                          F1 Score
  -------------------------- ------------
  Logistic Regression              0.8846
  Random Forest Classifier     **0.9434**
  SGD Classifier                   0.9183

**Best Classification Model:** Random Forest Classifier

### Regression Models

  Model                         R² Score
  ------------------------- ------------
  Ridge Regression                0.7476
  Random Forest Regressor     **0.9800**
  SGD Regressor                   0.7473

**Best Regression Model:** Random Forest Regressor

> The metrics above are taken from the project's MLflow tracking output.

------------------------------------------------------------------------

## 📊 MLflow Experiment Tracking

The project uses **MLflow** for experiment tracking.

### Experiment

``` text
EMI_Eligibility_Prediction
```

The MLflow experiment tracks:

### Classification

-   Model type
-   Task
-   Accuracy
-   Precision
-   Recall
-   F1 score
-   Trained model artifact

### Regression

-   Model type
-   Task
-   MAE
-   RMSE
-   R² score
-   Trained model artifact

All six candidate models were logged, along with the selected best
classification and regression models.

------------------------------------------------------------------------

## 📁 Project Structure

``` text
EMI_Prediction_App/
│
├── app.py
├── classification_model.pkl
├── regression_model.pkl
├── preprocessor.pkl
├── emi_final_predictions.csv
├── requirements.txt
├── .gitattributes
└── README.md
```

### File Description

  File                          Purpose
  ----------------------------- ------------------------------------------------
  `app.py`                      Streamlit application and prediction interface
  `classification_model.pkl`    Saved classification model
  `regression_model.pkl`        Saved regression model
  `preprocessor.pkl`            Saved preprocessing pipeline/object
  `emi_final_predictions.csv`   Final EMI prediction output data
  `requirements.txt`            Python dependencies required by the project
  `.gitattributes`              Git repository file attributes
  `README.md`                   Project documentation

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   **Python**
-   **Pandas**
-   **Scikit-learn**
-   **MLflow**
-   **Streamlit**
-   **Git**
-   **GitHub**
-   **Pickle (`.pkl`) model artifacts**

------------------------------------------------------------------------

## ⚙️ How to Run the Application Locally

### 1. Clone the repository

``` bash
git clone https://github.com/gg4531417-ctrl/emi-eligibility-prediction.git
```

### 2. Open the project directory

``` bash
cd emi-eligibility-prediction
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

``` bash
streamlit run app.py
```

### 5. Open the local application

Streamlit will display a local URL similar to:

``` text
http://localhost:8501
```

Open that address in your browser.

------------------------------------------------------------------------

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Deployment Flow

``` text
Local Project
     │
     ▼
   Git
     │
     ▼
  GitHub Repository
     │
     ▼
Streamlit Deployment
     │
     ▼
Public HTTPS Application
```

### Live URL

https://emi-eligibility-4531417.streamlit.app/

------------------------------------------------------------------------

## 🔬 Prediction Workflow

The application follows this general workflow:

1.  The user enters applicant information.
2.  The Streamlit application collects the input.
3.  The saved preprocessing object transforms the input.
4.  The classification model predicts the EMI eligibility class.
5.  The regression model predicts the maximum monthly EMI.
6.  The application displays both results to the user.

------------------------------------------------------------------------

## 📈 Model Selection

The project evaluates multiple algorithms rather than relying on a
single model.

For classification, the tracked F1 scores show that the **Random Forest
Classifier** achieved the highest F1 score among the three evaluated
models:

``` text
Random Forest Classifier → F1 = 0.9434
```

For regression, the **Random Forest Regressor** achieved the highest R²
score:

``` text
Random Forest Regressor → R² = 0.9800
```

These models were therefore selected as the best-performing
classification and regression models in the tracked experiment.

------------------------------------------------------------------------

## 🧪 MLflow Run Summary

The MLflow experiment contains runs for:

### Classification

``` text
Classification_Logistic Regression
Classification_Random Forest Classifier
Classification_SGD Classifier
BEST_Classification_Model
```

### Regression

``` text
Regression_Ridge Regression
Regression_Random Forest Regressor
Regression_SGD Regressor
BEST_Regression_Model
```

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Build an EMI eligibility prediction system.
-   Compare multiple classification algorithms.
-   Compare multiple regression algorithms.
-   Track experiments and metrics using MLflow.
-   Save trained models for application use.
-   Build an interactive Streamlit application.
-   Deploy the application as a public HTTPS web application.
-   Demonstrate an end-to-end machine learning workflow.

------------------------------------------------------------------------

## 🔐 Notes

This project is intended as a **machine learning project/demo
application**.

Predictions generated by the application should not be treated as
financial, lending, or credit decisions without appropriate validation,
regulatory review, and domain-specific controls.

------------------------------------------------------------------------

## 👨‍💻 Author

**Gopi Chand B**

GitHub: https://github.com/gg4531417-ctrl

------------------------------------------------------------------------

## ⭐ Project Highlights

-   ✅ End-to-end machine learning workflow
-   ✅ Classification and regression
-   ✅ Multiple model comparison
-   ✅ MLflow experiment tracking
-   ✅ Model artifact persistence
-   ✅ Interactive Streamlit application
-   ✅ GitHub version control
-   ✅ Public HTTPS deployment
-   ✅ Best classification model: Random Forest Classifier
-   ✅ Best regression model: Random Forest Regressor

------------------------------------------------------------------------

## 📄 License

No license has been specified for this repository.

If this project is intended for public reuse, add an appropriate
open-source license to the repository.
