# End-to-End Customer Churn Prediction System

## Overview

This project implements a complete, production-style machine learning pipeline for **customer churn prediction** using the IBM Telco Customer Churn dataset. The system covers the entire lifecycle of an ML solution — from data ingestion and validation to model training, evaluation, single-customer inference, and deployment using Streamlit.

The primary goal is to predict whether a customer is likely to churn based on demographic information, service usage, and billing details, while following strong ML engineering and software design practices.

---

## Problem Statement

Customer churn is a critical business problem for subscription-based services. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project aims to:

* Identify customers at high risk of churn
* Provide churn probability for proactive retention strategies
* Build a reusable and deployment-ready ML pipeline

---

## Key Features

* Schema-driven data validation
* Clear separation of raw and derived features
* Robust preprocessing pipeline
* Feature engineering based on domain insights
* Threshold-tuned classification for business needs
* Single-customer inference pipeline
* Streamlit-based web application
* Clean GitHub repository with proper version control hygiene

---

## Project Architecture

```
Customer-churn-prediction-app/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and train-test split
│   ├── data_validation.py      # Schema-based validation (training & inference)
│   ├── feature_engineering.py  # Derived feature creation
│   ├── preprocessing.py        # Scikit-learn preprocessing pipelines
│   ├── train.py                # Model training and artifact generation
│   ├── evaluate.py             # Model evaluation and metrics
│   └── predict.py              # Single-customer inference logic
│
├── data/
│   ├── raw/                    # Raw dataset (ignored in Git)
│
├── artifacts/
│   ├── feature_schema.json     # Feature and schema definition
│   ├── model.pkl               # Trained model (ignored in Git)
│   └── preprocessor.pkl        # Preprocessing pipeline (ignored in Git)
│
├── app.py                      # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

* **Source:** IBM Telco Customer Churn Dataset (Kaggle)
* **Target Variable:** `Churn`
* **Data Types:**

  * Numerical: tenure, monthly charges
  * Categorical: demographic, service, and billing features

Raw data is intentionally excluded from the repository to maintain data hygiene and reproducibility.

---

## Machine Learning Pipeline

### 1. Data Ingestion

* Loads raw CSV data
* Splits data into train and test sets using stratification

### 2. Data Validation

* Schema-driven validation using `feature_schema.json`
* Ensures required raw features are present
* Differentiates between training and inference validation

### 3. Feature Engineering

Derived features created based on domain knowledge:

* `service_count`: number of subscribed internet services
* `tenure_group`: binned customer tenure
* `has_internet`: internet availability flag

### 4. Preprocessing

* Numerical features: median imputation + robust scaling
* Categorical features: most-frequent imputation + one-hot encoding
* Implemented using `ColumnTransformer` and `Pipeline`

### 5. Model Training

* Classification model trained on processed features
* Evaluation metric: ROC-AUC
* Business-oriented decision threshold (optimized for recall)

### 6. Model Evaluation

* ROC-AUC
* Precision, Recall, F1-score
* Confusion Matrix

### 7. Inference

* Supports single-customer prediction
* Reuses trained preprocessing and feature engineering logic
* Outputs churn probability and label

---

## Streamlit Application

The Streamlit app provides:

* Interactive UI for customer input
* Real-time churn prediction
* Probability-based risk interpretation

Run the app locally using:

```bash
streamlit run app.py
```

---

## Installation and Setup

### Requirements File

Create a `requirements.txt` file at the project root with the following contents:

```
pandas
numpy
scikit-learn
joblib
streamlit
```

These dependencies are sufficient to run training, evaluation, inference, and the Streamlit application.

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Dataset

Download the IBM Telco Customer Churn dataset and place it in:

```
data/raw/
```

### 4. Train the Model

```bash
python -m src.train
```

### 5. Evaluate the Model

```bash
python -m src.evaluate
```

### 6. Run Streamlit App

```bash
streamlit run app.py
```

---

## Model Performance (Sample)

* ROC-AUC: ~0.84
* High recall to prioritize churn detection
* Suitable for customer retention use cases

---

## Version Control Practices

* Raw data excluded using `.gitignore`
* Model artifacts excluded to keep repository lightweight
* Logical commits for major milestones

---

## Author

Sri Harsha Vardhan Chikkala