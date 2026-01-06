print("evaluate.py started")

import joblib
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_loader import load_data, split_data
from src.data_validation import validate_dataframe, load_schema
from src.feature_engineering import apply_feature_engineering


ARTIFACT_DIR = "artifacts"
DATA_PATH = "data/raw/telco_churn.csv"
SCHEMA_PATH = "artifacts/feature_schema.json"


def evaluate(threshold: float = 0.5):
  
    preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.pkl")
    model = joblib.load(f"{ARTIFACT_DIR}/model.pkl")

    df = load_data(DATA_PATH)
    df = validate_dataframe(df, SCHEMA_PATH)
    df = apply_feature_engineering(df)

    schema = load_schema(SCHEMA_PATH)
    target_col = schema["target"]

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=target_col
    )

 
    X_test_processed = preprocessor.transform(X_test)

 
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_true = (y_test == "Yes").astype(int)

    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

   
    print(f"Evaluation Threshold: {threshold}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate(threshold=0.4)
