import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data_loader import load_data, split_data
from src.data_validation import validate_dataframe, load_schema
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import build_preprocessor


ARTIFACT_DIR = "artifacts"
DATA_PATH = "data/raw/telco_churn.csv"

SCHEMA_PATH = "artifacts/feature_schema.json"


def train():
   
    df = load_data(DATA_PATH)
    df = validate_dataframe(df, SCHEMA_PATH)

    df = apply_feature_engineering(df)

    schema = load_schema(SCHEMA_PATH)
    target_col = schema["target"]

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=target_col
    )


    preprocessor = build_preprocessor(SCHEMA_PATH)


    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(
        (y_test == "Yes").astype(int),
        y_pred_proba
    )

    print(f"ROC-AUC on test set: {roc_auc:.4f}")

  
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    joblib.dump(
        pipeline.named_steps["preprocessor"],
        f"{ARTIFACT_DIR}/preprocessor.pkl"
    )
    joblib.dump(
        pipeline.named_steps["model"],
        f"{ARTIFACT_DIR}/model.pkl"
    )

    print("Artifacts saved successfully.")


if __name__ == "__main__":
    train()
