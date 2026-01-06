import joblib
import pandas as pd

from src.data_validation import validate_dataframe, load_schema
from src.feature_engineering import apply_feature_engineering


ARTIFACT_DIR = "artifacts"
SCHEMA_PATH = "artifacts/feature_schema.json"


def predict_single_customer(customer_data: dict, threshold: float = 0.4):
    """
    Predict churn for a single customer.
    """
   
    preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.pkl")
    model = joblib.load(f"{ARTIFACT_DIR}/model.pkl")

    df = pd.DataFrame([customer_data])

    df = validate_dataframe(df, SCHEMA_PATH, require_target=False)


    
    df = apply_feature_engineering(df)

    X_processed = preprocessor.transform(df)

    churn_proba = model.predict_proba(X_processed)[0][1]
    churn_label = "Churn" if churn_proba >= threshold else "No Churn"

    return {
        "churn_probability": round(float(churn_proba), 4),
        "prediction": churn_label
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5
    }

    result = predict_single_customer(sample_customer)
    print(result)
