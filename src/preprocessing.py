import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer


def load_schema(schema_path: str) -> dict:
    """
    Load feature schema from JSON.
    """
    with open(schema_path, "r") as f:
        return json.load(f)


def build_preprocessor(schema_path: str) -> ColumnTransformer:
    """
    Build preprocessing pipeline using schema.
    """
    schema = load_schema(schema_path)

    numerical_features = schema["numerical_features"]
    categorical_features = schema["categorical_features"]

    # Numerical pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor

