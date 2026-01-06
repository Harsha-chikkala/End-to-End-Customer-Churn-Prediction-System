import json
import pandas as pd


def load_schema(schema_path: str) -> dict:
    """
    Load feature schema from JSON.
    """
    with open(schema_path, "r") as f:
        schema = json.load(f)
    return schema


def validate_columns(df: pd.DataFrame, schema: dict):
    """
    Ensure all required columns are present.
    """
    required_columns = (
        schema["numerical_features"]
        + schema["categorical_features"]
        + [schema["target"]]
    )

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def drop_excluded_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Drop columns not used for modeling.
    """
    excluded = schema.get("excluded_features", [])
    return df.drop(columns=[col for col in excluded if col in df.columns])


def validate_target(df: pd.DataFrame, target_col: str):
    """
    Validate target column values.
    """
    unique_vals = df[target_col].unique()
    if len(unique_vals) != 2:
        raise ValueError(
            f"Target column '{target_col}' must be binary. Found: {unique_vals}"
        )


def validate_dataframe(
    df: pd.DataFrame,
    schema_path: str,
    require_target: bool = True
) -> pd.DataFrame:
    """
    Full validation pipeline.
    If require_target=False, target column is not enforced (used for inference).
    """
    schema = load_schema(schema_path)

    required_columns = (
        schema["numerical_features"]
        + schema["categorical_features"]
    )

    if require_target:
        required_columns = required_columns + [schema["target"]]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


    if require_target:
        validate_target(df, schema["target"])

  
    df = drop_excluded_columns(df, schema)

    return df
