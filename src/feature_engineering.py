import pandas as pd


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=["0-12", "12-24", "24-48", "48-60", "60+"]
    )
    return df


def add_has_internet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_internet"] = df["InternetService"].apply(
        lambda x: "No" if x == "No" else "Yes"
    )
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    internet_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["service_count"] = df[internet_services].apply(
        lambda row: sum(val == "Yes" for val in row),
        axis=1
    )
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_tenure_group(df)
    df = add_has_internet(df)
    df = add_service_count(df)

    return df

