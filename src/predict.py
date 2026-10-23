
from pathlib import Path

import joblib
import pandas as pd


# Locate the saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraudshield_random_forest.joblib"

# Load the trained pipeline
model = joblib.load(MODEL_PATH)

# Columns expected by the model
REQUIRED_FEATURES = list(model.feature_names_in_)


def predict_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict fraud for transactions in a DataFrame.
    """

    # Check whether all required columns are present
    missing_columns = [
        col for col in REQUIRED_FEATURES
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select columns in the correct training order
    features = data[REQUIRED_FEATURES].copy()

    # Check for invalid or missing numerical values
    features = features.apply(
        pd.to_numeric, errors="coerce"
    )

    if features.isna().any().any():
        raise ValueError(
            "Transaction features must contain valid numerical values."
        )

    # Predict fraud and probabilities
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    # Return results without modifying the original data
    results = data.copy()

    results["Fraud_Prediction"] = predictions
    results["Fraud_Probability"] = probabilities

    return results
