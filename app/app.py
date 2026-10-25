
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_transactions

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

REQUIRED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

MAX_ROWS = 100_000


# --------------------------------------------------
# 2. VALIDATE UPLOADED DATA
# --------------------------------------------------

def validate_input(data: pd.DataFrame):

    if data.empty:
        raise ValueError("The CSV is empty.")

    if len(data) > MAX_ROWS:
        raise ValueError(
            f"Please upload at most {MAX_ROWS:,} transactions."
        )

    if not data.columns.is_unique:
        raise ValueError("Duplicate column names found.")

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    features = data[REQUIRED_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if features.isna().any().any():
        raise ValueError(
            "Input features must contain valid numbers "
            "without missing values."
        )

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values.")

    if (features["Amount"] < 0).any():
        raise ValueError("Transaction Amount cannot be negative.")

    return data


# --------------------------------------------------
# 3. WEBSITE HEADER
# --------------------------------------------------

st.title("🛡️ FraudShield AI")

st.caption(
    "Credit Card Fraud Detection | Random Forest Machine Learning"
)

st.write(
    "Upload anonymized transaction data to identify "
    "transactions that may require further review."
)

st.info(
    "Required columns: Time, V1–V28, and Amount. "
    "The Class column is optional and is not used for prediction."
)


# --------------------------------------------------
# 4. CSV INSTRUCTIONS
# --------------------------------------------------

with st.expander("How to prepare your CSV"):

    st.write(
        "Each row represents one transaction. "
        "The file must contain the 30 input features "
        "used during model training."
    )

    st.write(
        "**Time:** Seconds since the first transaction."
    )

    st.write(
        "**V1–V28:** Anonymized PCA-transformed features."
    )

    st.write(
        "**Amount:** Transaction amount."
    )

    st.write(
        "**Class:** Optional actual label "
        "(0 = legitimate, 1 = fraud)."
    )

    st.warning(
        "Ordinary bank statements cannot be uploaded directly "
        "because they do not contain the required PCA features."
    )


# --------------------------------------------------
# 5. UPLOAD CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)


# --------------------------------------------------
# 6. HANDLE FILE CHANGES
# --------------------------------------------------

if uploaded_file is not None:

    # Create a unique identifier from the file contents
    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    # Clear previous results if a different file is uploaded
    if st.session_state.get("current_file_hash") != file_hash:

        st.session_state["current_file_hash"] = file_hash

        st.session_state.pop(
            "fraudshield_results",
            None
        )

    try:

        # Read and validate CSV
        data = pd.read_csv(uploaded_file)

        data = validate_input(data)

        # Show preview
        st.subheader("Uploaded Data Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        # ------------------------------------------
        # 7. RUN FRAUD DETECTION
        # ------------------------------------------

        if st.button("🔍 Detect Fraud", type="primary"):

            with st.spinner("Analyzing transactions..."):

                results = predict_transactions(data).copy()

                # Create readable prediction labels
                results["Fraud_Status"] = results[
                    "Fraud_Prediction"
                ].map({
                    0: "Legitimate",
                    1: "Flagged for Review"
                })

                # Convert fraud score into percentage
                results["Fraud_Score_Percent"] = (
                    results["Fraud_Probability"] * 100
                ).round(2)

                # Add transaction IDs
                results.insert(
                    0,
                    "Transaction_ID",
                    range(1, len(results) + 1)
                )

                # Put important columns first
                important_columns = [
                    "Transaction_ID",
                    "Fraud_Status",
                    "Fraud_Score_Percent",
                    "Amount",
                    "Time",
                    "Fraud_Prediction",
                    "Fraud_Probability"
                ]

                if "Class" in results.columns:
                    important_columns.append("Class")

                other_columns = [
                    column for column in results.columns
                    if column not in important_columns
                ]

                results = results[
                    important_columns + other_columns
                ]

                # Store predictions for the current file
                st.session_state[
                    "fraudshield_results"
                ] = results

                st.session_state[
                    "results_file_hash"
                ] = file_hash

    except (
        ValueError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OverflowError
    ) as error:

        st.session_state.pop(
            "fraudshield_results",
            None
        )

        st.error(f"Could not process the file: {error}")

    # --------------------------------------------------
    # 8. SHOW RESULTS ONLY FOR THE CURRENT FILE
    # --------------------------------------------------

    if (
        "fraudshield_results" in st.session_state
        and st.session_state.get("results_file_hash") == file_hash
    ):

        results = st.session_state["fraudshield_results"]

        total = len(results)

        flagged = int(
            (results["Fraud_Prediction"] == 1).sum()
        )

        legitimate = total - flagged

        flagged_rows = results[
            results["Fraud_Prediction"] == 1
        ]

        st.success("Fraud detection completed!")

        # ------------------------------------------
        # 9. DASHBOARD METRICS
        # ------------------------------------------

        st.subheader("Prediction Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Transactions Analyzed",
            f"{total:,}"
        )

        col2.metric(
            "Predicted Legitimate",
            f"{legitimate:,}"
        )

        col3.metric(
            "Flagged for Review",
            f"{flagged:,}"
        )

        col4.metric(
            "Flagged Rate",
            f"{flagged / total * 100:.2f}%"
        )

        # ------------------------------------------
        # 10. VISUALIZATIONS
        # ------------------------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("Prediction Distribution")

            chart_data = pd.DataFrame({
                "Status": ["Legitimate", "Flagged"],
                "Transactions": [legitimate, flagged]
            })

            st.bar_chart(
                chart_data.set_index("Status")
            )

            st.caption(
                "Fraud is rare, so the flagged bar "
                "may appear very small."
            )

        with right:

            st.subheader("Fraud Score Distribution")

            # Group scores into ranges
            score_bins = pd.cut(
                results["Fraud_Score_Percent"],
                bins=[-0.01, 10, 20, 30, 40, 50,
                      60, 70, 80, 90, 100],
                labels=[
                    "0–10", "10–20", "20–30",
                    "30–40", "40–50", "50–60",
                    "60–70", "70–80",
                    "80–90", "90–100"
                ],
                include_lowest=True
            )

            score_counts = (
                score_bins.value_counts()
                .sort_index()
            )

            score_chart = pd.DataFrame({
                "Score Range (%)": score_counts.index.astype(str),
                "Transactions": score_counts.values
            })

            st.bar_chart(
                score_chart.set_index("Score Range (%)")
            )

            st.caption(
                "Number of transactions in each fraud-score range."
            )

        # ------------------------------------------
        # 11. FLAGGED TRANSACTIONS
        # ------------------------------------------

        st.subheader("Transactions Flagged for Review")

        if flagged_rows.empty:

            st.info(
                "No transactions were flagged in this upload."
            )

        else:

            st.dataframe(
                flagged_rows[[
                    "Transaction_ID",
                    "Amount",
                    "Fraud_Score_Percent",
                    "Fraud_Status"
                ]],
                use_container_width=True,
                hide_index=True
            )

        # ------------------------------------------
        # 12. ALL PREDICTION RESULTS
        # ------------------------------------------

        st.subheader("All Prediction Results")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # 13. DOWNLOAD RESULTS
        # ------------------------------------------

        st.download_button(
            label="⬇️ Download Complete Prediction Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "Fraud scores are model estimates, not guarantees. "
            "The model uses a 0.50 classification threshold. "
            "Flagged transactions require further review."
        )

else:

    # Clear stored predictions when no file is uploaded
    st.session_state.pop("current_file_hash", None)
    st.session_state.pop("results_file_hash", None)
    st.session_state.pop("fraudshield_results", None)


# --------------------------------------------------
# 14. ABOUT THE MODEL
# --------------------------------------------------

st.divider()

with st.expander("About the Model and Limitations"):

    st.write("**Algorithm:** Random Forest Classifier")

    st.write(
        "**Dataset:** ULB/Kaggle Credit Card Fraud Detection"
    )

    st.write(
        "**Dataset size:** 284,807 transactions, "
        "including 492 labeled fraud cases."
    )

    st.write("**Final held-out test performance:**")

    st.write(
        "Precision: 95.16% | "
        "Recall: 79.73% | "
        "F1-score: 86.76% | "
        "Average Precision: 81.79%"
    )

    st.warning(
        "This is an educational machine learning project, "
        "not a production banking fraud detection system. "
        "It uses historical anonymized data, and predictions "
        "may not generalize to live transactions. "
        "Fraud probabilities may not be calibrated. "
        "Do not upload sensitive real customer information."
    )
