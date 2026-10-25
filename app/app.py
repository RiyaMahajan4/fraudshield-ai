import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow Python to find our src folder
sys.path.insert(0, str(PROJECT_ROOT))

# Import our prediction function
from src.predict import predict_transactions

# Configure the Streamlit page
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FraudShield AI")

st.write(
    "Credit Card Fraud Detection using Machine Learning"
)

st.info(
    "Upload a CSV file containing Time, Amount, and V1–V28."
)

# Upload CSV file
uploaded_file = st.file_uploader(
    "Choose a transaction CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Transaction Data")
        st.dataframe(data.head(10))

        # Run fraud detection
        if st.button("Detect Fraud"):

            results = predict_transactions(data)

            st.success("Fraud detection completed!")

            st.subheader("Prediction Results")
            st.dataframe(results.head(20))

    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to process the CSV: {error}")
