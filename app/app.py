import streamlit as st

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
    "Upload a transaction CSV file to detect potential fraud."
)
