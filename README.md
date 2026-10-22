# FraudShield AI
### Explainable Machine Learning for Credit Card Fraud Detection

## Project Overview
FraudShield AI is an end-to-end machine learning project focused on identifying fraudulent credit card transactions in highly imbalanced datasets.

The project explores supervised classification, anomaly detection, model optimization, and explainable AI to develop an interpretable fraud detection prototype.

## Problem Statement
Credit card fraud detection is challenging because fraudulent transactions represent only a small fraction of total transactions.

A model can achieve high accuracy by predicting most transactions as legitimate while failing to detect actual fraud.

This project investigates how machine learning can improve fraud identification while balancing missed fraudulent transactions and false alerts.

## Dataset
**Source:** [Credit Card Fraud Detection — ULB/Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Total transactions: 284,807
- Fraudulent transactions: 492
- Legitimate transactions: 284,315
- Features: 30 input variables and 1 target
- Target: `Class` (0 = legitimate, 1 = fraud)

The dataset contains anonymized transaction features, transaction amounts, and elapsed transaction times.

## Project Objectives
1. Analyze transaction patterns and class imbalance.
2. Develop reproducible preprocessing and feature engineering pipelines.
3. Compare baseline and advanced supervised ML algorithms.
4. Evaluate models using precision, recall, F1-score, ROC-AUC, and PR-AUC.
5. Investigate anomaly detection using Isolation Forest.
6. Optimize decision thresholds to balance fraud detection and false alerts.
7. Explain model predictions using SHAP.
8. Develop and deploy an interactive fraud detection prototype.

## Planned Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Streamlit, Git, and GitHub.

## Project Status
**In progress — Data understanding and exploratory analysis.**

Model results, experiment comparisons, and application details will be added as development progresses.

## Dataset Limitations
The dataset contains anonymized historical transactions. It is suitable for evaluating machine learning methods but does not establish production readiness or performance on future banking transactions.
