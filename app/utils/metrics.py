"""Metrics display helpers for prediction history analytics."""

import streamlit as st
from config import LABEL_PHISHING, LABEL_LEGITIMATE, CONFIDENCE_COL, PREDICTION_COL


def show_history_metrics(history_df):
    """Render summary metrics for the prediction history.

    Args:
        history_df (pandas.DataFrame): The prediction history data.
    """
    total_predictions = len(history_df)
    phishing_count = len(history_df[history_df[PREDICTION_COL] == LABEL_PHISHING])
    legitimate_count = len(history_df[history_df[PREDICTION_COL] == LABEL_LEGITIMATE])
    average_confidence = history_df[CONFIDENCE_COL].mean() if total_predictions > 0 else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", total_predictions)
    with col2:
        st.metric("Phishing", phishing_count)
    with col3:
        st.metric("Legitimate", legitimate_count)
    with col4:
        st.metric("Average Confidence", f"{average_confidence:.2f}%")
