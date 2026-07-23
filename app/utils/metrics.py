"""Metrics display helpers for prediction history analytics."""

import streamlit as st
from .metrics_utils import calculate_history_metrics


def show_history_metrics(history_df_or_metrics):
    """Render summary metrics for the prediction history.

    Parameters
    ----------
    history_df_or_metrics : pandas.DataFrame or dict
        The prediction history data DataFrame or precomputed metrics dictionary.
    """
    if isinstance(history_df_or_metrics, dict):
        metrics_dict = history_df_or_metrics
    else:
        metrics_dict = calculate_history_metrics(history_df_or_metrics)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", metrics_dict["total_predictions"])
    with col2:
        st.metric("Phishing", metrics_dict["phishing_count"])
    with col3:
        st.metric("Legitimate", metrics_dict["legitimate_count"])
    with col4:
        st.metric("Average Confidence", f"{metrics_dict['average_confidence']:.2f}%")
