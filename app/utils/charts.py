"""Charts module for AI Phishing Detection.

Provides visualization components for dashboard analytics with optimized chart rendering,
memory cleanup (explicit plt.close), and precomputed metrics reuse.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import LABEL_LEGITIMATE, LABEL_PHISHING, TIMESTAMP_COL, CONFIDENCE_COL
from .metrics_utils import calculate_history_metrics


def show_pie_chart(history_df_or_metrics):
    """Render the prediction distribution pie chart.

    Parameters
    ----------
    history_df_or_metrics : pandas.DataFrame or dict
        Prediction history DataFrame or precomputed metrics dictionary.
    """
    if isinstance(history_df_or_metrics, dict):
        metrics_dict = history_df_or_metrics
    else:
        metrics_dict = calculate_history_metrics(history_df_or_metrics)

    sizes = [metrics_dict["legitimate_count"], metrics_dict["phishing_count"]]
    labels = [LABEL_LEGITIMATE, LABEL_PHISHING]

    with st.expander("🥧 Prediction Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title("Prediction Distribution")
        st.pyplot(fig)
        plt.close(fig)


def show_daily_trend(history_df):
    """Render the daily prediction trend line chart.

    Parameters
    ----------
    history_df : pandas.DataFrame
        Prediction history DataFrame.
    """
    if history_df.empty:
        return

    dates_series = pd.to_datetime(history_df[TIMESTAMP_COL]).dt.date
    daily_counts = dates_series.value_counts().sort_index()

    with st.expander("📈 Daily Prediction Trend", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(daily_counts.index, daily_counts.values, marker="o")
        ax.set_title("Daily Prediction Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Predictions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def show_confidence_histogram(history_df):
    """Render the confidence score histogram.

    Parameters
    ----------
    history_df : pandas.DataFrame
        Prediction history DataFrame.
    """
    if history_df.empty:
        return

    confidence_scores = history_df[CONFIDENCE_COL]

    with st.expander("📊 Confidence Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(confidence_scores, bins=10)
        ax.set_title("Confidence Distribution")
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Number of Predictions")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def show_history_charts(history_df, precomputed_metrics=None):
    """Render all history visualization charts for the dashboard.

    Parameters
    ----------
    history_df : pandas.DataFrame
        Prediction history DataFrame.
    precomputed_metrics : dict, optional
        Precomputed summary metrics dictionary.
    """
    if history_df.empty:
        return

    metrics = precomputed_metrics or calculate_history_metrics(history_df)
    show_pie_chart(metrics)
    show_daily_trend(history_df)
    show_confidence_histogram(history_df)
