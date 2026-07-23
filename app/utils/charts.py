"""Charts module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import LABEL_LEGITIMATE, LABEL_PHISHING, PREDICTION_COL, TIMESTAMP_COL, CONFIDENCE_COL


def show_pie_chart(history_df):
    """Render the prediction distribution pie chart."""
    sizes = [
        len(history_df[history_df[PREDICTION_COL] == LABEL_LEGITIMATE]),
        len(history_df[history_df[PREDICTION_COL] == LABEL_PHISHING]),
    ]
    labels = [LABEL_LEGITIMATE, LABEL_PHISHING]

    with st.expander("🥧 Prediction Distribution", expanded=False):
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title("Prediction Distribution")
        st.pyplot(fig)


def show_daily_trend(history_df):
    """Render the daily prediction trend line chart."""
    history_df[TIMESTAMP_COL] = pd.to_datetime(history_df[TIMESTAMP_COL])
    history_df["Date"] = history_df[TIMESTAMP_COL].dt.date
    daily_predictions = history_df.groupby("Date").size()

    with st.expander("📈 Daily Prediction Trend", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(daily_predictions.index, daily_predictions.values, marker="o")
        ax.set_title("Daily Prediction Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Predictions")
        plt.xticks(rotation=45)
        st.pyplot(fig)


def show_confidence_histogram(history_df):
    """Render the confidence score histogram."""
    confidence_scores = history_df[CONFIDENCE_COL]

    with st.expander("📊 Confidence Distribution", expanded=False):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(confidence_scores, bins=10)
        ax.set_title("Confidence Distribution")
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("Number of Predictions")
        st.pyplot(fig)


def show_history_charts(history_df):
    """Render all history visualization charts for the dashboard."""
    if history_df.empty:
        return

    show_pie_chart(history_df)
    show_daily_trend(history_df)
    show_confidence_histogram(history_df)
