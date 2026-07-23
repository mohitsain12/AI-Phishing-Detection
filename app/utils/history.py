"""History module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import streamlit as st
from config import LABEL_PHISHING, LABEL_LEGITIMATE, URL_COL, CONFIDENCE_COL, TIMESTAMP_COL, PREDICTION_COL


def show_prediction_history(history_df):
    """Show prediction history.
    
    Parameters
    ----------
    history_df : TYPE
        Description of history_df.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    st.markdown("---")
    st.subheader("📜 Prediction History")

    if history_df.empty:
        st.info("No prediction history available.")
        return

    with st.expander("📋 Prediction History", expanded=False):
        st.dataframe(history_df, width="stretch", hide_index=True)


def show_recent_activity(history_df):
    """Show recent activity.
    
    Parameters
    ----------
    history_df : TYPE
        Description of history_df.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    if history_df.empty:
        return

    history_df = history_df.sort_values(by=TIMESTAMP_COL, ascending=False)
    recent_predictions = history_df.head(5)

    with st.expander("📅 Recent Activity", expanded=False):
        for _, row in recent_predictions.iterrows():
            if row[PREDICTION_COL] == LABEL_PHISHING:
                st.error("🔴 Phishing URL")
            else:
                st.success("🟢 Legitimate URL")

            st.write(f"**🌐 URL:** {row[URL_COL]}")
            st.write(f"**📊 Confidence:** {row[CONFIDENCE_COL]:.2f}%")
            st.write(f"**🕒 Time:** {row[TIMESTAMP_COL]}")
            st.divider()


def show_history_controls(history_df, clear_history):
    """Show history controls.
    
    Parameters
    ----------
    history_df : TYPE
        Description of history_df.
    clear_history : TYPE
        Description of clear_history.
    """
    if st.button("🗑️ Clear History"):
        clear_history()
        st.success("Prediction history cleared successfully.")
        st.rerun()

    if not history_df.empty:
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
