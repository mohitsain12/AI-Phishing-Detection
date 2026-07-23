"""History persistence utilities for prediction logging.

Provides helper functions to read, write, and reset prediction history stored in CSV format.
Uses centralized configuration constants for file paths and column definitions,
and caches CSV read operations using Streamlit's @st.cache_data.
"""

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    HISTORY_FILE,
    HISTORY_COLUMNS,
    TIMESTAMP_COL,
    URL_COL,
    PREDICTION_COL,
    CONFIDENCE_COL
)
from utils.logger import logger


def _raw_load_history():
    """Internal function to load prediction history from CSV file.

    Returns
    -------
    pd.DataFrame
        The history table DataFrame, or an empty DataFrame if missing or unparseable.
    """
    if HISTORY_FILE.exists():
        try:
            return pd.read_csv(HISTORY_FILE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc_error:
            logger.error("History CSV read failed: %s", exc_error)
            return pd.DataFrame(columns=HISTORY_COLUMNS)

    return pd.DataFrame(columns=HISTORY_COLUMNS)


try:
    import streamlit as st
    load_history = st.cache_data(_raw_load_history)
except Exception:
    load_history = _raw_load_history


def save_prediction(url_text, prediction_label, confidence_score):
    """Save a new prediction record entry to the history CSV file.

    Parameters
    ----------
    url_text : str
        The URL that was analyzed.
    prediction_label : str
        The predicted class label.
    confidence_score : float
        The model confidence percentage.
    """
    new_record = {
        TIMESTAMP_COL: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        URL_COL: url_text,
        PREDICTION_COL: prediction_label,
        CONFIDENCE_COL: round(confidence_score, 2)
    }

    if HISTORY_FILE.exists():
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if history_df.empty and len(history_df.columns) == 0:
                raise pd.errors.EmptyDataError
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc_error:
            logger.error("History CSV read failed during save: %s", exc_error)
            history_df = pd.DataFrame(columns=HISTORY_COLUMNS)
    else:
        history_df = pd.DataFrame(columns=HISTORY_COLUMNS)

    history_df.loc[len(history_df)] = new_record
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(HISTORY_FILE, index=False)
        logger.info("Saved history entry for URL %s", url_text)
        if hasattr(load_history, "clear"):
            load_history.clear()
    except OSError as exc_error:
        logger.error("Failed to write history file %s: %s", HISTORY_FILE, exc_error, exc_info=True)


def clear_history():
    """Clear all prediction records and rewrite an empty history CSV file."""
    empty_df = pd.DataFrame(columns=HISTORY_COLUMNS)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    empty_df.to_csv(HISTORY_FILE, index=False)
    logger.info("Cleared history file %s", HISTORY_FILE)
    if hasattr(load_history, "clear"):
        load_history.clear()


if __name__ == "__main__":
    save_prediction("https://google.com", "Legitimate", 99.82)