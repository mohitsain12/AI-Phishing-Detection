"""History persistence utilities for prediction logging.

This module provides helper functions to read, write, and reset prediction
history stored in a CSV file. It uses centralized config constants for the
history file path and column names.
"""

import os
import pandas as pd
from datetime import datetime
from config import HISTORY_FILE, HISTORY_COLUMNS, TIMESTAMP_COL, URL_COL, PREDICTION_COL, CONFIDENCE_COL
from utils.logger import logger


def save_prediction(url, prediction, confidence):
    """Save a new prediction entry to the history CSV.

    Args:
        url (str): The URL that was analyzed.
        prediction (str): The predicted label.
        confidence (float): The model confidence score.
    """

    new_row = {
        TIMESTAMP_COL: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        URL_COL: url,
        PREDICTION_COL: prediction,
        CONFIDENCE_COL: round(confidence, 2)
    }

    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if history_df.empty and len(history_df.columns) == 0:
                raise pd.errors.EmptyDataError
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("History CSV read failed during save: %s", exc)
            history_df = pd.DataFrame(
                columns=HISTORY_COLUMNS
            )
    else:
        history_df = pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    history_df.loc[len(history_df)] = new_row
    try:
        history_df.to_csv(HISTORY_FILE, index=False)
        logger.info("Saved history entry for URL %s", url)
    except OSError as exc:
        logger.error("Failed to write history file %s: %s", HISTORY_FILE, exc, exc_info=True)
        
def load_history():
    """Load prediction history from the CSV file.

    Returns:
        pandas.DataFrame: The history table, or an empty DataFrame if the file is
        missing, empty, or corrupted.
    """

    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.error("History CSV read failed: %s", exc)
            return pd.DataFrame(columns=HISTORY_COLUMNS)

    return pd.DataFrame(
        columns=HISTORY_COLUMNS
    )        

def clear_history():
    """Clear all prediction history and rewrite an empty CSV file."""

    empty_df = pd.DataFrame(
        columns=HISTORY_COLUMNS
    )

    empty_df.to_csv(HISTORY_FILE, index=False)
    logger.info("History cleared")
    logger.info("Cleared history file %s", HISTORY_FILE)

if __name__ == "__main__":
    save_prediction(
        "https://google.com",
        "Legitimate",
        99.82
    )
  