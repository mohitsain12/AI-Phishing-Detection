import os
import pandas as pd
from datetime import datetime

HISTORY_FILE = "../data/prediction_history.csv"

def save_prediction(url, prediction, confidence):

    new_row = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "URL": url,
        "Prediction": prediction,
        "Confidence": round(confidence, 2)
    }

    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if history_df.empty and len(history_df.columns) == 0:
                raise pd.errors.EmptyDataError
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            history_df = pd.DataFrame(
                columns=["Timestamp", "URL", "Prediction", "Confidence"]
            )
    else:
        history_df = pd.DataFrame(
            columns=["Timestamp", "URL", "Prediction", "Confidence"]
        )

    history_df.loc[len(history_df)] = new_row

    history_df.to_csv(HISTORY_FILE, index=False)
        
def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame(
        columns=[
            "Timestamp",
            "URL",
            "Prediction",
            "Confidence"
        ]
    )        

def clear_history():

    empty_df = pd.DataFrame(
        columns=[
            "Timestamp",
            "URL",
            "Prediction",
            "Confidence"
        ]
    )

    empty_df.to_csv(HISTORY_FILE, index=False)

if __name__ == "__main__":
    save_prediction(
        "https://google.com",
        "Legitimate",
        99.82
    )
  