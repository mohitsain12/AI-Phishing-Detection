"""Metrics calculation utility module.

Computes summary statistics once from history DataFrame to avoid duplicate
computations across metrics cards and visualization charts.
"""

from config import LABEL_PHISHING, LABEL_LEGITIMATE, CONFIDENCE_COL, PREDICTION_COL


def calculate_history_metrics(history_df):
    """Compute summary metrics from prediction history DataFrame.

    Parameters
    ----------
    history_df : pd.DataFrame
        Prediction history data.

    Returns
    -------
    dict
        Dictionary containing precomputed metric counts and statistics:
        - total_predictions
        - phishing_count
        - legitimate_count
        - average_confidence
        - phishing_ratio
        - legitimate_ratio
    """
    if history_df is None or history_df.empty:
        return {
            "total_predictions": 0,
            "phishing_count": 0,
            "legitimate_count": 0,
            "average_confidence": 0.0,
            "phishing_ratio": 0.0,
            "legitimate_ratio": 0.0
        }

    total_predictions = len(history_df)
    phishing_count = int((history_df[PREDICTION_COL] == LABEL_PHISHING).sum())
    legitimate_count = int((history_df[PREDICTION_COL] == LABEL_LEGITIMATE).sum())
    average_confidence = float(history_df[CONFIDENCE_COL].mean()) if total_predictions > 0 else 0.0

    phishing_ratio = round((phishing_count / total_predictions) * 100, 1) if total_predictions > 0 else 0.0
    legitimate_ratio = round((legitimate_count / total_predictions) * 100, 1) if total_predictions > 0 else 0.0

    return {
        "total_predictions": total_predictions,
        "phishing_count": phishing_count,
        "legitimate_count": legitimate_count,
        "average_confidence": average_confidence,
        "phishing_ratio": phishing_ratio,
        "legitimate_ratio": legitimate_ratio
    }
