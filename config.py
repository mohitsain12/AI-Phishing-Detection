"""Config module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Path and file constants
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
HISTORY_FILE = ROOT_DIR / "data" / "prediction_history.csv"
MODEL_PATH = MODEL_DIR / "phishing_detector_tuned.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "phishing_feature_names.pkl"

# -----------------------------------------------------------------------------
# Application identity
# -----------------------------------------------------------------------------
MODEL_NAME = "Random Forest"
APP_NAME = "AI Phishing Detector"
TOTAL_FEATURE_COUNT = 33

# -----------------------------------------------------------------------------
# Risk thresholds
# -----------------------------------------------------------------------------
HIGH_RISK_THRESHOLD = 95.0
MEDIUM_RISK_THRESHOLD = 75.0

# -----------------------------------------------------------------------------
# Labels and CSV schema
# -----------------------------------------------------------------------------
LABEL_PHISHING = "Phishing"
LABEL_LEGITIMATE = "Legitimate"

TIMESTAMP_COL = "Timestamp"
URL_COL = "URL"
PREDICTION_COL = "Prediction"
CONFIDENCE_COL = "Confidence"
HISTORY_COLUMNS = [TIMESTAMP_COL, URL_COL, PREDICTION_COL, CONFIDENCE_COL]
