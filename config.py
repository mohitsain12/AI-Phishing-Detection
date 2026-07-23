"""Config module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Path and file constants
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
GRAPH_DIR = ROOT_DIR / "graphs"
LOG_DIR = ROOT_DIR / "logs"

# -----------------------------------------------------------------------------
# Data and Model File Paths
# -----------------------------------------------------------------------------
PHISHING_DATASET_PATH = DATA_DIR / "phishing.csv"
RAW_URLS_PATH = DATA_DIR / "raw_urls.csv"
FEATURES_DATASET_PATH = DATA_DIR / "features.csv"
HISTORY_FILE = DATA_DIR / "prediction_history.csv"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.pkl"
DECISION_TREE_MODEL_PATH = MODEL_DIR / "decision_tree_model.pkl"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
KNN_MODEL_PATH = MODEL_DIR / "knn_model.pkl"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
TUNED_MODEL_PATH = MODEL_DIR / "phishing_detector_tuned.pkl"
MODEL_PATH = TUNED_MODEL_PATH
FEATURE_NAMES_PATH = MODEL_DIR / "phishing_feature_names.pkl"

# -----------------------------------------------------------------------------
# Graph Output Paths
# -----------------------------------------------------------------------------
CONFUSION_MATRIX_GRAPH = GRAPH_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_GRAPH = GRAPH_DIR / "feature_importance.png"
MODEL_COMPARISON_GRAPH = GRAPH_DIR / "model_comparison.png"
TUNED_CONFUSION_MATRIX_GRAPH = GRAPH_DIR / "tuned_confusion_matrix.png"
TUNED_FEATURE_IMPORTANCE_GRAPH = GRAPH_DIR / "tuned_feature_importance.png"

# -----------------------------------------------------------------------------
# Application Identity & Thresholds
# -----------------------------------------------------------------------------
MODEL_NAME = "Random Forest"
APP_NAME = "AI Phishing Detector"
TOTAL_FEATURE_COUNT = 33

HIGH_RISK_THRESHOLD = 95.0
MEDIUM_RISK_THRESHOLD = 75.0

# -----------------------------------------------------------------------------
# Feature Extraction Thresholds & Rules
# -----------------------------------------------------------------------------
MAX_URL_LENGTH_THRESHOLD = 75
MAX_DOMAIN_LENGTH_THRESHOLD = 30
MANY_DOTS_THRESHOLD = 4
MAX_DIRECTORY_DEPTH_THRESHOLD = 5
HIGH_ENTROPY_THRESHOLD = 4.5
PARAMETER_COUNT_THRESHOLD = 3
SUBDOMAIN_COUNT_THRESHOLD = 2

SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "gq", "ml", "cf", "tk"
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "verification", "secure", "security",
    "update", "confirm", "account", "password", "wallet", "payment",
    "invoice", "bonus", "gift", "reward", "winner", "free"
]

SUSPICIOUS_EXTENSIONS = (
    ".php", ".exe", ".zip", ".rar", ".scr", ".js"
)

BRAND_NAMES = [
    "google", "paypal", "amazon", "microsoft", "apple",
    "facebook", "instagram", "netflix", "bank", "gmail"
]

TRUSTED_DOMAINS = {
    "github.com", "github.io", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "stackoverflow.email", "python.org", "pypi.org",
    "google.com", "youtube.com", "gmail.com", "facebook.com", "twitter.com",
    "linkedin.com", "amazon.com", "apple.com", "microsoft.com", "netflix.com",
    "paypal.com", "wikipedia.org", "reddit.com", "instagram.com", "yahoo.com"
}

# -----------------------------------------------------------------------------
# Machine Learning Training Default Constants
# -----------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# -----------------------------------------------------------------------------
# Labels and CSV Schema
# -----------------------------------------------------------------------------
LABEL_PHISHING = "Phishing"
LABEL_LEGITIMATE = "Legitimate"

TIMESTAMP_COL = "Timestamp"
URL_COL = "URL"
PREDICTION_COL = "Prediction"
CONFIDENCE_COL = "Confidence"
HISTORY_COLUMNS = [TIMESTAMP_COL, URL_COL, PREDICTION_COL, CONFIDENCE_COL]

