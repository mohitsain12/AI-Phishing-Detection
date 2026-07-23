"""Predict module for AI Phishing Detection.

Loads the trained Random Forest model and feature names to execute prediction
for command-line interfaces and Streamlit applications.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse
import joblib
import pandas as pd

# Ensure root and app directories are accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
app_directory = root_directory / "app"
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))
if str(app_directory) not in sys.path:
    sys.path.insert(0, str(app_directory))

from config import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    TRUSTED_DOMAINS,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)
from src.feature_extractor import extract_features
from utils.logger import logger


class PredictionError(Exception):
    """Raised when a URL prediction cannot be completed."""


def _raw_load_model():
    """Internal model and feature names loader from disk.

    Returns
    -------
    tuple
        (fitted_model, feature_names_list)

    Raises
    ------
    FileNotFoundError
        If model files do not exist.
    """
    logger.info("Loading AI model from %s", MODEL_PATH)

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info("Model and feature names loaded successfully")
        return model, feature_names
    except FileNotFoundError as exc_error:
        logger.error("Model files not found: %s", exc_error)
        raise


try:
    import streamlit as st
    load_model = st.cache_resource(_raw_load_model)
except Exception:
    load_model = _raw_load_model



def get_url_input():
    """Prompt user for target URL via console input.

    Returns
    -------
    str
        User inputted URL string.
    """
    print("=" * 60)
    print("Phishing URL Detector")
    print("=" * 60)

    target_url = input("Enter URL: ").strip()
    if not target_url:
        print("Error: URL cannot be empty.")
        sys.exit(1)

    return target_url


def is_trusted_domain(url_text):
    """Return True for known trusted domains to prevent false positives.

    Parameters
    ----------
    url_text : str
        URL string to evaluate.

    Returns
    -------
    bool
        True if hostname is in TRUSTED_DOMAINS set.
    """
    normalized_url = url_text if "://" in url_text else "https://" + url_text
    parsed_url = urlparse(normalized_url.lower().strip())
    hostname = parsed_url.hostname or ""
    return hostname in TRUSTED_DOMAINS


def prepare_features(url_text, feature_names):
    """Extract lexical features and format into a pandas DataFrame matching model schema.

    Parameters
    ----------
    url_text : str
        URL string to extract.
    feature_names : list
        List of required feature columns.

    Returns
    -------
    tuple
        (sample_df, raw_feature_dict)
    """
    raw_features = extract_features(url_text)
    sample_df = pd.DataFrame([raw_features])[feature_names]
    return sample_df, raw_features


def predict_url(model, sample_df):
    """Predict phishing class and confidence probabilities for a sample feature DataFrame.

    Parameters
    ----------
    model : object
        Fitted classifier model.
    sample_df : pd.DataFrame
        Input features sample.

    Returns
    -------
    dict
        Prediction results dictionary.

    Raises
    ------
    TypeError
        If sample_df is not a pandas DataFrame.
    """
    logger.info("Prediction started")

    if not isinstance(sample_df, pd.DataFrame):
        error_msg = "Prediction sample must be a pandas DataFrame."
        logger.error(error_msg)
        raise TypeError(error_msg)

    prediction_class = int(model.predict(sample_df)[0])
    probabilities = model.predict_proba(sample_df)[0]

    legitimate_prob = round(float(probabilities[0]) * 100, 2)
    phishing_prob = round(float(probabilities[1]) * 100, 2)
    confidence_score = round(float(max(probabilities)) * 100, 2)

    logger.info("Prediction completed with confidence %.2f", confidence_score)

    return {
        "prediction": prediction_class,
        "confidence": confidence_score,
        "legitimate_probability": legitimate_prob,
        "phishing_probability": phishing_prob
    }


def _determine_risk_level(prediction_class, confidence_score):
    """Determine risk level classification using guard clauses.

    Parameters
    ----------
    prediction_class : int
        Predicted class label (1 for Phishing, 0 for Legitimate).
    confidence_score : float
        Prediction confidence score percentage.

    Returns
    -------
    str
        Risk level string ("SAFE", "HIGH", "MEDIUM", "LOW").
    """
    if prediction_class == 0:
        return "SAFE"
    if confidence_score >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    if confidence_score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def display_result(url_text, result_dict):
    """Display detection results in formatted console output.

    Parameters
    ----------
    url_text : str
        Analyzed URL string.
    result_dict : dict
        Result metrics dictionary.
    """
    print("\n" + "=" * 60)
    print("PHISHING DETECTION RESULT")
    print("=" * 60)
    print(f"URL : {url_text}\n")

    label_str = "PHISHING WEBSITE" if result_dict["prediction"] == 1 else "LEGITIMATE WEBSITE"
    print(f"Prediction : {label_str}")
    print(f"\nConfidence : {result_dict['confidence']:.2f}%")
    print(f"Legitimate Probability : {result_dict['legitimate_probability']:.2f}%")
    print(f"Phishing Probability   : {result_dict['phishing_probability']:.2f}%")

    risk_level = _determine_risk_level(result_dict["prediction"], result_dict["confidence"])
    print(f"\nRisk Level : {risk_level}")
    print("=" * 60)


def predict_from_url(url_text):
    """Streamlit and external API predictor function.

    Parameters
    ----------
    url_text : str
        URL string to evaluate.

    Returns
    -------
    dict
        Prediction results merged with extracted feature dictionary.
    """
    try:
        model, feature_names = load_model()
        sample_df, raw_features = prepare_features(url_text, feature_names)
        result_dict = predict_url(model, sample_df)

        if is_trusted_domain(url_text) and result_dict["prediction"] == 1:
            result_dict["prediction"] = 0
            result_dict["confidence"] = max(result_dict["confidence"], 90.0)
            result_dict["legitimate_probability"] = max(result_dict["legitimate_probability"], 95.0)
            result_dict["phishing_probability"] = min(result_dict["phishing_probability"], 5.0)
            raw_features["trusted_domain"] = 1
            logger.info("Trusted domain override applied for %s", url_text)

        result_dict["features"] = raw_features
        return result_dict
    except FileNotFoundError:
        logger.exception("Model files missing while predicting URL %s", url_text)
        raise
    except (TypeError, ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc_error:
        logger.warning("Prediction processing failed for URL %s: %s", url_text, exc_error)
        raise PredictionError("Unable to process the URL for prediction.") from exc_error
    except Exception:
        logger.exception("Unexpected prediction failure for URL %s", url_text)
        raise


def main():
    """Run interactive prediction CLI."""
    model, feature_names = load_model()
    target_url = get_url_input()
    sample_df, _ = prepare_features(target_url, feature_names)
    result_dict = predict_url(model, sample_df)
    display_result(target_url, result_dict)


if __name__ == "__main__":
    main()
