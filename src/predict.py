"""Predict module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

# ==========================================================
# AI PHISHING DETECTION
# Module 5 : URL Prediction
# ==========================================================

import joblib
import pandas as pd
from urllib.parse import urlparse

from utils.logger import logger
from feature_extractor import extract_features
from config import MODEL_PATH, FEATURE_NAMES_PATH


class PredictionError(Exception):
    """Raised when a URL prediction cannot be completed."""

# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    """Load model.
    
    Returns
    -------
    TYPE
        Description of return value.
    
    Raises
    ------
    Exception
        If an error occurs during execution.
    """
    logger.info("Loading AI model from %s", MODEL_PATH)

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)

        logger.info("Model loaded successfully")
        logger.info("Feature names loaded successfully")

        return model, feature_names

    except FileNotFoundError as exc:
        logger.error("Model files not found: %s", exc)
        raise

# ==========================================================
# User Input
# ==========================================================

def get_url():

    """Get url.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    print("=" * 60)
    print("Phishing URL Detector")
    print("=" * 60)

    url = input("Enter URL: ").strip()

    if not url:

        print("Error: URL cannot be empty.")

        exit()

    return url


def is_trusted_domain(url):
    """Return True for domains that should not be misclassified as phishing."""

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url.lower().strip())
    hostname = parsed.hostname or ""

    trusted_domains = {
        "github.com",
        "github.io",
        "gitlab.com",
        "bitbucket.org",
        "stackoverflow.com",
        "stackoverflow.email",
        "python.org",
        "pypi.org",
        "google.com",
        "youtube.com",
        "gmail.com",
        "facebook.com",
        "twitter.com",
        "linkedin.com",
        "amazon.com",
        "apple.com",
        "microsoft.com",
        "netflix.com",
        "paypal.com",
        "wikipedia.org",
        "reddit.com",
        "instagram.com",
        "yahoo.com"
    }

    return hostname in trusted_domains


# ==========================================================
# Feature Extraction
# ==========================================================

def prepare_features(url, feature_names):

    """Prepare features.
    
    Parameters
    ----------
    url : TYPE
        Description of url.
    feature_names : TYPE
        Description of feature_names.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    print("\nExtracting URL Features...\n")

    features = extract_features(url)

    sample = pd.DataFrame([features])

    sample = sample[feature_names]
    print(sample.T)

    print("Feature Extraction Completed!\n")

    return sample, features

# ==========================================================
# Predict URL
# ==========================================================

def predict_url(model, sample):

    """Predict url.
    
    Parameters
    ----------
    model : TYPE
        Description of model.
    sample : TYPE
        Description of sample.
    
    Returns
    -------
    TYPE
        Description of return value.
    
    Raises
    ------
    Exception
        If an error occurs during execution.
    """
    logger.info("Prediction started")

    if not isinstance(sample, pd.DataFrame):
        error_msg = "Prediction sample must be a pandas DataFrame."
        logger.error(error_msg)
        raise TypeError(error_msg)

    # Predict class
    prediction = model.predict(sample)[0]

    # Predict probabilities
    probabilities = model.predict_proba(sample)[0]

    legitimate_probability = probabilities[0] * 100
    phishing_probability = probabilities[1] * 100
    confidence = max(probabilities) * 100

    logger.info("Prediction completed with confidence %.2f", confidence)

    result = {

        "prediction": prediction,

        "confidence": confidence,

        "legitimate_probability": legitimate_probability,

        "phishing_probability": phishing_probability

    }

    return result

# ==========================================================
# Display Result
# ==========================================================

def display_result(url, result):

    """Display result.
    
    Parameters
    ----------
    url : TYPE
        Description of url.
    result : TYPE
        Description of result.
    """
    print("\n" + "=" * 60)
    print("PHISHING DETECTION RESULT")
    print("=" * 60)

    print(f"URL : {url}\n")

    if result["prediction"] == 0:

        print("Prediction : LEGITIMATE WEBSITE")

    else:

        print("Prediction : PHISHING WEBSITE")

    print(f"\nConfidence : {result['confidence']:.2f}%")

    print(f"Legitimate Probability : {result['legitimate_probability']:.2f}%")

    print(f"Phishing Probability   : {result['phishing_probability']:.2f}%")

    # --------------------------------------------
    # Risk Level
    # --------------------------------------------

    if result["prediction"] == 1:

        if result["confidence"] >= 95:

            risk = "HIGH"

        elif result["confidence"] >= 75:

            risk = "MEDIUM"

        else:

            risk = "LOW"

    else:

        risk = "SAFE"

    print(f"\nRisk Level : {risk}")

    print("=" * 60)
    
# ==========================================================
# Main
# ==========================================================

def main():

    """Main.
    """
    model, feature_names = load_model()

    url = get_url()

    sample, features = prepare_features(

        url,

        feature_names

    )

    result = predict_url(

        model,

        sample

    )

    display_result(

        url,

        result

    )

def predict_from_url(url):
    """
    Streamlit-friendly prediction function.
    Takes a URL and returns prediction results with features.
    """

    try:
        model, feature_names = load_model()
        sample, features = prepare_features(
            url,
            feature_names
        )

        result = predict_url(
            model,
            sample
        )

        if is_trusted_domain(url) and result["prediction"] == 1:
            # Override false positive on known safe domains.
            result["prediction"] = 0
            result["confidence"] = max(result["confidence"], 90.0)
            result["legitimate_probability"] = max(result["legitimate_probability"], 95.0)
            result["phishing_probability"] = min(result["phishing_probability"], 5.0)
            features["trusted_domain"] = 1
            logger.info("Trusted domain override applied for %s", url)

        result["features"] = features
        return result
    except FileNotFoundError:
        logger.exception("Model files missing while predicting URL %s", url)
        raise
    except (TypeError, ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning("Prediction processing failed for URL %s: %s", url, exc)
        raise PredictionError("Unable to process the URL for prediction.") from exc
    except Exception:
        logger.exception("Unexpected prediction failure for URL %s", url)
        raise


if __name__ == "__main__":

    main()
