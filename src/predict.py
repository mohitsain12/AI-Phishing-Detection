# ==========================================================
# AI PHISHING DETECTION
# Module 5 : URL Prediction
# ==========================================================

import joblib
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

from feature_extractor import extract_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    print("=" * 60)
    print("Loading AI Model")
    print("=" * 60)

    try:

        model = joblib.load(
            MODEL_DIR / "phishing_detector_tuned.pkl"
        )

        feature_names = joblib.load(
            MODEL_DIR / "phishing_feature_names.pkl"
        )

        print("Model Loaded Successfully!")

        print("Feature Names Loaded Successfully!\n")

        return model, feature_names

    except FileNotFoundError:

        print("Error: Model files not found.")

        print("Run train_model.py or tune_model.py first.")

        exit()

# ==========================================================
# User Input
# ==========================================================

def get_url():

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

    print("=" * 60)
    print("Analyzing URL")
    print("=" * 60)

    if not isinstance(sample, pd.DataFrame):
        raise TypeError("Prediction sample must be a pandas DataFrame.")

    # Predict class
    prediction = model.predict(sample)[0]

    # Predict probabilities
    probabilities = model.predict_proba(sample)[0]

    legitimate_probability = probabilities[0] * 100

    phishing_probability = probabilities[1] * 100

    confidence = max(probabilities) * 100

    print("Prediction Completed!\n")

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

    result["features"] = features

    return result


if __name__ == "__main__":

    main()
