# ==========================================================
# AI PHISHING DETECTION
# Module 5 : URL Prediction
# ==========================================================

import joblib
import pandas as pd

from feature_extractor import extract_features

# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    print("=" * 60)
    print("Loading AI Model")
    print("=" * 60)

    try:

        model = joblib.load(
            "../models/random_forest_model.pkl"
        )

        feature_names = joblib.load(
            "../models/phishing_feature_names.pkl"
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

    return sample

# ==========================================================
# Predict URL
# ==========================================================

def predict_url(model, sample):

    print("=" * 60)
    print("Analyzing URL")
    print("=" * 60)

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

    sample = prepare_features(

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
    Takes a URL and returns prediction results.
    """

    model, feature_names = load_model()

    sample = prepare_features(
        url,
        feature_names
    )

    result = predict_url(
        model,
        sample
    )

    return result


if __name__ == "__main__":

    main()