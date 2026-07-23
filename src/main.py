"""Main module for AI Phishing Detection.

Demonstrates URL feature extraction and prediction for sample URLs.
"""

import sys
from pathlib import Path

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from src.predict import predict_from_url


def run_demo():
    """Run prediction demonstration on a list of benchmark URLs."""
    sample_urls = [
        "https://google.com",
        "https://www.google.com",
        "https://github.com",
        "https://www.github.com",
        "https://amazon.com",
        "https://www.amazon.com",
    ]

    print("AI Phishing Detection Demo\n" + "=" * 50)
    for target_url in sample_urls:
        prediction_result = predict_from_url(target_url)
        print(f"URL    : {target_url}")
        print(f"Result : {prediction_result['prediction']} ({'Phishing' if prediction_result['prediction'] == 1 else 'Legitimate'})")
        print(f"Confidence: {prediction_result['confidence']:.2f}%")
        print("-" * 50)


if __name__ == "__main__":
    run_demo()