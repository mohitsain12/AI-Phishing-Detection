"""Utility wrapper for Streamlit prediction calls.

This module handles the app-facing prediction request and forwards it to the
core prediction service. It also ensures logging is performed consistently
for each incoming URL prediction attempt.
"""

from .logger import logger

from predict import predict_from_url


def predict_url(url):
    """Predict a URL and return the model result dictionary.

    Args:
        url (str): The URL string to analyze.

    Returns:
        dict: Prediction result containing class, confidence, and feature data.
    """
    logger.info("Predict URL requested: %s", url)
    return predict_from_url(url)
