"""Test module for AI Phishing Detection prediction pipeline.

Validates error handling and prediction workflows.
"""

import sys
from pathlib import Path
import pytest

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from src.predict import PredictionError, predict_from_url


def test_predict_from_url_raises_prediction_error(monkeypatch):
    """Verify predict_from_url raises PredictionError when feature preparation fails."""
    def mock_prepare_features(url_text, feature_names):
        raise ValueError("Invalid URL format")

    monkeypatch.setattr("src.predict.prepare_features", mock_prepare_features)
    monkeypatch.setattr("src.predict.load_model", lambda: (None, []))

    with pytest.raises(PredictionError):
        predict_from_url("https://malformed-url-test.local")
