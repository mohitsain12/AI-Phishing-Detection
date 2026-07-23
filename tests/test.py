"""Test module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import pytest
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(ROOT_DIR, "app")
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ROOT_DIR)
from src.predict import PredictionError, predict_from_url

def test_predict_from_url_raises_prediction_error(monkeypatch):
    """Test predict from url raises prediction error.
    
    Parameters
    ----------
    monkeypatch : TYPE
        Description of monkeypatch.
    
    Raises
    ------
    Exception
        If an error occurs during execution.
    """
    def fake_prepare_features(url, feature_names):
        """Fake prepare features.
        
        Parameters
        ----------
        url : TYPE
            Description of url.
        feature_names : TYPE
            Description of feature_names.
        
        Raises
        ------
        Exception
            If an error occurs during execution.
        """
        raise ValueError("bad url")

    monkeypatch.setattr("src.predict.prepare_features", fake_prepare_features)
    monkeypatch.setattr("src.predict.load_model", lambda: (None, []))

    with pytest.raises(PredictionError):
        predict_from_url("https://google.com.com")