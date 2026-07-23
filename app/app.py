"""App module for AI Phishing Detection.

Streamlit frontend application incorporating glassmorphism UI styling,
single-pass model caching, single CSV load, and optimized analytics calculations.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Add app, root, and src folders to Python path
APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
src_dir = os.path.abspath(os.path.join(APP_DIR, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import APP_NAME, MODEL_NAME, TOTAL_FEATURE_COUNT, LABEL_PHISHING, LABEL_LEGITIMATE
from utils.logger import logger
from utils.predictor import predict_url, PredictionError
from utils.result_ui import show_result_card
from utils.feature_panel import show_feature_panel
from utils.recommendation import show_recommendation
from utils.history import (
    show_prediction_history,
    show_recent_activity,
    show_history_controls
)
from utils.charts import show_history_charts
from utils.metrics import show_history_metrics
from utils.metrics_utils import calculate_history_metrics

from history import (
    save_prediction,
    load_history,
    clear_history
)
from explainer import explain_prediction


# =============================================================
# Page Configuration
# =============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger.info("App rendering started")

# =============================================================
# Custom CSS — Premium Dark Theme with Glassmorphism
# =============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 25%, #1b2838 50%, #0d1b2a 75%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(100, 200, 255, 0.1);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }

    .hero-title {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }

    .hero-title h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(100, 200, 255, 0.2);
        box-shadow: 0 8px 40px rgba(59, 130, 246, 0.1);
    }

    .scanning-animation {
        text-align: center;
        padding: 2rem;
    }

    .scanning-animation .scanner-text {
        font-size: 1.2rem;
        color: #60a5fa;
        font-weight: 600;
        animation: scanner-pulse 1.5s ease-in-out infinite;
    }

    @keyframes scanner-pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }

    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.2), transparent);
        margin: 2rem 0;
        border: none;
    }

    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Sidebar
# =============================================================

with st.sidebar:
    st.markdown("## 🛡️ PhishGuard AI")
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.markdown(f"""
    - **Algorithm**: {MODEL_NAME}
    - **Features**: {TOTAL_FEATURE_COUNT} lexical URL features
    - **Accuracy**: ~99.5%
    - **Training Data**: 100k+ URLs
    """)
    st.markdown("---")
    st.markdown("### 🔬 How It Works")
    st.markdown(f"""
    1. **Extract** {TOTAL_FEATURE_COUNT} features from the URL
    2. **Analyze** using {MODEL_NAME} ML model
    3. **Predict** phishing probability
    4. **Report** risk level & confidence
    """)


# =============================================================
# Hero Section
# =============================================================

st.markdown(f"""
<div class="hero-title">
    <h1>🛡️ {APP_NAME}</h1>
</div>
<p class="hero-subtitle">
    Detect phishing URLs instantly using Machine Learning.
    Enter any URL below to analyze its safety.
</p>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# =============================================================
# URL Input Section
# =============================================================

url = st.text_input(
    "🔗 Enter URL to Analyze",
    placeholder="https://example.com/login/verify-account",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze = st.button("🔍  Analyze URL", width="stretch")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# =============================================================
# Analysis Results
# =============================================================

if analyze:
    logger.info("User submitted URL: %s", url)

    if not url or url.strip() == "":
        st.markdown("""
        <div class="glass-card" style="text-align: center; border-color: rgba(250, 204, 21, 0.3);">
            <p style="color: #facc15; font-size: 1.1rem; font-weight: 600; margin: 0;">
                ⚠️ Please enter a URL to analyze
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(""):
            st.markdown("""
            <div class="scanning-animation">
                <p class="scanner-text">🔍 Scanning URL...</p>
            </div>
            """, unsafe_allow_html=True)

            try:
                result = predict_url(url)
                feature_dict = result.get("features", {})
            except PredictionError as exc:
                st.error("Unable to analyze the URL. Please verify the address and try again.")
                st.stop()
            except FileNotFoundError:
                st.error("The prediction model is currently unavailable. Please contact support.")
                st.stop()
            except Exception:
                st.error("An unexpected error occurred while analyzing the URL. Please try again later.")
                st.stop()

        is_phishing = result["prediction"] == 1
        confidence = result["confidence"]
        prediction = LABEL_PHISHING if is_phishing else LABEL_LEGITIMATE

        save_prediction(
            url=url,
            prediction=prediction,
            confidence=confidence
        )

        show_result_card(url, result)
        show_recommendation(is_phishing)

        reasons = explain_prediction(feature_dict)
        show_feature_panel(feature_dict, reasons)

# =============================================================
# Prediction History Analytics (Single-Pass Load & Precomputed Metrics)
# =============================================================

history_df = load_history()
metrics_dict = calculate_history_metrics(history_df)

st.markdown("---")
show_history_metrics(metrics_dict)
show_prediction_history(history_df)
show_history_charts(history_df, metrics_dict)
show_recent_activity(history_df)
show_history_controls(history_df, clear_history)

# =============================================================
# Footer
# =============================================================

st.markdown(f"""
<div class="footer">
    <p>Built by <strong>Mohit Sain</strong> • {APP_NAME} Project</p>
    <p>Powered by {MODEL_NAME} ML • {TOTAL_FEATURE_COUNT} Lexical URL Features • 100k+ Training URLs</p>
</div>
""", unsafe_allow_html=True)
