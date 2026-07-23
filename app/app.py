"""App module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add app, root, and src folders to Python path
APP_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(APP_DIR, "../src")))

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

logger.info("App started")

# =============================================================
# Custom CSS — Premium Dark Theme with Glassmorphism
# =============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ─────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 25%, #1b2838 50%, #0d1b2a 75%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default header/footer */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ── Sidebar ────────────────────────────────────── */
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

    /* ── Hero Title ─────────────────────────────────── */
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

    /* ── Glass Card ─────────────────────────────────── */
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

    /* ── Metric Cards ──────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
    }

    .metric-card .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 800;
    }

    .metric-card.legit .value {
        color: #4ade80;
    }

    .metric-card.phish .value {
        color: #f87171;
    }

    .metric-card.conf .value {
        color: #60a5fa;
    }

    /* ── Result Badge ──────────────────────────────── */
    .result-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1rem 0;
        animation: pulse-glow 2s ease-in-out infinite;
    }

    .result-badge.safe {
        background: linear-gradient(135deg, rgba(74, 222, 128, 0.2), rgba(34, 197, 94, 0.1));
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.4);
    }

    .result-badge.danger {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.2), rgba(239, 68, 68, 0.1));
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* ── Risk Level Bars ───────────────────────────── */
    .risk-indicator {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
    }

    .risk-safe {
        background: rgba(74, 222, 128, 0.1);
        border: 1px solid rgba(74, 222, 128, 0.3);
        color: #4ade80;
    }

    .risk-low {
        background: rgba(250, 204, 21, 0.1);
        border: 1px solid rgba(250, 204, 21, 0.3);
        color: #facc15;
    }

    .risk-medium {
        background: rgba(251, 146, 60, 0.1);
        border: 1px solid rgba(251, 146, 60, 0.3);
        color: #fb923c;
    }

    .risk-high {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: #f87171;
    }

    /* ── Progress Bar Override ─────────────────────── */
    .stProgress > div > div > div > div {
        border-radius: 10px;
    }

    /* ── URL Display ───────────────────────────────── */
    .url-display {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: #60a5fa;
        word-break: break-all;
        margin: 0.5rem 0 1.5rem 0;
    }

    /* ── Feature Table ─────────────────────────────── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 0.6rem;
        margin: 1rem 0;
    }

    .feature-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        transition: all 0.2s ease;
    }

    .feature-item:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(100, 200, 255, 0.2);
    }

    .feature-name {
        color: #94a3b8;
        font-weight: 500;
    }

    .feature-value {
        color: #e2e8f0;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Scanning Animation ────────────────────────── */
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

    /* ── Section Headers ───────────────────────────── */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        letter-spacing: 0.5px;
    }

    /* ── Example URL Buttons ───────────────────────── */
    .example-url {
        display: block;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        color: #94a3b8;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        word-break: break-all;
        transition: all 0.2s ease;
    }

    .example-url:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
    }

    /* ── Divider ───────────────────────────────────── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 200, 255, 0.2), transparent);
        margin: 2rem 0;
        border: none;
    }

    /* ── Footer ────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: 2rem;
    }

    .footer a {
        color: #60a5fa;
        text-decoration: none;
    }

    /* ── Streamlit Overrides ───────────────────────── */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(96, 165, 250, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        color: #94a3b8 !important;
        padding: 0.5rem 1.2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(96, 165, 250, 0.15) !important;
        color: #60a5fa !important;
        border-color: rgba(96, 165, 250, 0.3) !important;
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

    # --- Validation ---
    if not url or url.strip() == "":
        logger.warning("Invalid URL entered: %s", url)
        st.markdown("""
        <div class="glass-card" style="text-align: center; border-color: rgba(250, 204, 21, 0.3);">
            <p style="color: #facc15; font-size: 1.1rem; font-weight: 600; margin: 0;">
                ⚠️ Please enter a URL to analyze
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # --- Scanning animation ---
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
                logger.warning("User-correctable prediction error for URL %s: %s", url, exc)
                st.error("Unable to analyze the URL. Please verify the address and try again.")
                st.stop()
            except FileNotFoundError:
                logger.exception("Missing prediction model while analyzing URL %s", url)
                st.error("The prediction model is currently unavailable. Please contact support.")
                st.stop()
            except Exception:
                logger.exception("Unexpected error analyzing URL %s", url)
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
# Prediction History
# =============================================================

history_df = load_history()
st.markdown("---")
show_history_metrics(history_df)
show_prediction_history(history_df)
show_history_charts(history_df)
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
