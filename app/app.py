import streamlit as st
import sys
import os
import pandas as pd

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from predict import predict_from_url

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
    page_title="AI Phishing Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

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
    st.markdown("""
    - **Algorithm**: Random Forest
    - **Features**: 33 lexical URL features
    - **Accuracy**: ~99.5%
    - **Training Data**: 100k+ URLs
    """)

    st.markdown("---")

    st.markdown("### 🔬 How It Works")
    st.markdown("""
    1. **Extract** 33 features from the URL
    2. **Analyze** using Random Forest ML model
    3. **Predict** phishing probability
    4. **Report** risk level & confidence
    """)


# =============================================================
# Hero Section
# =============================================================

st.markdown("""
<div class="hero-title">
    <h1>🛡️ PhishGuard AI</h1>
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

    # --- Validation ---
    if not url or url.strip() == "":
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
                result = predict_from_url(url)
                feature_dict = result.get("features", {})
            except Exception as e:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-color: rgba(248, 113, 113, 0.3);">
                    <p style="color: #f87171; font-size: 1.1rem; font-weight: 600; margin: 0;">
                        ❌ Error analyzing URL: {str(e)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.stop()

        # --- Determine risk ---
        is_phishing = result["prediction"] == 1
        confidence = result["confidence"]
        
        prediction = "Phishing" if is_phishing else "Legitimate"

        save_prediction(
            url=url,
            prediction=prediction,
            confidence=confidence
            )

        if is_phishing:
            if confidence >= 95:
                risk_label = "HIGH RISK"
                risk_class = "risk-high"
                risk_emoji = "🔴"
            elif confidence >= 75:
                risk_label = "MEDIUM RISK"
                risk_class = "risk-medium"
                risk_emoji = "🟠"
            else:
                risk_label = "LOW RISK"
                risk_class = "risk-low"
                risk_emoji = "🟡"
        else:
            risk_label = "SAFE"
            risk_class = "risk-safe"
            risk_emoji = "🟢"

        # ────────────────────────────────────────────
        # Result Card
        # ────────────────────────────────────────────

        badge_class = "danger" if is_phishing else "safe"
        badge_text = "⚠️ PHISHING DETECTED" if is_phishing else "✅ LEGITIMATE WEBSITE"
        badge_icon = "🚨" if is_phishing else "🛡️"

        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{badge_icon}</div>
            <div class="result-badge {badge_class}">{badge_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ────────────────────────────────────────────
        # Scanned URL
        # ────────────────────────────────────────────

        st.markdown('<div class="section-header">🔗 Scanned URL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="url-display">{url}</div>', unsafe_allow_html=True)

        # ────────────────────────────────────────────
        # Metrics Row
        # ────────────────────────────────────────────

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card legit">
                <div class="label">Legitimate</div>
                <div class="value">{result['legitimate_probability']:.1f}%</div>
            </div>
            <div class="metric-card phish">
                <div class="label">Phishing</div>
                <div class="value">{result['phishing_probability']:.1f}%</div>
            </div>
            <div class="metric-card conf">
                <div class="label">Confidence</div>
                <div class="value">{result['confidence']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ────────────────────────────────────────────
        # Risk Level
        # ────────────────────────────────────────────

        st.markdown(f"""
        <div class="risk-indicator {risk_class}">
            <span style="font-size: 1.5rem;">{risk_emoji}</span>
            <span>Risk Level: {risk_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # ────────────────────────────────────────────
        # Recommendation
        # ────────────────────────────────────────────

        if is_phishing:
            rec_text = "🚨 <strong>Do NOT visit this website.</strong> This URL exhibits characteristics commonly associated with phishing attacks. It may attempt to steal your personal information."
            rec_border = "rgba(248, 113, 113, 0.3)"
            rec_bg = "rgba(248, 113, 113, 0.05)"
        else:
            rec_text = "✅ <strong>This URL appears safe.</strong> Our analysis indicates this is a legitimate website. However, always exercise caution when entering personal information online."
            rec_border = "rgba(74, 222, 128, 0.3)"
            rec_bg = "rgba(74, 222, 128, 0.05)"

        st.markdown(f"""
        <div style="
            background: {rec_bg};
            border: 1px solid {rec_border};
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin: 1rem 0;
            color: #e2e8f0;
            font-size: 0.95rem;
            line-height: 1.6;
        ">
            {rec_text}
        </div>
        """, unsafe_allow_html=True)

        # ────────────────────────────────────────────
        # Explanation & Feature Inspection
        # ────────────────────────────────────────────

        reasons = explain_prediction(feature_dict)

        st.subheader("🔍 Why this prediction?")

        for reason in reasons:
            st.write(f"• {reason}")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # =============================================================
        # Feature Inspection Panel
        # =============================================================

        with st.expander("🔬 Feature Inspection Panel", expanded=False):
            
            st.markdown("""
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
            Below are the 33 lexical features extracted from the URL. These features are analyzed by the Random Forest model to predict phishing.
            </p>
            """, unsafe_allow_html=True)
            
            # Create tabs for feature categories
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📏 URL Structure",
                "🌐 Domain & TLD",
                "🔤 Character Analysis",
                "⚠️ Suspicious Patterns",
                "📊 Advanced Metrics"
            ])
            
            # ─────────────────────────────────────────────────
            # Tab 1: URL Structure
            # ─────────────────────────────────────────────────
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("URL Length", f"{feature_dict.get('url_length', 0)} chars")
                    st.metric("Long URL (>75)", "✅ Yes" if feature_dict.get('long_url', 0) == 1 else "✅ No")
                    st.metric("Parameter Count", feature_dict.get('parameter_count', 0))
                    st.metric("Directory Depth", feature_dict.get('directory_depth', 0))
                
                with col2:
                    st.metric("Total Dots", feature_dict.get('dots', 0))
                    st.metric("Many Dots (≥4)", "⚠️ Yes" if feature_dict.get('many_dots', 0) == 1 else "✅ No")
                    st.metric("Slashes", feature_dict.get('slashes', 0))
                    st.metric("Question Marks", feature_dict.get('question_marks', 0))
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Equal Signs", feature_dict.get('equal_signs', 0))
                    st.metric("Ampersands", feature_dict.get('ampersands', 0))
                with col2:
                    st.metric("Underscores", feature_dict.get('underscores', 0))
                    st.metric("HTTPS", "🔒 Yes" if feature_dict.get('https', 0) == 1 else "⚠️ No")
            
            # ─────────────────────────────────────────────────
            # Tab 2: Domain & TLD
            # ─────────────────────────────────────────────────
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Domain Length", f"{feature_dict.get('domain_length', 0)} chars")
                    st.metric("Long Domain", "⚠️ Yes" if feature_dict.get('long_domain', 0) == 1 else "✅ No")
                    st.metric("Subdomains", feature_dict.get('subdomain_count', 0))
                    st.metric("Domain Has Digits", "⚠️ Yes" if feature_dict.get('domain_has_digits', 0) == 1 else "✅ No")
                
                with col2:
                    st.metric("TLD Length", feature_dict.get('tld_length', 0))
                    st.metric("Suspicious TLD", "⚠️ Yes" if feature_dict.get('suspicious_tld', 0) == 1 else "✅ No")
                    st.metric("WWW Count", feature_dict.get('www_count', 0))
                    st.metric("Contains Email", "⚠️ Yes" if feature_dict.get('contains_email', 0) == 1 else "✅ No")
            
            # ─────────────────────────────────────────────────
            # Tab 3: Character Analysis
            # ─────────────────────────────────────────────────
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Digits", feature_dict.get('digits', 0))
                    st.metric("Digit Ratio", f"{feature_dict.get('digit_ratio', 0):.2%}")
                    st.metric("Special Characters", feature_dict.get('special_characters', 0))
                    st.metric("Special Char Ratio", f"{feature_dict.get('special_character_ratio', 0):.2%}")
                
                with col2:
                    st.metric("Hyphen Count", "⚠️ Yes" if feature_dict.get('hyphen', 0) == 1 else "✅ No")
                    st.metric("Double Hyphen", "⚠️ Yes" if feature_dict.get('double_hyphen', 0) == 1 else "✅ No")
                    st.metric("@ Symbol", "⚠️ Yes" if feature_dict.get('at_symbol', 0) == 1 else "✅ No")
                    st.metric("Repeated Chars", "⚠️ Yes" if feature_dict.get('repeated_chars', 0) == 1 else "✅ No")
            
            # ─────────────────────────────────────────────────
            # Tab 4: Suspicious Patterns
            # ─────────────────────────────────────────────────
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("URL Shortener", "⚠️ Yes" if feature_dict.get('url_shortener', 0) == 1 else "✅ No")
                    st.metric("IP Address", "⚠️ Yes" if feature_dict.get('ip_address', 0) == 1 else "✅ No")
                    st.metric("Has Port", "⚠️ Yes" if feature_dict.get('has_port', 0) == 1 else "✅ No")
                    st.metric("Suspicious Extension", "⚠️ Yes" if feature_dict.get('has_suspicious_extension', 0) == 1 else "✅ No")
                
                with col2:
                    st.metric("Multiple Special Chars", "⚠️ Yes" if feature_dict.get('multiple_special', 0) == 1 else "✅ No")
                    st.metric("Keyword Count", feature_dict.get('keyword_count', 0))
                    st.metric("Brand Count", feature_dict.get('brand_count', 0))
                    st.metric("Starts with Digit", "⚠️ Yes" if feature_dict.get('starts_with_digit', 0) == 1 else "✅ No")
            
            # ─────────────────────────────────────────────────
            # Tab 5: Advanced Metrics
            # ─────────────────────────────────────────────────
            with tab5:
                col1, col2 = st.columns(2)
                
                with col1:
                    entropy = feature_dict.get('entropy', 0)
                    st.metric("URL Entropy", f"{entropy:.3f}")
                    if entropy > 4.5:
                        st.write("🔴 High entropy (suspicious)")
                    elif entropy < 2.0:
                        st.write("🟢 Low entropy (normal)")
                    else:
                        st.write("🟡 Medium entropy")
                
                with col2:
                    pass
                
                st.markdown("---")
                st.markdown("**Entropy Interpretation:**")
                st.write("""
                - **Low entropy (<2.0)**: URL contains predictable patterns (legitimate)
                - **Medium entropy (2.0-4.5)**: Mix of predictable and random characters
                - **High entropy (>4.5)**: URL contains mostly random characters (suspicious)
                """)
            
            # ─────────────────────────────────────────────────
            # Feature Summary Table
            # ─────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("**Complete Feature Dictionary:**")
            
            # Create a formatted table
            feature_df = pd.DataFrame([feature_dict]).T
            feature_df.columns = ["Value"]
            feature_df = feature_df.reset_index()
            feature_df.columns = ["Feature", "Value"]
            
            # Format values
            feature_df["Value"] = feature_df["Value"].apply(
                lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            )
            
            st.dataframe(
                feature_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Feature": st.column_config.TextColumn("Feature Name", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="small"),
                }
            )

# =============================================================
# Prediction History
# =============================================================

history_df = load_history()
st.markdown("---")
st.subheader("📜 Prediction History")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Predictions", len(history_df))

with col2:
    st.metric(
        "Phishing",
        len(history_df[history_df["Prediction"] == "Phishing"])
    )

with col3:
    st.metric(
        "Legitimate",
        len(history_df[history_df["Prediction"] == "Legitimate"])
    )
    
if st.button("🗑️ Clear History"):

    clear_history()

    st.success("Prediction history cleared successfully.")

    st.rerun()

if not history_df.empty:

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="📥 Download History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
    
if history_df.empty:
    st.info("No prediction history available.")
else:
    st.dataframe(
        history_df,
        width="stretch",
        hide_index=True
    )
    


# =============================================================
# Footer
# =============================================================

st.markdown("""
<div class="footer">
    <p>Built by <strong>Mohit Sain</strong> • AI Phishing Detection Project</p>
    <p>Powered by Random Forest ML • 33 Lexical URL Features • 100k+ Training URLs</p>
</div>
""", unsafe_allow_html=True)
