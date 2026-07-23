"""Result ui module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import streamlit as st
from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, LABEL_LEGITIMATE, LABEL_PHISHING


def show_result_card(url, result):
    """Show result card.
    
    Parameters
    ----------
    url : TYPE
        Description of url.
    result : TYPE
        Description of result.
    """
    is_phishing = result["prediction"] == 1
    confidence = result["confidence"]

    if is_phishing:
        if confidence >= HIGH_RISK_THRESHOLD:
            risk_label = "HIGH RISK"
            risk_class = "risk-high"
            risk_emoji = "🔴"
        elif confidence >= MEDIUM_RISK_THRESHOLD:
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

    badge_class = "danger" if is_phishing else "safe"
    badge_text = f"⚠️ {LABEL_PHISHING.upper()} DETECTED" if is_phishing else f"✅ {LABEL_LEGITIMATE.upper()} WEBSITE"
    badge_icon = "🚨" if is_phishing else "🛡️"

    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{badge_icon}</div>
        <div class="result-badge {badge_class}">{badge_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔗 Scanned URL</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="url-display">{url}</div>', unsafe_allow_html=True)

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

    st.markdown(f"""
    <div class="risk-indicator {risk_class}">
        <span style="font-size: 1.5rem;">{risk_emoji}</span>
        <span>Risk Level: {risk_label}</span>
    </div>
    """, unsafe_allow_html=True)
