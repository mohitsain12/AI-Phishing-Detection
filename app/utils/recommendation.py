"""Recommendation module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import streamlit as st


def show_recommendation(is_phishing):
    """Show recommendation.
    
    Parameters
    ----------
    is_phishing : TYPE
        Description of is_phishing.
    """
    if is_phishing:
        rec_text = (
            "🚨 <strong>Do NOT visit this website.</strong> "
            "This URL exhibits characteristics commonly associated with phishing attacks. "
            "It may attempt to steal your personal information."
        )
        rec_border = "rgba(248, 113, 113, 0.3)"
        rec_bg = "rgba(248, 113, 113, 0.05)"
    else:
        rec_text = (
            "✅ <strong>This URL appears safe.</strong> "
            "Our analysis indicates this is a legitimate website. However, always exercise caution when entering personal information online."
        )
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
