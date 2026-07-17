import streamlit as st
import sys
import os

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from predict import predict_from_url

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🎣",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎣 AI Powered Phishing URL Detector")

st.write(
    "Detect phishing websites using Machine Learning."
)

st.divider()
with st.sidebar:

    st.title("About")

    st.write(
        """
        AI Powered Phishing URL Detector

        Model:
        Random Forest

        Dataset:
        Phishing Websites Dataset
        """
    )
# -----------------------------
# URL Input
# -----------------------------
url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com/login"
)

# -----------------------------
# Analyze Button
# -----------------------------
analyze = st.button(" 🔍 Analyze URL")

st.divider()

# -----------------------------
# Placeholder Result
# -----------------------------
if analyze:

    if url == "":
        st.warning("Please enter a URL.")

    else:
        result = predict_from_url(url)

    st.header("Prediction")
    st.subheader("Scanned URL")
    st.code(url)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Legitimate",
            f"{result['legitimate_probability']:.2f}%"
        )

    with col2:
        st.metric(
            "Phishing",
            f"{result['phishing_probability']:.2f}%"
        )

    st.subheader("Confidence")
    st.progress(result["confidence"] / 100)
    st.write(f"{result['confidence']:.2f}%")

    st.write(
        f"Legitimate Probability : {result['legitimate_probability']:.2f}%"
    )

    st.write(
        f"Phishing Probability : {result['phishing_probability']:.2f}%"
    )

    # -----------------------------
    # Risk Level
    # -----------------------------

    if result["prediction"] == 1:

        if result["confidence"] >= 95:
            risk = "🔴 HIGH"

        elif result["confidence"] >= 75:
            risk = "🟠 MEDIUM"

        else:
            risk = "🟡 LOW"

    else:
        risk = "🟢 SAFE"
        
    if result["prediction"] == 0:
        recommendation = "✅ Safe to visit."
    else:
        recommendation = "🚨 Do NOT visit this website."
        
    st.subheader("Recommendation")
    st.info(recommendation)
    
    st.write(f"### Risk Level : {risk}")
    

st.divider()

st.caption(
    "Developed by Mohit Sain | AI Powered Phishing Detection Project"
)