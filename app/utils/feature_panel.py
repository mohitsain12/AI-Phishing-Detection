"""Feature panel module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import pandas as pd
import streamlit as st
from config import TOTAL_FEATURE_COUNT, MODEL_NAME


def show_feature_panel(feature_dict, reasons):
    """Show feature panel.
    
    Parameters
    ----------
    feature_dict : TYPE
        Description of feature_dict.
    reasons : TYPE
        Description of reasons.
    """
    st.subheader("🔍 Why this prediction?")

    for reason in reasons:
        st.write(f"• {reason}")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.expander("🔬 Feature Inspection Panel", expanded=False):
        st.markdown(
            f"""
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
            Below are the {TOTAL_FEATURE_COUNT} lexical features extracted from the URL. These features are analyzed by the {MODEL_NAME} model to predict phishing.
            </p>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📏 URL Structure",
            "🌐 Domain & TLD",
            "🔤 Character Analysis",
            "⚠️ Suspicious Patterns",
            "📊 Advanced Metrics"
        ])

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
            st.write(
                """
                - **Low entropy (<2.0)**: URL contains predictable patterns (legitimate)
                - **Medium entropy (2.0-4.5)**: Mix of predictable and random characters
                - **High entropy (>4.5)**: URL contains mostly random characters (suspicious)
                """
            )

        st.markdown("---")
        st.markdown("**Complete Feature Dictionary:**")

        feature_df = pd.DataFrame([feature_dict]).T
        feature_df.columns = ["Value"]
        feature_df = feature_df.reset_index()
        feature_df.columns = ["Feature", "Value"]
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
