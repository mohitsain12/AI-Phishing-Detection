"""Explainer module for AI Phishing Detection.

Generates human-readable explanation reason strings based on extracted URL feature dictionaries.
"""

import sys
from pathlib import Path

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    MAX_URL_LENGTH_THRESHOLD,
    PARAMETER_COUNT_THRESHOLD,
    SUBDOMAIN_COUNT_THRESHOLD,
    MAX_DIRECTORY_DEPTH_THRESHOLD,
    HIGH_ENTROPY_THRESHOLD
)


def explain_prediction(feature_dictionary):
    """Generate list of human-readable rationale statements from feature dictionary.

    Parameters
    ----------
    feature_dictionary : dict
        Extracted URL features dictionary.

    Returns
    -------
    list
        List of explanation reason strings.
    """
    reasons = []

    # Guard Clause: Trusted Domain Whitelist Override
    if feature_dictionary.get("trusted_domain", 0) == 1:
        return [
            "This URL belongs to a trusted domain and has been treated as safe despite weak suspicious signals."
        ]

    if feature_dictionary.get("url_length", 0) > MAX_URL_LENGTH_THRESHOLD:
        reasons.append(
            f"The URL is unusually long (>{MAX_URL_LENGTH_THRESHOLD} characters), often used to hide phishing intent"
        )

    if feature_dictionary.get("https", 0) == 0:
        reasons.append("The URL does not use HTTPS encryption, which is a security risk")
    else:
        reasons.append("The URL uses HTTPS encryption, indicating a secure connection")

    if feature_dictionary.get("many_dots", 0) == 1:
        reasons.append("The URL contains many dots (≥4), indicating a suspicious domain structure")

    param_count = feature_dictionary.get("parameter_count", 0)
    if param_count > PARAMETER_COUNT_THRESHOLD:
        reasons.append(f"The URL contains {param_count} parameters, which is uncommon")

    if feature_dictionary.get("multiple_special", 0) == 1:
        reasons.append("The URL contains multiple special characters, which may indicate phishing")

    if feature_dictionary.get("hyphen", 0) == 1:
        reasons.append("The domain contains hyphens, sometimes used in phishing URLs")

    if feature_dictionary.get("at_symbol", 0) == 1:
        reasons.append("The URL contains '@' symbol, which can mask the real domain")

    if feature_dictionary.get("url_shortener", 0) == 1:
        reasons.append("The URL uses a shortened URL service, which can hide the real destination")

    if feature_dictionary.get("repeated_chars", 0) == 1:
        reasons.append("The URL contains repeated characters, may indicate domain spoofing")

    subdomain_count = feature_dictionary.get("subdomain_count", 0)
    if subdomain_count > SUBDOMAIN_COUNT_THRESHOLD:
        reasons.append(f"The URL has {subdomain_count} subdomains, which is unusual")

    if feature_dictionary.get("long_domain", 0) == 1:
        reasons.append("The domain name is unusually long, sometimes used to confuse users")

    if feature_dictionary.get("suspicious_tld", 0) == 1:
        reasons.append("The URL uses a suspicious top-level domain (TLD)")

    if feature_dictionary.get("domain_has_digits", 0) == 1:
        reasons.append("The domain contains digits, which legitimate domains rarely have")

    if feature_dictionary.get("double_hyphen", 0) == 1:
        reasons.append("The domain contains double hyphens, a known phishing pattern")

    directory_depth = feature_dictionary.get("directory_depth", 0)
    if directory_depth > MAX_DIRECTORY_DEPTH_THRESHOLD:
        reasons.append(f"The URL has deep directory nesting ({directory_depth} levels), which is unusual")

    if feature_dictionary.get("ip_address", 0) == 1:
        reasons.append("The URL uses an IP address instead of a domain name, which is suspicious")

    if feature_dictionary.get("has_port", 0) == 1:
        reasons.append("The URL explicitly specifies a port, which is unusual for legitimate websites")

    if feature_dictionary.get("keyword_count", 0) > 0:
        reasons.append("The URL contains suspicious keywords")

    if feature_dictionary.get("has_suspicious_extension", 0) == 1:
        reasons.append("The URL uses a suspicious file extension")

    if feature_dictionary.get("brand_count", 0) > 0:
        reasons.append("The URL contains brand names, potentially impersonating legitimate companies")

    entropy_score = feature_dictionary.get("entropy", 0.0)
    if entropy_score > HIGH_ENTROPY_THRESHOLD:
        reasons.append(
            f"The URL has high entropy ({entropy_score:.2f}), indicating randomness typical of phishing URLs"
        )

    if not reasons:
        reasons.append("No obvious phishing indicators were detected in this URL")

    return reasons