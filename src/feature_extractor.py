"""Feature extractor module for AI Phishing Detection.

Provides lexical feature extraction from URL strings using modular,
single-responsibility helper functions and centralized configuration constants.
"""

import math
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    MAX_URL_LENGTH_THRESHOLD,
    MAX_DOMAIN_LENGTH_THRESHOLD,
    MANY_DOTS_THRESHOLD,
    URL_SHORTENERS,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_EXTENSIONS,
    BRAND_NAMES
)


def calculate_entropy(text):
    """Calculate Shannon entropy of a string.

    Higher entropy indicates greater randomness in character distribution.

    Parameters
    ----------
    text : str
        The input string to analyze.

    Returns
    -------
    float
        Rounded Shannon entropy value.
    """
    if not text:
        return 0.0

    entropy = 0.0
    text_length = len(text)

    for character in set(text):
        probability = text.count(character) / text_length
        entropy -= probability * math.log2(probability)

    return round(entropy, 3)


def _extract_basic_url_features(parsed_url, url_text):
    """Extract general structural and syntax features from a URL.

    Parameters
    ----------
    parsed_url : ParseResult
        Parsed urllib URL components.
    url_text : str
        Normalized URL string.

    Returns
    -------
    dict
        Dictionary of basic lexical features.
    """
    url_length = len(url_text)
    is_https = 1 if parsed_url.scheme == "https" else 0
    dot_count = url_text.count(".")
    has_many_dots = int(dot_count >= MANY_DOTS_THRESHOLD)

    query_string = parsed_url.query
    parameter_count = query_string.count("&") + (1 if query_string else 0)
    is_long_url = int(url_length > MAX_URL_LENGTH_THRESHOLD)

    hostname = parsed_url.hostname or ""
    domain_labels = hostname.split(".") if hostname else []
    www_count = domain_labels.count("www")
    if domain_labels[:1] == ["www"]:
        www_count -= 1

    contains_email = int(bool(re.search(r"\S+@\S+", url_text)))
    starts_with_digit = int(bool(hostname[:1].isdigit()))
    multiple_special = int(bool(re.search(r"[-_?=&]{2,}", url_text)))
    has_hyphen = 1 if "-" in url_text else 0
    has_at_symbol = 1 if "@" in url_text else 0

    is_shortener = int(
        any(hostname.endswith(shortener) for shortener in URL_SHORTENERS)
    )

    repetition_target = re.sub(r"(?<=://)www\.", "", url_text, count=1)
    has_repeated_chars = int(
        bool(re.search(r"([a-zA-Z])\1{2,}", repetition_target))
    )

    return {
        "url_length": url_length,
        "https": is_https,
        "dots": dot_count,
        "many_dots": has_many_dots,
        "parameter_count": parameter_count,
        "long_url": is_long_url,
        "www_count": www_count,
        "contains_email": contains_email,
        "starts_with_digit": starts_with_digit,
        "multiple_special": multiple_special,
        "hyphen": has_hyphen,
        "at_symbol": has_at_symbol,
        "url_shortener": is_shortener,
        "repeated_chars": has_repeated_chars
    }


def _extract_domain_features(parsed_url, url_text):
    """Extract domain, subdomain, TLD, and IP address features.

    Parameters
    ----------
    parsed_url : ParseResult
        Parsed urllib URL components.
    url_text : str
        Normalized URL string.

    Returns
    -------
    dict
        Dictionary of domain-specific features.
    """
    hostname = (parsed_url.hostname or "").lower()
    domain = hostname[4:] if hostname.startswith("www.") else hostname

    domain_length = len(domain)
    is_long_domain = int(domain_length > MAX_DOMAIN_LENGTH_THRESHOLD)
    domain_parts = domain.split(".")
    subdomain_count = max(len(domain_parts) - 2, 0)

    top_level_domain = domain_parts[-1] if len(domain_parts) > 1 else ""
    tld_length = len(top_level_domain)
    is_suspicious_tld = 1 if top_level_domain in SUSPICIOUS_TLDS else 0

    domain_has_digits = int(any(char.isdigit() for char in domain))
    has_double_hyphen = int("--" in url_text)

    path_segments = [seg for seg in parsed_url.path.split("/") if seg]
    directory_depth = len(path_segments)

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    is_ip_address = 1 if re.match(ip_pattern, domain) else 0

    return {
        "domain_length": domain_length,
        "long_domain": is_long_domain,
        "subdomain_count": subdomain_count,
        "suspicious_tld": is_suspicious_tld,
        "tld_length": tld_length,
        "domain_has_digits": domain_has_digits,
        "double_hyphen": has_double_hyphen,
        "directory_depth": directory_depth,
        "ip_address": is_ip_address
    }


def _extract_content_and_keyword_features(parsed_url, url_text):
    """Extract keyword matching, extensions, and brand features.

    Parameters
    ----------
    parsed_url : ParseResult
        Parsed urllib URL components.
    url_text : str
        Normalized URL string.

    Returns
    -------
    dict
        Dictionary of keyword and content features.
    """
    has_port = int(parsed_url.port is not None)
    keyword_count = sum(kw in url_text for kw in SUSPICIOUS_KEYWORDS)

    path_string = parsed_url.path
    has_suspicious_ext = int(path_string.endswith(SUSPICIOUS_EXTENSIONS))
    brand_count = sum(brand in url_text for brand in BRAND_NAMES)

    return {
        "has_port": has_port,
        "keyword_count": keyword_count,
        "has_suspicious_extension": has_suspicious_ext,
        "brand_count": brand_count
    }


def _extract_character_and_entropy_features(url_text, url_length):
    """Extract character composition counts, ratios, and Shannon entropy.

    Parameters
    ----------
    url_text : str
        Normalized URL string.
    url_length : int
        Length of the URL string.

    Returns
    -------
    dict
        Dictionary of character and entropy features.
    """
    digit_count = sum(char.isdigit() for char in url_text)
    special_count = sum(not char.isalnum() for char in url_text)

    digit_ratio = round(digit_count / url_length, 3) if url_length else 0.0
    special_ratio = round(special_count / url_length, 3) if url_length else 0.0

    return {
        "digits": digit_count,
        "special_characters": special_count,
        "digit_ratio": digit_ratio,
        "special_character_ratio": special_ratio,
        "slashes": url_text.count("/"),
        "question_marks": url_text.count("?"),
        "equal_signs": url_text.count("="),
        "ampersands": url_text.count("&"),
        "underscores": url_text.count("_"),
        "entropy": calculate_entropy(url_text)
    }


def extract_features(url):
    """Extract all 37 lexical features from a given URL string.

    Parameters
    ----------
    url : str
        URL string to analyze.

    Returns
    -------
    dict
        Extracted features dictionary.
    """
    normalized_url = url.lower().strip()
    if "://" not in normalized_url:
        normalized_url = "https://" + normalized_url

    parsed_url = urlparse(normalized_url)

    basic_features = _extract_basic_url_features(parsed_url, normalized_url)
    domain_features = _extract_domain_features(parsed_url, normalized_url)
    content_features = _extract_content_and_keyword_features(parsed_url, normalized_url)
    character_features = _extract_character_and_entropy_features(
        normalized_url,
        basic_features["url_length"]
    )

    feature_dictionary = {}
    feature_dictionary.update(basic_features)
    feature_dictionary.update(domain_features)
    feature_dictionary.update(content_features)
    feature_dictionary.update(character_features)

    return feature_dictionary
