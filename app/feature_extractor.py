import re
import math
from urllib.parse import urlparse


def calculate_entropy(text):
    """
    Calculate Shannon entropy of a string.
    Higher entropy usually means more randomness.
    """
    if not text:
        return 0

    entropy = 0

    for char in set(text):
        probability = text.count(char) / len(text)
        entropy -= probability * math.log2(probability)

    return round(entropy, 3)


def extract_features(url):
    """
    Extract lexical features from a URL.
    """

    url = url.lower().strip()

    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # -----------------------------
    # Basic Features
    # -----------------------------

    url_length = len(url)

    https = 1 if parsed.scheme == "https" else 0

    dots = url.count(".")

    hyphen = 1 if "-" in url else 0

    at_symbol = 1 if "@" in url else 0

    # -----------------------------
    # Domain Features
    # -----------------------------

    domain_length = len(domain)

    domain_parts = domain.split(".")

    subdomain_count = max(len(domain_parts) - 2, 0)

    suspicious_tlds = {
        "xyz",
        "top",
        "click",
        "gq",
        "ml",
        "cf",
        "tk"
    }

    tld = domain_parts[-1] if len(domain_parts) > 1 else ""

    suspicious_tld = 1 if tld in suspicious_tlds else 0

    # -----------------------------
    # IP Address Detection
    # -----------------------------

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    ip_address = 1 if re.match(ip_pattern, domain) else 0

    # -----------------------------
    # Suspicious Keywords
    # -----------------------------

    keywords = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "security",
        "update",
        "confirm",
        "account",
        "password",
        "wallet",
        "payment",
        "invoice",
        "bonus",
        "gift",
        "reward",
        "winner",
        "free"
    ]

    keyword_count = sum(
        keyword in url
        for keyword in keywords
    )

    # -----------------------------
    # Character Features
    # -----------------------------

    digits = sum(c.isdigit() for c in url)

    letters = sum(c.isalpha() for c in url)

    special_characters = sum(
        not c.isalnum()
        for c in url
    )

    digit_ratio = round(digits / url_length, 3) if url_length else 0

    special_character_ratio = round(
        special_characters / url_length,
        3
    ) if url_length else 0

    # -----------------------------
    # URL Structure
    # -----------------------------

    slashes = url.count("/")

    question_marks = url.count("?")

    equal_signs = url.count("=")

    ampersands = url.count("&")

    underscores = url.count("_")

    # -----------------------------
    # URL Entropy
    # -----------------------------

    entropy = calculate_entropy(url)

    # -----------------------------
    # Return Features
    # -----------------------------

    return {

        "url_length": url_length,

        "domain_length": domain_length,

        "https": https,

        "dots": dots,

        "subdomain_count": subdomain_count,

        "suspicious_tld": suspicious_tld,

        "hyphen": hyphen,

        "at_symbol": at_symbol,

        "ip_address": ip_address,

        "keyword_count": keyword_count,

        "digits": digits,

        "letters": letters,

        "digit_ratio": digit_ratio,

        "special_character_ratio": special_character_ratio,

        "slashes": slashes,

        "question_marks": question_marks,

        "equal_signs": equal_signs,

        "ampersands": ampersands,

        "underscores": underscores,

        "entropy": entropy

    }