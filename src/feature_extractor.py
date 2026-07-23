"""Feature extractor module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

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

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    # -----------------------------
    # Basic Features
    # -----------------------------

    url_length = len(url)

    https = 1 if parsed.scheme == "https" else 0

    dots = url.count(".")
    
    many_dots = int(dots >= 4)
    
    parameter_count = query.count("&") + (1 if query else 0)
    
    long_url = int(url_length > 75)
    
    domain_labels = hostname.split(".") if hostname else []

    www_count = domain_labels.count("www")

    if domain_labels[:1] == ["www"]:
        www_count -= 1
    
    contains_email = int(
        bool(
            re.search(
                r"\S+@\S+",
                url
            )
        )
    )
    
    starts_with_digit = int(
        domain[:1].isdigit()
    )
    
    multiple_special = int(
        bool(
            re.search(r"[-_?=&]{2,}", url)
        )
    )

    hyphen = 1 if "-" in url else 0

    at_symbol = 1 if "@" in url else 0

    shorteners = {
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly"
    }

    url_shortener = int(
        any(
            domain.endswith(short)
            for short in shorteners
        )
    )
    
    repetition_url = re.sub(
        r"(?<=://)www\.",
        "",
        url,
        count=1
    )

    repeated_chars = int(
        bool(
            re.search(r"([a-zA-Z])\1{2,}", repetition_url)
        )
    )
    
    # -----------------------------
    # Domain Features
    # -----------------------------

    domain = hostname.lower()

    if domain.startswith("www."):
        domain = domain[4:]
    
    domain_length = len(domain)
    
    long_domain = int(domain_length > 30)

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
    tld_length = len(tld)

    suspicious_tld = 1 if tld in suspicious_tlds else 0
    
    domain_has_digits = int(any(c.isdigit() for c in domain))
    
    double_hyphen = int("--" in url)
    
    directory_depth = len([p for p in path.split("/") if p])

    # -----------------------------
    # IP Address Detection
    # -----------------------------

    ip_pattern = r"^(?:\d{1,3}\.){3}\d{1,3}$"

    ip_address = 1 if re.match(ip_pattern, domain) else 0

    # -----------------------------
    # Suspicious Keywords
    # -----------------------------

    has_port = int(parsed.port is not None)

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
    
    suspicious_extensions = (
        ".php",
        ".exe",
        ".zip",
        ".rar",
        ".scr",
        ".js"
    )

    has_suspicious_extension = int(
        path.endswith(suspicious_extensions)
    )
    
    brands = [
       "google",
        "paypal",
        "amazon",
        "microsoft",
        "apple",
        "facebook",
        "instagram",
        "netflix",
        "bank",
        "gmail"
    ]
    
    brand_count = sum(
        brand in url
        for brand in brands
    )

    # -----------------------------
    # Character Features
    # -----------------------------

    digits = sum(c.isdigit() for c in url)

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
        "many_dots": many_dots,
        "parameter_count": parameter_count,
        "long_url": long_url,
        "www_count": www_count,
        "contains_email": contains_email,
        "starts_with_digit": starts_with_digit,
        "multiple_special": multiple_special,
        "hyphen": hyphen,
        "at_symbol": at_symbol,
        "url_shortener": url_shortener,
        "repeated_chars": repeated_chars,
        "subdomain_count": subdomain_count,
        "long_domain": long_domain,
        "suspicious_tld": suspicious_tld,
        "domain_has_digits": domain_has_digits,
        "double_hyphen": double_hyphen,
        "directory_depth": directory_depth,
        "ip_address": ip_address,
        "has_port": has_port,
        "keyword_count": keyword_count,
        "has_suspicious_extension": has_suspicious_extension,
        "brand_count": brand_count,
        "digits": digits,
        "special_characters": special_characters,
        "digit_ratio": digit_ratio,
        "special_character_ratio": special_character_ratio,
        "slashes": slashes,
        "question_marks": question_marks,
        "equal_signs": equal_signs,
        "ampersands": ampersands,
        "underscores": underscores,
        "entropy": entropy,
        "tld_length": tld_length,
    }
