import re

def extract_features(url):
    """
    Extract lexical features from a URL.

    Returns:
        dict: Dictionary containing extracted URL features.
    """

    # Convert URL to lowercase
    url = url.lower()

    # -------------------------
    # Basic Features
    # -------------------------

    url_length = len(url)

    https = 1 if url.startswith("https") else 0

    dots = url.count(".")

    at_symbol = 1 if "@" in url else 0

    hyphen = 1 if "-" in url else 0

    # -------------------------
    # IP Address Detection
    # -------------------------

    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    ip_address = 1 if re.search(ip_pattern, url) else 0

    # -------------------------
    # Suspicious Keywords
    # -------------------------

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
        "bank",
        "paypal",
        "amazon",
        "google",
        "microsoft",
        "office365",
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
        1 for keyword in keywords
        if keyword in url
    )

    # -------------------------
    # Additional Lexical Features
    # -------------------------

    digits = sum(char.isdigit() for char in url)

    letters = sum(char.isalpha() for char in url)

    special_characters = sum(
        not char.isalnum()
        for char in url
    )

    slashes = url.count("/")

    question_marks = url.count("?")

    equal_signs = url.count("=")

    ampersands = url.count("&")

    underscores = url.count("_")

    # -------------------------
    # Return Feature Dictionary
    # -------------------------

    return {

        "url_length": url_length,

        "https": https,

        "dots": dots,

        "at_symbol": at_symbol,

        "hyphen": hyphen,

        "ip_address": ip_address,

        "keyword_count": keyword_count,

        "digits": digits,

        "letters": letters,

        "special_characters": special_characters,

        "slashes": slashes,

        "question_marks": question_marks,

        "equal_signs": equal_signs,

        "ampersands": ampersands,

        "underscores": underscores
    }