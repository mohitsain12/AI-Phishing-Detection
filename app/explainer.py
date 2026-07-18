def explain_prediction(feature_dict):
    """
    Generate human-readable reasons based on extracted URL features.
    """

    reasons = []

    # --- URL Length ---
    if feature_dict.get("url_length", 0) > 75:
        reasons.append("The URL is unusually long (>75 characters), often used to hide phishing intent")
    
    # --- HTTPS ---
    if feature_dict.get("https", 0) == 0:
        reasons.append("The URL does not use HTTPS encryption, which is a security risk")
    else:
        reasons.append("The URL uses HTTPS encryption, indicating a secure connection")
    
    # --- Many Dots ---
    if feature_dict.get("many_dots", 0) == 1:
        reasons.append("The URL contains many dots (≥4), indicating a suspicious domain structure")
    
    # --- Parameters ---
    if feature_dict.get("parameter_count", 0) > 3:
        reasons.append(f"The URL contains {feature_dict.get('parameter_count', 0)} parameters, which is uncommon")
    
    # --- Multiple Special Characters ---
    if feature_dict.get("multiple_special", 0) == 1:
        reasons.append("The URL contains multiple special characters, which may indicate phishing")
    
    # --- Hyphens ---
    if feature_dict.get("hyphen", 0) == 1:
        reasons.append("The domain contains hyphens, sometimes used in phishing URLs")
    
    # --- @ Symbol ---
    if feature_dict.get("at_symbol", 0) == 1:
        reasons.append("The URL contains '@' symbol, which can mask the real domain")
    
    # --- URL Shortener ---
    if feature_dict.get("url_shortener", 0) == 1:
        reasons.append("The URL uses a shortened URL service, which can hide the real destination")
    
    # --- Repeated Characters ---
    if feature_dict.get("repeated_chars", 0) == 1:
        reasons.append("The URL contains repeated characters, may indicate domain spoofing")
    
    # --- Subdomains ---
    if feature_dict.get("subdomain_count", 0) > 2:
        reasons.append(f"The URL has {feature_dict.get('subdomain_count', 0)} subdomains, which is unusual")
    
    # --- Long Domain ---
    if feature_dict.get("long_domain", 0) == 1:
        reasons.append("The domain name is unusually long, sometimes used to confuse users")
    
    # --- Suspicious TLD ---
    if feature_dict.get("suspicious_tld", 0) == 1:
        reasons.append("The URL uses a suspicious top-level domain (TLD)")
    
    # --- Domain Has Digits ---
    if feature_dict.get("domain_has_digits", 0) == 1:
        reasons.append("The domain contains digits, which legitimate domains rarely have")
    
    # --- Double Hyphen ---
    if feature_dict.get("double_hyphen", 0) == 1:
        reasons.append("The domain contains double hyphens, a known phishing pattern")
    
    # --- Deep Directory ---
    if feature_dict.get("directory_depth", 0) > 5:
        reasons.append(f"The URL has deep directory nesting, which is unusual")
    
    # --- IP Address ---
    if feature_dict.get("ip_address", 0) == 1:
        reasons.append("The URL uses an IP address instead of a domain name, which is suspicious")
    
    # --- Port ---
    if feature_dict.get("has_port", 0) == 1:
        reasons.append("The URL explicitly specifies a port, which is unusual for legitimate websites")
    
    # --- Keywords ---
    if feature_dict.get("keyword_count", 0) > 0:
        reasons.append(f"The URL contains suspicious keywords")
    
    # --- Suspicious Extension ---
    if feature_dict.get("has_suspicious_extension", 0) == 1:
        reasons.append("The URL uses a suspicious file extension")
    
    # --- Brand ---
    if feature_dict.get("brand_count", 0) > 0:
        reasons.append("The URL contains brand names, potentially impersonating legitimate companies")
    
    # --- Entropy ---
    entropy = feature_dict.get("entropy", 0)
    if entropy > 4.5:
        reasons.append(f"The URL has high entropy ({entropy:.2f}), indicating randomness typical of phishing URLs")
    
    # Default
    if len(reasons) == 0:
        reasons.append("No obvious phishing indicators were detected in this URL")

    return reasons