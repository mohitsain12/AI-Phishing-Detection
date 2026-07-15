import pandas as pd
from feature_extractor import extract_features

def main():
    """
    Read raw URLs, extract lexical features,
    and save the processed dataset.
    """

    print("Loading dataset...")
    df = pd.read_csv("../data/raw_urls.csv")

    print(f"Dataset Loaded: {len(df)} URLs")

    urls = df["URL"]

    print("Extracting features...")
    feature_data = urls.apply(extract_features)

    features_df = pd.DataFrame(feature_data.tolist())
    
    # Convert labels to project standard
    # Original UCI:
    # 1 = Legitimate
    # 0 = Phishing

    # Project Standard:
    # 0 = Legitimate
    # 1 = Phishing
    features_df["label"] = df["label"].map({
        1: 0,   # Legitimate
        0: 1    # Phishing
    })

    print("Saving features...")
    features_df.to_csv("../data/features.csv", index=False)

    print("✅ Features saved successfully!")
    print(features_df.head())

if __name__ == "__main__":
    main()