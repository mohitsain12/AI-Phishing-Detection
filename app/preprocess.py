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

    features_df["label"] = df["label"]

    print("Saving features...")
    features_df.to_csv("../data/features.csv", index=False)

    print("✅ Features saved successfully!")
    print(features_df.head())

if __name__ == "__main__":
    main()