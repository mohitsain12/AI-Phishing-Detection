"""Preprocess module for AI Phishing Detection.

Cleans the raw URL dataset, generates www variants, extracts 37 lexical features,
and outputs standardized datasets using centralized configuration paths.
"""

import re
import sys
from pathlib import Path
import pandas as pd

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    PHISHING_DATASET_PATH,
    RAW_URLS_PATH,
    FEATURES_DATASET_PATH
)
from src.feature_extractor import extract_features

LABEL_MAPPING = {
    1: 0,  # 1 in raw data -> Legitimate (0)
    0: 1   # 0 in raw data -> Phishing (1)
}


def create_raw_dataset():
    """Load raw phishing dataset, standardize labels, deduplicate, and save raw_urls.csv.

    Raises
    ------
    ValueError
        If required columns are missing from the input dataset.
    """
    print("=" * 60)
    print("STEP 1 : Creating raw_urls.csv")
    print("=" * 60)

    original_df = pd.read_csv(PHISHING_DATASET_PATH)
    print(f"Original Dataset : {len(original_df)} rows")

    required_columns = ["URL", "label"]
    for column_name in required_columns:
        if column_name not in original_df.columns:
            raise ValueError(f"Missing required column: {column_name}")

    raw_df = original_df[["URL", "label"]].copy()
    raw_df.dropna(subset=["URL", "label"], inplace=True)
    raw_df.drop_duplicates(subset=["URL"], inplace=True)
    raw_df["label"] = raw_df["label"].map(LABEL_MAPPING)
    raw_df.reset_index(drop=True, inplace=True)

    RAW_URLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_URLS_PATH, index=False)

    print(f"Rows after cleaning : {len(raw_df)}")
    print(f"raw_urls.csv saved successfully to {RAW_URLS_PATH}!\n")


def add_www_variants(url_df):
    """Generate www-stripped URL variants to expand training data diversity.

    Parameters
    ----------
    url_df : pd.DataFrame
        DataFrame containing URL and label columns.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with deduplicated www variants.
    """
    variant_rows = []

    for _, row in url_df.iterrows():
        url_text = str(row["URL"])
        variant_url = re.sub(
            r"://www\.",
            "://",
            url_text,
            count=1,
            flags=re.IGNORECASE
        )
        if variant_url != url_text:
            variant_rows.append({
                "URL": variant_url,
                "label": row["label"]
            })

    if not variant_rows:
        return url_df

    augmented_df = pd.concat([url_df, pd.DataFrame(variant_rows)], ignore_index=True)
    augmented_df.drop_duplicates(subset=["URL"], inplace=True)
    augmented_df.reset_index(drop=True, inplace=True)
    return augmented_df


def create_feature_dataset():
    """Extract lexical features for all URLs in raw_urls.csv and save features.csv."""
    print("=" * 60)
    print("STEP 2 : Creating features.csv")
    print("=" * 60)

    raw_df = pd.read_csv(RAW_URLS_PATH)
    print(f"Raw Dataset Loaded : {len(raw_df)} URLs")

    augmented_df = add_www_variants(raw_df)
    print(f"Dataset After WWW Augmentation : {len(augmented_df)} URLs")

    feature_rows = []
    skipped_count = 0

    for index, row in augmented_df.iterrows():
        if (index + 1) % 10000 == 0:
            print(f"Processed {index + 1} URLs...")

        try:
            features = extract_features(row["URL"])
            features["label"] = row["label"]
            feature_rows.append(features)
        except Exception as exc_error:
            skipped_count += 1
            print(f"Skipped Row {index}: {exc_error}")

    features_df = pd.DataFrame(feature_rows)
    FEATURES_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_DATASET_PATH, index=False)

    print("\nFeature Extraction Completed!")
    print(f"Total URLs Processed : {len(augmented_df)}")
    print(f"Successfully Processed : {len(features_df)}")
    print(f"Skipped URLs : {skipped_count}")
    print(f"features.csv saved successfully to {FEATURES_DATASET_PATH}!\n")
    print(features_df.head())


def main():
    """Run full dataset preprocessing pipeline."""
    create_raw_dataset()
    create_feature_dataset()
    print("=" * 60)
    print("Dataset Preprocessing Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
