"""Preprocess module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

import pandas as pd
import re
from feature_extractor import extract_features

# -----------------------------------------------------
# Label Mapping
# Original Dataset:
#   1 = Legitimate
#   0 = Phishing
#
# Project Standard:
#   0 = Legitimate
#   1 = Phishing
# -----------------------------------------------------

LABEL_MAPPING = {
    1: 0,
    0: 1
}


# -----------------------------------------------------
# Create raw_urls.csv
# -----------------------------------------------------

def create_raw_dataset():

    """Create raw dataset.
    
    Raises
    ------
    Exception
        If an error occurs during execution.
    """
    print("=" * 60)
    print("STEP 1 : Creating raw_urls.csv")
    print("=" * 60)

    # Load original dataset
    df = pd.read_csv("../data/phishing.csv")

    print(f"Original Dataset : {len(df)} rows")

    # Check required columns
    required_columns = ["URL", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Keep only URL and label
    raw_df = df[["URL", "label"]].copy()

    # Remove missing values
    raw_df.dropna(subset=["URL", "label"], inplace=True)

    # Remove duplicate URLs
    raw_df.drop_duplicates(subset=["URL"], inplace=True)

    # Convert labels to project standard
    raw_df["label"] = raw_df["label"].map(LABEL_MAPPING)

    # Reset index
    raw_df.reset_index(drop=True, inplace=True)

    # Save dataset
    raw_df.to_csv("../data/raw_urls.csv", index=False)

    print(f"Rows after cleaning : {len(raw_df)}")
    print("raw_urls.csv saved successfully!\n")


# -----------------------------------------------------
# Create features.csv
# -----------------------------------------------------

def add_www_variants(df):

    """Add www variants.
    
    Parameters
    ----------
    df : TYPE
        Description of df.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    variant_rows = []

    for _, row in df.iterrows():

        url = str(row["URL"])

        variant_url = re.sub(
            r"://www\.",
            "://",
            url,
            count=1,
            flags=re.IGNORECASE
        )

        if variant_url != url:

            variant_rows.append(
                {
                    "URL": variant_url,
                    "label": row["label"]
                }
            )

    if not variant_rows:

        return df

    augmented_df = pd.concat(
        [
            df,
            pd.DataFrame(variant_rows)
        ],
        ignore_index=True
    )

    augmented_df.drop_duplicates(
        subset=["URL"],
        inplace=True
    )

    augmented_df.reset_index(
        drop=True,
        inplace=True
    )

    return augmented_df

def create_feature_dataset():

    """Create feature dataset.
    """
    print("=" * 60)
    print("STEP 2 : Creating features.csv")
    print("=" * 60)

    # Load raw dataset
    df = pd.read_csv("../data/raw_urls.csv")

    print(f"Raw Dataset Loaded : {len(df)} URLs")

    df = add_www_variants(df)

    print(f"Dataset After WWW Augmentation : {len(df)} URLs")

    feature_rows = []
    skipped = 0

    # Extract features
    for index, row in df.iterrows():

        if (index + 1) % 10000 == 0:
            print(f"Processed {index + 1} URLs...")
        
        try:

            features = extract_features(row["URL"])

            features["label"] = row["label"]

            feature_rows.append(features)

        except Exception as e:

            skipped += 1

            print(f"Skipped Row {index}: {e}")

    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Save dataset
    features_df.to_csv("../data/features.csv", index=False)

    print("\nFeature Extraction Completed!")
    print(f"Total URLs Processed : {len(df)}")
    print(f"Successfully Processed : {len(features_df)}")
    print(f"Skipped URLs : {skipped}")
    print("features.csv saved successfully!\n")

    print(features_df.head())


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    """Main.
    """
    create_raw_dataset()

    create_feature_dataset()

    print("=" * 60)
    print("Dataset Preprocessing Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
