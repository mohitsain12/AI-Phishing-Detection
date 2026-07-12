# from feature_extractor import extract_features

# url = input("Enter URL: ")

# features = extract_features(url)

# print("\nExtracted Features:\n")

# for key, value in features.items():
#     print(f"{key:20}: {value}")

import pandas as pd

df = pd.read_csv("../data/phishing.csv")

new_df = df[["URL", "label"]]

new_df.to_csv("../data/raw_urls.csv", index=False)

print(new_df.head())