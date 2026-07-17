# from feature_extractor import extract_features
# from train_model import *

# url = input("Enter URL: ")

# features = extract_features(url)

# print("\nExtracted Features:\n")

# for key, value in features.items():
#     print(f"{key:20}: {value}")

# print("AI Phishing Detection Project")
# import pandas as pd

# df = pd.read_csv("../data/features.csv")

# print(df["label"].value_counts())

# print(df.head(10))

# import pandas as pd

# df = pd.read_csv("../data/phishing.csv")

# print(df[df["URL"].str.contains("google", case=False, na=False)].head())

# print(df[df["URL"].str.contains("amazon", case=False, na=False)].head())

# print(df[df["URL"].str.contains("github", case=False, na=False)].head())

# import pandas as pd

# df = pd.read_csv("../data/raw_urls.csv")

# print(df["URL"].str.startswith("https://www.").sum())
# print(df["URL"].str.startswith("https://").sum())
# print(df["URL"].str.startswith("http://").sum())

# print(df.groupby("label")["URL"].apply(
#     lambda x: x.str.startswith("https://www.").sum()
# ))

# print(df.groupby("label")["URL"].apply(
#     lambda x: x.str.startswith("https://").sum()
# ))

from feature_extractor import extract_features
from predict import predict_from_url

urls = [
    "https://google.com",
    "https://www.google.com",
    "https://github.com",
    "https://www.github.com",
    "https://amazon.com",
    "https://www.amazon.com",
]

for url in urls:
    result = predict_from_url(url)
    print(url)
    print(result)
    print("-" * 50)