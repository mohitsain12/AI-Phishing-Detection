from feature_extractor import extract_features
from train_model import *

url = input("Enter URL: ")

features = extract_features(url)

print("\nExtracted Features:\n")

for key, value in features.items():
    print(f"{key:20}: {value}")

print("AI Phishing Detection Project")