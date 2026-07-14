from feature_extractor import extract_features

url = input("Enter URL: ")

features = extract_features(url)

print("\nExtracted Features:\n")

for key, value in features.items():
    print(f"{key:20}: {value}")

