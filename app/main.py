import pandas as pd

df = pd.read_csv("../data/dataset.csv")

# print(df.head(10))
# print(df.shape)

# print(df.info())

# print(df.describe()) 
# print(df["Result"].value_counts())  # show the no. of unique values in the Result column

# print(df.isnull().sum())
# print(df.duplicated().sum())

# url = "https://google.com"

# print(len(url))

# print(url.startswith("https"))

# print(url.count("."))

# print("@" in url)

# print("-" in url)


url = input("Enter URL: ")

print("Length:", len(url))
print("HTTPS:", url.startswith("https"))
print("Dots:", url.count("."))
print("Contains @:", "@" in url)
print("Contains -:", "-" in url)

keywords = ["login", "verify", "secure", "bank", "update"]

for word in keywords:
    if word in url.lower():
        print("Found keyword:", word)