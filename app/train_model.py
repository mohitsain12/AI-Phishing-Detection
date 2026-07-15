import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC

import os
from joblib import dump

os.makedirs("../models", exist_ok=True)
os.makedirs("../graphs", exist_ok=True)

# ============================
# Load Dataset
# ============================

try:
    df = pd.read_csv("../data/features.csv")
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: features.csv not found.")
    exit()

# ============================
# Check Dataset
# ============================

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

if df.isnull().sum().sum() > 0:
    print("Warning: Missing values detected.")
    print(df.isnull().sum())
    print()

# ============================
# Features and Labels
# ============================

X = df.drop(columns=["label"])
y = df["label"]

feature_names = X.columns

# ============================
# Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print()

# ============================
# Define Models
# ============================

from sklearn.svm import LinearSVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}


results = {}
trained_models = {}

# ============================
# Train and Evaluate Models
# ============================

print("=" * 45)
print("Training Models")
print("=" * 45)

# for name, model in models.items():

#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)

#     accuracy = accuracy_score(y_test, predictions)

#     results[name] = accuracy

#     print(f"{name:<25} {accuracy:.4f}")
for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    trained_models[name] = model

    print(f"{name:<25} {accuracy:.4f}")
    print(f"{name} completed!")  
    print("\n" + "=" * 45)
    
# ============================
# Best Model
# ============================

best_model_name = max(results, key=results.get)

best_model = trained_models[best_model_name]

# Save Best Model
dump(best_model, "../models/phishing_detector_model.pkl")
dump(feature_names.tolist(), "../models/phishing_feature_names.pkl")

# Save Feature Names
dump(feature_names.tolist(), "../models/feature_names.pkl")

print("\nBest model saved successfully!")

print("\n" + "=" * 45)
print("Best Model")
print("=" * 45)
print(best_model_name)
print(f"Accuracy : {results[best_model_name]*100:.2f}%")

# ============================
# Plot Accuracy
# ============================

plt.figure(figsize=(9, 5))

bars = plt.bar(results.keys(), results.values())

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.xticks(rotation=20)

# Add Accuracy Labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height*100:.2f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.tight_layout()

plt.savefig(
    "../screenshots/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved successfully!")

plt.show()