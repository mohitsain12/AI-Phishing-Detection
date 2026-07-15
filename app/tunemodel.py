import os
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

X = df.drop("label", axis=1)

y = df["label"]

feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


param_grid = {
    "n_estimators": [100],
    "max_depth": [10, None],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}

grid_search = GridSearchCV(

    estimator=RandomForestClassifier(random_state=42),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")

print(grid_search.best_score_)

best_rf = grid_search.best_estimator_

predictions = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nTest Accuracy:")

print(round(accuracy * 100, 2), "%")

print("\nClassification Report")

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(
    cmap="Blues",
    values_format="d"   # Show integers instead of scientific notation
)

plt.title("Random Forest (Tuned)")

disp.plot(cmap="Blues")

plt.title("Random Forest (Tuned)")

plt.savefig(
    "../graphs/tuned_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

dump(

    feature_names.tolist(),

    "../models/phishing_feature_names.pkl"

)
dump(
    best_rf,
    "../models/phishing_detector_tuned.pkl"
)

print("\n" + "=" * 50)

print("Hyperparameter Tuning Completed Successfully")

print("=" * 50)

print("Best Parameters:")

print(grid_search.best_params_)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")

print("\nFiles Saved")

print("Model    : ../models/phishing_detector_tuned.pkl")

print("Features : ../models/phishing_feature_names.pkl")

print("Graph    : ../graphs/tuned_confusion_matrix.png")