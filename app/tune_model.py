# ==========================================================
# AI PHISHING DETECTION
# Module 4 : Hyperparameter Tuning
# ==========================================================

import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# Create Required Folders
# ==========================================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../graphs", exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    try:

        df = pd.read_csv("../data/features.csv")

    except FileNotFoundError:

        print("Error : features.csv not found.")

        exit()

    print(f"Dataset Loaded Successfully")

    print(f"Rows    : {df.shape[0]}")

    print(f"Columns : {df.shape[1]}")

    print()

    return df

# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(df):

    print("=" * 60)

    print("Dataset Validation")

    print("=" * 60)

    if "label" not in df.columns:

        raise ValueError(
            "Dataset must contain 'label' column."
        )

    print("Missing Values")

    print(df.isnull().sum())

    print()

    print("Dataset Information")

    print(df.info())

    print()

    print("Class Distribution")

    print(df["label"].value_counts())

    print()
    
# ==========================================================
# Split Dataset
# ==========================================================

def split_dataset(df):

    X = df.drop(
        columns=["label"]
    )

    y = df["label"]

    feature_names = X.columns.tolist()

    return X, y, feature_names

# ==========================================================
# Train Test Split
# ==========================================================

def create_train_test_split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    print("=" * 60)

    print("Train Test Split")

    print("=" * 60)

    print(f"Training Samples : {len(X_train)}")

    print(f"Testing Samples  : {len(X_test)}")

    print()

    return X_train, X_test, y_train, y_test

# ==========================================================
# Hyperparameter Grid
# ==========================================================

def create_parameter_grid():

    param_grid = {

        "n_estimators": [100, 200, 300],

        "max_depth": [10, 20, None],

        "min_samples_split": [2, 5, 10],

        "min_samples_leaf": [1, 2, 4]

    }

    return param_grid

# ==========================================================
# Grid Search
# ==========================================================

def create_grid_search(param_grid):

    grid_search = GridSearchCV(

        estimator=RandomForestClassifier(

            random_state=42,

            n_jobs=-1

        ),

        param_grid=param_grid,

        cv=5,

        scoring="f1",

        n_jobs=-1,

        verbose=2

    )

    return grid_search

# ==========================================================
# Hyperparameter Tuning
# ==========================================================

def tune_model(grid_search, X_train, y_train):

    print("=" * 60)
    print("Hyperparameter Tuning Started")
    print("=" * 60)

    start_time = time.time()

    grid_search.fit(
        X_train,
        y_train
    )

    end_time = time.time()

    training_time = end_time - start_time

    print("\nHyperparameter Tuning Completed!")

    print(f"Training Time : {training_time:.2f} seconds")

    print()

    print("=" * 60)
    print("Best Hyperparameters")
    print("=" * 60)

    for parameter, value in grid_search.best_params_.items():

        print(f"{parameter:<20}: {value}")

    print()

    print(f"Best Cross Validation F1 Score : {grid_search.best_score_:.4f}")

    print()

    return grid_search.best_estimator_

# ==========================================================
# Evaluate Tuned Model
# ==========================================================

def evaluate_model(best_model, X_test, y_test):

    print("=" * 60)
    print("Evaluating Tuned Random Forest")
    print("=" * 60)

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC AUC   : {roc_auc:.4f}")

    print()

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    metrics = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc

    }

    return predictions, metrics

# ==========================================================
# Compare Results
# ==========================================================

def compare_results(metrics):

    original_accuracy = 0.9953

    tuned_accuracy = metrics["Accuracy"]

    improvement = tuned_accuracy - original_accuracy

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(f"Original Accuracy : {original_accuracy*100:.2f}%")

    print(f"Tuned Accuracy    : {tuned_accuracy*100:.2f}%")

    print(f"Improvement       : {improvement*100:.3f}%")

    print()
    
# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(predictions, y_test):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"]
    )

    plt.figure(figsize=(6,6))

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Random Forest (Tuned) - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        "../graphs/tuned_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion Matrix Saved Successfully!")
    
# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(best_model, feature_names):

    importance = pd.Series(
        best_model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values()

    plt.figure(figsize=(10,8))

    importance.plot(kind="barh")

    plt.title("Random Forest (Tuned) - Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "../graphs/tuned_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Feature Importance Graph Saved Successfully!")
    
# ==========================================================
# Save Tuned Model
# ==========================================================

def save_model(best_model, feature_names):

    print("=" * 60)
    print("Saving Tuned Model")
    print("=" * 60)

    joblib.dump(

        best_model,

        "../models/phishing_detector_tuned.pkl"

    )

    joblib.dump(

        feature_names,

        "../models/phishing_feature_names.pkl"

    )

    print("Tuned Model Saved Successfully!")

    print("Feature Names Saved Successfully!")

    print()
    
# ==========================================================
# Main
# ==========================================================

def main():

    df = load_dataset()

    validate_dataset(df)

    X, y, feature_names = split_dataset(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    param_grid = create_parameter_grid()

    grid_search = create_grid_search(param_grid)

    best_model = tune_model(
        grid_search,
        X_train,
        y_train
    )

    predictions, metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    compare_results(metrics)

    plot_confusion_matrix(
        predictions,
        y_test
    )

    plot_feature_importance(
        best_model,
        feature_names
    )

    save_model(
        best_model,
        feature_names
    )

    print("=" * 60)

    print("Hyperparameter Tuning Completed Successfully!")

    print("=" * 60)

    print("Files Generated")

    print()

    print("Model")
    print("  ../models/phishing_detector_tuned.pkl")

    print()

    print("Feature Names")
    print("  ../models/phishing_feature_names.pkl")

    print()

    print("Graphs")
    print("  ../graphs/tuned_confusion_matrix.png")
    print("  ../graphs/tuned_feature_importance.png")

    print("=" * 60)


if __name__ == "__main__":

    main()