"""Train model module for AI Phishing Detection.

Provides utilities and application logic for the project.
"""

# ==========================================================
# AI PHISHING DETECTION
# Module 3 : Model Training
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

    """Load dataset.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
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

    """Validate dataset.
    
    Parameters
    ----------
    df : TYPE
        Description of df.
    
    Raises
    ------
    Exception
        If an error occurs during execution.
    """
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column.")

    print("No Missing Values")

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

    """Split dataset.
    
    Parameters
    ----------
    df : TYPE
        Description of df.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    X = df.drop(columns=["label"])

    y = df["label"]

    feature_names = X.columns.tolist()

    return X, y, feature_names

# ==========================================================
# Train Test Split
# ==========================================================

def create_train_test_split(X, y):

    """Create train test split.
    
    Parameters
    ----------
    X : TYPE
        Description of X.
    y : TYPE
        Description of y.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
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
# Machine Learning Models
# ==========================================================

def create_models():

    """Create models.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )

    }

    return models

# ==========================================================
# Train & Evaluate Models
# ==========================================================

def train_models(models, X_train, X_test, y_train, y_test):

    """Train models.
    
    Parameters
    ----------
    models : TYPE
        Description of models.
    X_train : TYPE
        Description of X_train.
    X_test : TYPE
        Description of X_test.
    y_train : TYPE
        Description of y_train.
    y_test : TYPE
        Description of y_test.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    print("=" * 60)
    print("Training Machine Learning Models")
    print("=" * 60)

    results = {}
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # --------------------------
        # Train Model
        # --------------------------

        model.fit(X_train, y_train)

        # --------------------------
        # Predictions
        # --------------------------

        predictions = model.predict(X_test)

        # Probability Prediction
        # (Not all models support it)
        # --------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )

        else:

            roc_auc = None

        # --------------------------
        # Evaluation Metrics
        # --------------------------

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

        # --------------------------
        # Store Results
        # --------------------------

        results[name] = {

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC AUC": roc_auc

        }

        trained_models[name] = model

        # --------------------------
        # Display Results
        # --------------------------

        print("-" * 40)

        print(f"Accuracy  : {accuracy:.4f}")

        print(f"Precision : {precision:.4f}")

        print(f"Recall    : {recall:.4f}")

        print(f"F1 Score  : {f1:.4f}")

        if roc_auc is not None:
            print(f"ROC AUC   : {roc_auc:.4f}")

        else:
            print("ROC AUC   : Not Available")

        print("-" * 40)

        print("\nClassification Report\n")

        print(

            classification_report(

                y_test,

                predictions

            )

        )

    return results, trained_models

# ==========================================================
# Best Model
# ==========================================================

def get_best_model(results, trained_models):

    """Get best model.
    
    Parameters
    ----------
    results : TYPE
        Description of results.
    trained_models : TYPE
        Description of trained_models.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    best_model_name = max(

        results,

        key=lambda x: results[x]["Accuracy"]

    )

    best_model = trained_models[best_model_name]

    print("=" * 60)

    print("Best Model")

    print("=" * 60)

    print(best_model_name)

    print()

    for metric, value in results[best_model_name].items():

        if value is not None:

            print(f"{metric:<12}: {value:.4f}")

    print()

    return best_model_name, best_model

# ==========================================================
# Save Models
# ==========================================================

def save_models(trained_models, best_model, feature_names):

    """Save models.
    
    Parameters
    ----------
    trained_models : TYPE
        Description of trained_models.
    best_model : TYPE
        Description of best_model.
    feature_names : TYPE
        Description of feature_names.
    """
    print("=" * 60)
    print("Saving Models")
    print("=" * 60)

    model_files = {
        "Logistic Regression": "../models/logistic_model.pkl",
        "Decision Tree": "../models/decision_tree_model.pkl",
        "Random Forest": "../models/random_forest_model.pkl",
        "KNN": "../models/knn_model.pkl"
    }

    for name, model in trained_models.items():

        joblib.dump(model, model_files[name])

        print(f"{name} saved.")

    joblib.dump(
        best_model,
        "../models/best_model.pkl"
    )

    joblib.dump(
        feature_names,
        "../models/phishing_feature_names.pkl"
    )

    print("\nBest Model Saved Successfully!")
    print("Feature Names Saved Successfully!\n")
    
# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(best_model, X_test, y_test):

    """Plot confusion matrix.
    
    Parameters
    ----------
    best_model : TYPE
        Description of best_model.
    X_test : TYPE
        Description of X_test.
    y_test : TYPE
        Description of y_test.
    """
    predictions = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"]
    )

    disp.plot()

    plt.title("Confusion Matrix")

    plt.savefig(
        "../graphs/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion Matrix Saved.")

# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(best_model, feature_names):

    """Plot feature importance.
    
    Parameters
    ----------
    best_model : TYPE
        Description of best_model.
    feature_names : TYPE
        Description of feature_names.
    
    Returns
    -------
    TYPE
        Description of return value.
    """
    if not hasattr(best_model, "feature_importances_"):

        print("Feature Importance Not Available.")
        return

    importance = pd.Series(
        best_model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values()

    plt.figure(figsize=(10,8))

    importance.plot(kind="barh")

    plt.title("Feature Importance")

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig(
        "../graphs/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Feature Importance Graph Saved.")
    
# ==========================================================
# Model Comparison
# ==========================================================

def plot_model_comparison(results):

    """Plot model comparison.
    
    Parameters
    ----------
    results : TYPE
        Description of results.
    """
    accuracy = {

        name: values["Accuracy"]

        for name, values in results.items()

    }

    plt.figure(figsize=(9,5))

    bars = plt.bar(
        accuracy.keys(),
        accuracy.values()
    )

    plt.title("Machine Learning Model Comparison")

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.ylim(0,1)

    plt.xticks(rotation=15)

    for bar in bars:

        height = bar.get_height()

        plt.text(

            bar.get_x()+bar.get_width()/2,

            height+0.01,

            f"{height*100:.2f}%",

            ha="center"

        )

    plt.tight_layout()

    plt.savefig(

        "../graphs/model_comparison.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print("Model Comparison Graph Saved.")
    
# ==========================================================
# Main
# ==========================================================

def main():

    """Main.
    """
    df = load_dataset()

    validate_dataset(df)

    X, y, feature_names = split_dataset(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    models = create_models()

    results, trained_models = train_models(

        models,

        X_train,

        X_test,

        y_train,

        y_test

    )

    best_model_name, best_model = get_best_model(

        results,

        trained_models

    )

    save_models(

        trained_models,

        best_model,

        feature_names

    )

    plot_confusion_matrix(

        best_model,

        X_test,

        y_test

    )

    plot_feature_importance(

        best_model,

        feature_names

    )

    plot_model_comparison(results)

    print("="*60)

    print("Training Completed Successfully!")

    print("="*60)

    print(f"Best Model : {best_model_name}")

    print("All Graphs Saved Inside graphs/")

    print("All Models Saved Inside models/")


if __name__ == "__main__":
    main()