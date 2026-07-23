"""Model utilities module for AI Phishing Detection.

Provides reusable dataset loading, validation, splitting, and visualization utilities
shared across model training and hyperparameter tuning pipelines.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    FEATURES_DATASET_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    GRAPH_DIR
)


def load_feature_dataset():
    """Load the feature extraction dataset from CSV.

    Returns
    -------
    pd.DataFrame
        The loaded features DataFrame.
    """
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    try:
        features_df = pd.read_csv(FEATURES_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {FEATURES_DATASET_PATH}")
        sys.exit(1)

    print("Dataset Loaded Successfully")
    print(f"Rows    : {features_df.shape[0]}")
    print(f"Columns : {features_df.shape[1]}\n")
    return features_df


def validate_feature_dataset(features_df):
    """Validate feature dataset for missing values and class balance.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataset to validate.

    Raises
    ------
    ValueError
        If 'label' column is missing.
    """
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)

    if "label" not in features_df.columns:
        raise ValueError("Dataset must contain 'label' column.")

    print("Missing Values Check:")
    print(features_df.isnull().sum())
    print("\nDataset Info:")
    print(features_df.info())
    print("\nClass Distribution:")
    print(features_df["label"].value_counts())
    print()


def prepare_dataset_split(features_df):
    """Separate target variable and extract feature names.

    Parameters
    ----------
    features_df : pd.DataFrame
        Complete features dataset with 'label' column.

    Returns
    -------
    tuple
        (features_matrix, target_series, feature_names_list)
    """
    features_matrix = features_df.drop(columns=["label"])
    target_series = features_df["label"]
    feature_names = features_matrix.columns.tolist()
    return features_matrix, target_series, feature_names


def split_train_test(features_matrix, target_series):
    """Create stratified train-test split for dataset.

    Parameters
    ----------
    features_matrix : pd.DataFrame
        Feature matrix (X).
    target_series : pd.Series
        Target labels (y).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_matrix,
        target_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_series
    )

    print("=" * 60)
    print("Train Test Split")
    print("=" * 60)
    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}\n")
    return X_train, X_test, y_train, y_test


def plot_and_save_confusion_matrix(model_instance, X_test, y_test, output_path, title="Confusion Matrix"):
    """Generate and save confusion matrix visualization.

    Parameters
    ----------
    model_instance : object
        Fitted model or predictions array.
    X_test : pd.DataFrame or None
        Test features (used if model_instance is estimator).
    y_test : pd.Series
        True test labels.
    output_path : Path or str
        Destination path for saved graph.
    title : str
        Title of the chart.
    """
    if hasattr(model_instance, "predict"):
        predictions = model_instance.predict(X_test)
    else:
        predictions = model_instance

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_matrix = confusion_matrix(y_test, predictions)
    display_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm_matrix,
        display_labels=["Legitimate", "Phishing"]
    )

    plt.figure(figsize=(6, 6))
    display_cm.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")


def plot_and_save_feature_importance(model_instance, feature_names, output_path, title="Feature Importance"):
    """Generate and save feature importance horizontal bar graph.

    Parameters
    ----------
    model_instance : object
        Fitted model with feature_importances_ attribute.
    feature_names : list
        List of feature names matching model features.
    output_path : Path or str
        Destination path for saved graph.
    title : str
        Title of the graph.
    """
    if not hasattr(model_instance, "feature_importances_"):
        print("Feature Importance Not Available for this model.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importance_series = pd.Series(
        model_instance.feature_importances_,
        index=feature_names
    ).sort_values()

    plt.figure(figsize=(10, 8))
    importance_series.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature Importance graph saved to {output_path}")
