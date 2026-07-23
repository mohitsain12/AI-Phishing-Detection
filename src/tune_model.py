"""Tune model module for AI Phishing Detection.

Performs hyperparameter tuning for Random Forest Classifier using GridSearchCV,
evaluates tuned model metrics, and saves tuned artifacts.
"""

import sys
import time
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    TUNED_MODEL_PATH,
    FEATURE_NAMES_PATH,
    TUNED_CONFUSION_MATRIX_GRAPH,
    TUNED_FEATURE_IMPORTANCE_GRAPH,
    RANDOM_STATE,
    CV_FOLDS
)
from src.model_utils import (
    load_feature_dataset,
    validate_feature_dataset,
    prepare_dataset_split,
    split_train_test,
    plot_and_save_confusion_matrix,
    plot_and_save_feature_importance
)


def create_parameter_grid():
    """Create parameter grid dictionary for Random Forest tuning.

    Returns
    -------
    dict
        Hyperparameter grid dictionary.
    """
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }


def create_grid_search(parameter_grid):
    """Instantiate GridSearchCV object for Random Forest.

    Parameters
    ----------
    parameter_grid : dict
        Hyperparameter search space.

    Returns
    -------
    GridSearchCV
        Grid search CV estimator instance.
    """
    return GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        param_grid=parameter_grid,
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )


def execute_tuning(grid_search_estimator, X_train, y_train):
    """Fit grid search and log training duration and optimal parameters.

    Parameters
    ----------
    grid_search_estimator : GridSearchCV
        Configured grid search object.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    object
        Tuned best estimator model.
    """
    print("=" * 60)
    print("Hyperparameter Tuning Started")
    print("=" * 60)

    start_time = time.time()
    grid_search_estimator.fit(X_train, y_train)
    duration = time.time() - start_time

    print(f"\nHyperparameter Tuning Completed in {duration:.2f} seconds!\n")
    print("=" * 60)
    print("Best Hyperparameters")
    print("=" * 60)

    for param_name, param_value in grid_search_estimator.best_params_.items():
        print(f"{param_name:<20}: {param_value}")

    print(f"\nBest Cross Validation F1 Score: {grid_search_estimator.best_score_:.4f}\n")
    return grid_search_estimator.best_estimator_


def evaluate_tuned_model(best_estimator, X_test, y_test):
    """Evaluate tuned model performance metrics on test dataset.

    Parameters
    ----------
    best_estimator : object
        Fitted best model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    tuple
        (predictions_array, metrics_dict)
    """
    print("=" * 60)
    print("Evaluating Tuned Random Forest")
    print("=" * 60)

    predictions = best_estimator.predict(X_test)
    probabilities = best_estimator.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}\n")

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, predictions))

    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    }
    return predictions, metrics_dict


def save_tuned_model_artifacts(best_estimator, feature_names):
    """Save tuned model and feature names pickle files to disk.

    Parameters
    ----------
    best_estimator : object
        Tuned best model instance.
    feature_names : list
        Feature column names.
    """
    print("=" * 60)
    print("Saving Tuned Model")
    print("=" * 60)

    TUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_estimator, TUNED_MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print(f"Tuned Model Saved to {TUNED_MODEL_PATH}")
    print(f"Feature Names Saved to {FEATURE_NAMES_PATH}\n")


def main():
    """Run hyperparameter tuning pipeline."""
    feature_df = load_feature_dataset()
    validate_feature_dataset(feature_df)

    X, y, feature_names = prepare_dataset_split(feature_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    param_grid = create_parameter_grid()
    grid_search_estimator = create_grid_search(param_grid)
    best_estimator = execute_tuning(grid_search_estimator, X_train, y_train)

    predictions, metrics = evaluate_tuned_model(best_estimator, X_test, y_test)
    save_tuned_model_artifacts(best_estimator, feature_names)

    plot_and_save_confusion_matrix(
        predictions, X_test, y_test, TUNED_CONFUSION_MATRIX_GRAPH, "Random Forest (Tuned) - Confusion Matrix"
    )
    plot_and_save_feature_importance(
        best_estimator, feature_names, TUNED_FEATURE_IMPORTANCE_GRAPH, "Random Forest (Tuned) - Feature Importance"
    )

    print("=" * 60)
    print("Hyperparameter Tuning Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()