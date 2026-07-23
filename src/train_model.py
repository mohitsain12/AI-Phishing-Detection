"""Train model module for AI Phishing Detection.

Trains Logistic Regression, Decision Tree, Random Forest, and KNN classifiers,
evaluates performance metrics, and saves models and graphs.
"""

import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    classification_report
)

# Ensure root directory is accessible in sys.path
root_directory = Path(__file__).resolve().parent.parent
if str(root_directory) not in sys.path:
    sys.path.insert(0, str(root_directory))

from config import (
    LOGISTIC_MODEL_PATH,
    DECISION_TREE_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    KNN_MODEL_PATH,
    BEST_MODEL_PATH,
    FEATURE_NAMES_PATH,
    CONFUSION_MATRIX_GRAPH,
    FEATURE_IMPORTANCE_GRAPH,
    MODEL_COMPARISON_GRAPH,
    RANDOM_STATE
)
from src.model_utils import (
    load_feature_dataset,
    validate_feature_dataset,
    prepare_dataset_split,
    split_train_test,
    plot_and_save_confusion_matrix,
    plot_and_save_feature_importance
)


def create_models():
    """Instantiate standard Machine Learning classifiers.

    Returns
    -------
    dict
        Dictionary mapping model names to model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        )
    }


def train_models(models_dict, X_train, X_test, y_train, y_test):
    """Train each classifier and calculate evaluation metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary of model estimators.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Testing labels.

    Returns
    -------
    tuple
        (results_dict, trained_models_dict)
    """
    print("=" * 60)
    print("Training Machine Learning Models")
    print("=" * 60)

    results = {}
    trained_models = {}

    for model_name, model_instance in models_dict.items():
        print(f"\nTraining {model_name}...")
        model_instance.fit(X_train, y_train)
        predictions = model_instance.predict(X_test)

        probabilities = (
            model_instance.predict_proba(X_test)[:, 1]
            if hasattr(model_instance, "predict_proba")
            else None
        )
        roc_auc = (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None
            else None
        )

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results[model_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        }
        trained_models[model_name] = model_instance

        print("-" * 40)
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}" if roc_auc is not None else "ROC AUC   : N/A")
        print("-" * 40)
        print("\nClassification Report\n")
        print(classification_report(y_test, predictions))

    return results, trained_models


def get_best_model(results_dict, trained_models_dict):
    """Determine best performing model based on test accuracy.

    Parameters
    ----------
    results_dict : dict
        Metrics results dictionary.
    trained_models_dict : dict
        Trained model instances dictionary.

    Returns
    -------
    tuple
        (best_model_name, best_model_instance)
    """
    best_model_name = max(
        results_dict,
        key=lambda key_name: results_dict[key_name]["Accuracy"]
    )
    best_model = trained_models_dict[best_model_name]

    print("=" * 60)
    print("Best Model Selected")
    print("=" * 60)
    print(f"Name: {best_model_name}\n")

    for metric_name, metric_value in results_dict[best_model_name].items():
        if metric_value is not None:
            print(f"{metric_name:<12}: {metric_value:.4f}")
    print()

    return best_model_name, best_model


def save_trained_models(trained_models_dict, best_model_instance, feature_names):
    """Serialize trained models and feature names to disk.

    Parameters
    ----------
    trained_models_dict : dict
        Dictionary of all trained models.
    best_model_instance : object
        Best performing model instance.
    feature_names : list
        List of feature names.
    """
    print("=" * 60)
    print("Saving Models")
    print("=" * 60)

    file_mapping = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "Decision Tree": DECISION_TREE_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH,
        "KNN": KNN_MODEL_PATH
    }

    for name, model_instance in trained_models_dict.items():
        target_path = file_mapping[name]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_instance, target_path)
        print(f"{name} saved to {target_path}")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_instance, BEST_MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    print("\nBest Model Saved Successfully!")
    print(f"Feature Names Saved to {FEATURE_NAMES_PATH}\n")


def plot_model_comparison(results_dict):
    """Plot and save accuracy comparison bar chart across models.

    Parameters
    ----------
    results_dict : dict
        Model metrics dictionary.
    """
    accuracy_data = {
        name: metrics["Accuracy"] for name, metrics in results_dict.items()
    }

    plt.figure(figsize=(9, 5))
    bars = plt.bar(accuracy_data.keys(), accuracy_data.values())
    plt.title("Machine Learning Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height * 100:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    MODEL_COMPARISON_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(MODEL_COMPARISON_GRAPH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model Comparison Graph saved to {MODEL_COMPARISON_GRAPH}")


def main():
    """Run model training and evaluation workflow."""
    feature_df = load_feature_dataset()
    validate_feature_dataset(feature_df)

    X, y, feature_names = prepare_dataset_split(feature_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    models_dict = create_models()
    results_dict, trained_models = train_models(
        models_dict, X_train, X_test, y_train, y_test
    )

    best_name, best_model = get_best_model(results_dict, trained_models)
    save_trained_models(trained_models, best_model, feature_names)

    plot_and_save_confusion_matrix(
        best_model, X_test, y_test, CONFUSION_MATRIX_GRAPH, "Confusion Matrix"
    )
    plot_and_save_feature_importance(
        best_model, feature_names, FEATURE_IMPORTANCE_GRAPH, "Feature Importance"
    )
    plot_model_comparison(results_dict)

    print("=" * 60)
    print("Training Completed Successfully!")
    print("=" * 60)
    print(f"Best Model : {best_name}")


if __name__ == "__main__":
    main()