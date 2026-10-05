# src/train_with_mlflow.py

"""Train and track multiple model configurations with MLflow."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
TARGET_COL = "species"
DEFAULT_EXPERIMENT = "iris-classification-baseline"


def resolve_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI")
    if configured:
        return configured

    default_local = "http://127.0.0.1:5000"
    try:
        with urlopen(default_local, timeout=2):
            return default_local
    except Exception:
        return (Path.cwd() / "mlruns").as_uri()


def configure_mlflow(experiment_name: str = DEFAULT_EXPERIMENT) -> None:
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def train_and_log_models(data_path: str = "data/processed/iris_features.csv") -> list[dict[str, float | str]]:
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())

    X = df[FEATURE_COLS].copy().fillna(df[FEATURE_COLS].median())
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=200, C=1.0),
        "random_forest_shallow": RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
        ),
        "random_forest_deep": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        ),
    }

    results: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print("Training:", model_name)
        print("=" * 60)

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            mlflow.log_param("model_type", model_name)

            if model_name == "logistic_regression":
                mlflow.log_param("max_iter", 200)
                mlflow.log_param("C", 1.0)
            else:
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("f1_macro", f1)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
            disp.plot()
            plt.title(f"Confusion Matrix - {model_name}")
            plt.tight_layout()

            cm_filename = Path(f"confusion_matrix_{model_name}.png")
            plt.savefig(cm_filename)
            plt.close()
            mlflow.log_artifact(str(cm_filename))

            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append(
                {
                    "model_name": model_name,
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "run_id": mlflow.active_run().info.run_id,
                }
            )

    return results


def main() -> None:
    configure_mlflow()
    train_and_log_models()


if __name__ == "__main__":
    main()

