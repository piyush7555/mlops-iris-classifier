# src/load_registered_model.py

import os
from pathlib import Path
from urllib.request import urlopen

import mlflow
import mlflow.sklearn

MODEL_URI = "models:/iris-classifier-prod/Staging"


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


def main() -> None:
    mlflow.set_tracking_uri(resolve_tracking_uri())

    print("=" * 60)
    print("LOADING REGISTERED MODEL")
    print("=" * 60)
    print("Model URI:", MODEL_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)

    print("\nModel loaded successfully!")
    print("Model type:", type(model))
    print("\nLoaded Model:")
    print(model)


if __name__ == "__main__":
    main()
