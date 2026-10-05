# src/register_best_model.py

"""Register the best-performing MLflow run as a production model candidate."""

import os
from pathlib import Path
from urllib.request import urlopen

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT = "iris-classification-baseline"
MODEL_NAME = "iris-classifier-prod"


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


def get_best_run(client: MlflowClient, experiment_name: str = DEFAULT_EXPERIMENT):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_macro DESC"])
    if not runs:
        raise RuntimeError(f"No runs found in the experiment '{experiment_name}'.")
    return runs[0]


def main() -> None:
    mlflow.set_tracking_uri(resolve_tracking_uri())
    client = MlflowClient()

    best_run = get_best_run(client)
    best_run_id = best_run.info.run_id
    best_f1 = best_run.data.metrics["f1_macro"]
    best_model_type = best_run.data.params["model_type"]

    print("\nBest Run")
    print("Run ID:", best_run_id)
    print("Model:", best_model_type)
    print("F1 Score:", best_f1)

    model_uri = f"runs:/{best_run_id}/model"
    print("\nRegistering model...")
    print("Model URI:", model_uri)

    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print("\nModel Registered Successfully")
    print("Model Name:", MODEL_NAME)
    print("Version:", model_version.version)

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Staging",
    )

    print("\nModel moved to Staging successfully.")
    print(f"Model URI: models:/{MODEL_NAME}/Staging")


if __name__ == "__main__":
    main()
