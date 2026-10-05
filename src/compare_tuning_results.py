"""Summarize the baseline, grid-search, and random-search MLflow runs."""

import os

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
client = MlflowClient()

experiment = client.get_experiment_by_name("iris-hyperparameter-tuning")
if experiment is None:
    raise RuntimeError("Experiment 'iris-hyperparameter-tuning' not found.")

runs = client.search_runs(experiment_ids=[experiment.experiment_id])

print(f"{'Run Name':<28}{'CV f1_macro':<15}{'Test Accuracy':<15}{'Total Fits':<12}")
print("-" * 70)
for run in sorted(runs, key=lambda item: item.data.tags.get("mlflow.runName", "")):
    name = run.data.tags.get("mlflow.runName", "unknown")
    cv_score = run.data.metrics.get("cv_f1_macro_mean", run.data.metrics.get("best_cv_f1_macro", 0.0))
    test_acc = run.data.metrics.get("test_accuracy", 0.0)
    n_iter = run.data.params.get("total_combinations") or run.data.params.get("n_iter") or "1"
    total_fits = int(n_iter) * 5 if name != "baseline_decision_tree" else 5
    print(f"{name:<28}{cv_score:<15.4f}{test_acc:<15.4f}{total_fits:<12}")
