import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from feast import FeatureStore


FEATURE_NAMES = [
    "iris_measurements:sepal length (cm)",
    "iris_measurements:sepal width (cm)",
    "iris_measurements:petal length (cm)",
    "iris_measurements:petal width (cm)",
    "iris_engineered_features:sepal_area",
    "iris_engineered_features:petal_area",
    "iris_engineered_features:sepal_to_petal_length_ratio",
    "iris_engineered_features:petal_length_bin",
]


def _feast_cli() -> list[str]:
    feast_bin = shutil.which("feast")
    if feast_bin:
        return [feast_bin]

    scripts_dir = Path(sys.prefix) / "Scripts"
    candidate = scripts_dir / ("feast.exe" if os.name == "nt" else "feast")
    if candidate.exists():
        return [str(candidate)]

    return ["feast"]


def run_demo():
    repo_path = Path(__file__).resolve().parent
    store = FeatureStore(repo_path=str(repo_path))
    feast_cmd = _feast_cli()

    print("\n--- Run feast apply ---")
    subprocess.run(feast_cmd + ["--chdir", str(repo_path), "apply"], check=True)

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=False)

    print("\n--- Historical features for batch scoring ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=True)

    print("\n--- Load features into online store ---")
    store.materialize_incremental(end_date=datetime.now())

    print("\n--- Online features ---")
    fetch_online_features(store)

    print("\n--- Online features retrieved (instead) through a feature service---")
    fetch_online_features(store, source="feature_service")

    print("\n--- Run feast teardown (best effort) ---")
    store.close()
    try:
        subprocess.run(feast_cmd + ["--chdir", str(repo_path), "teardown"], check=True)
    except (PermissionError, subprocess.CalledProcessError):
        print("Feast teardown skipped because the SQLite online store is still locked.")


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool):
    source_df = pd.read_parquet(repo_path / "data" / "iris_features.parquet")
    entity_df = source_df[["sample_id", "event_timestamp"]].head(5).copy()

    if for_batch_scoring:
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_NAMES,
    ).to_df()
    print(training_df.head())


def fetch_online_features(store, source: str = ""):
    entity_rows = [{"sample_id": 1}, {"sample_id": 2}]
    if source == "feature_service":
        features_to_fetch = store.get_feature_service("iris_feature_service")
    else:
        features_to_fetch = FEATURE_NAMES

    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)


repo_path = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_demo()
