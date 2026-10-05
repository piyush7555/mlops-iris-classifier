"""Small, reproducible local feature store for the Iris pipeline."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Sequence

import pandas as pd


FEATURE_COLUMNS = [
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
]
ENTITY_COLUMN = "entity_id"
TIMESTAMP_COLUMN = "event_timestamp"
DEFAULT_TIMESTAMP = "1970-01-01T00:00:00+00:00"


class FeatureStoreError(ValueError):
    """Raised when feature data does not match the store contract."""


def prepare_feature_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add stable entity and timestamp columns to engineered features."""
    missing = set(FEATURE_COLUMNS) - set(data.columns)
    if missing:
        raise FeatureStoreError(f"Missing feature columns: {sorted(missing)}")

    result = data.copy()
    if ENTITY_COLUMN not in result:
        result.insert(0, ENTITY_COLUMN, [f"iris_{index:05d}" for index in range(len(result))])
    if result[ENTITY_COLUMN].duplicated().any():
        raise FeatureStoreError("entity_id values must be unique")

    if TIMESTAMP_COLUMN not in result:
        result.insert(1, TIMESTAMP_COLUMN, DEFAULT_TIMESTAMP)
    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return result


def create_feature_store(input_path: str | Path, store_path: str | Path) -> int:
    """Write engineered features and their schema to a SQLite feature store."""
    data = prepare_feature_data(pd.read_csv(input_path))
    store = Path(store_path)
    store.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(store)) as connection:
        connection.execute("DROP TABLE IF EXISTS iris_features")
        connection.execute("DROP TABLE IF EXISTS feature_schema")
        data.to_sql("iris_features", connection, index=False)
        schema = {"entity": ENTITY_COLUMN, "timestamp": TIMESTAMP_COLUMN, "features": FEATURE_COLUMNS}
        connection.execute(
            "CREATE TABLE feature_schema (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO feature_schema(name, value) VALUES (?, ?)",
            ("iris", json.dumps(schema)),
        )
        connection.commit()
    return len(data)


def get_features(
    store_path: str | Path,
    entity_ids: Sequence[str],
    features: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Retrieve feature values for entity IDs in the requested order."""
    unknown = set(features) - set(FEATURE_COLUMNS)
    if unknown:
        raise FeatureStoreError(f"Unknown features: {sorted(unknown)}")
    if not entity_ids:
        return pd.DataFrame(columns=[ENTITY_COLUMN, *features])

    selected = ", ".join([ENTITY_COLUMN, *features])
    placeholders = ", ".join("?" for _ in entity_ids)
    query = f"SELECT {selected} FROM iris_features WHERE {ENTITY_COLUMN} IN ({placeholders})"
    with closing(sqlite3.connect(store_path)) as connection:
        result = pd.read_sql_query(query, connection, params=list(entity_ids))

    if len(result) != len(entity_ids):
        found = set(result[ENTITY_COLUMN])
        missing = [entity_id for entity_id in entity_ids if entity_id not in found]
        raise FeatureStoreError(f"Unknown entity IDs: {missing}")
    result[ENTITY_COLUMN] = pd.Categorical(
        result[ENTITY_COLUMN], categories=list(entity_ids), ordered=True
    )
    return result.sort_values(ENTITY_COLUMN).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_features.csv")
    parser.add_argument("--store", default="data/feature_store/iris.db")
    args = parser.parse_args()
    print(f"Stored {create_feature_store(args.input, args.store)} feature rows")