import math

import pandas as pd

from src.pipeline.collect import collect_data
from src.pipeline.features import engineer_features
from src.pipeline.preprocess import preprocess
from src.pipeline.validate import DataValidationError, validate
from src.feature_store import create_feature_store, get_features


def test_collect_data_creates_valid_csv(tmp_path):
    output = tmp_path / "iris_raw.csv"
    df = collect_data(str(output))

    assert list(df.columns)[:5] == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]
    assert "collected_at" in df.columns
    assert len(df) == 150
    assert output.exists()


def test_preprocess_handles_missing_and_duplicates():
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 5.1, None],
            "sepal width (cm)": [3.5, 3.5, 4.0],
            "petal length (cm)": [1.4, 1.4, 1.5],
            "petal width (cm)": [0.2, 0.2, 0.3],
            "species": ["setosa", "setosa", "versicolor"],
            "collected_at": ["x", "x", "y"],
        }
    )

    output = "data/processed/test_preprocessed.csv"
    cleaned = preprocess(df.to_csv(index=False), output_path=output)

    assert cleaned.shape[0] == 2
    assert "collected_at" not in cleaned.columns
    assert set(cleaned["species"].unique()) == {"setosa", "versicolor"}


def test_feature_engineering_and_validation():
    df = pd.DataFrame(
        {
            "sepal length (cm)": [5.1, 6.1],
            "sepal width (cm)": [3.5, 2.7],
            "petal length (cm)": [1.4, 4.5],
            "petal width (cm)": [0.2, 1.5],
            "species": ["setosa", "virginica"],
        }
    )

    output = "data/processed/test_features.csv"
    feature_df = engineer_features(df.to_csv(index=False), output_path=output)

    assert "sepal_area" in feature_df.columns
    assert "petal_area" in feature_df.columns
    assert "sepal_to_petal_length_ratio" in feature_df.columns
    assert "petal_length_bin" in feature_df.columns

    validated = validate(feature_df)
    assert list(validated.columns) == list(feature_df.columns)

    bad = feature_df.copy()
    bad.loc[0, "sepal length (cm)"] = 50
    try:
        validate(bad)
        assert False, "Expected DataValidationError for out-of-range value"
    except DataValidationError:
        pass


def test_feature_store_round_trip(tmp_path):
    source = tmp_path / "features.csv"
    store = tmp_path / "iris.db"
    engineer_features(
        pd.DataFrame(
            {
                "sepal length (cm)": [5.1, 6.1],
                "sepal width (cm)": [3.5, 2.7],
                "petal length (cm)": [1.4, 4.5],
                "petal width (cm)": [0.2, 1.5],
                "species": ["setosa", "virginica"],
            }
        ),
        output_path=str(source),
    )

    assert create_feature_store(source, store) == 2
    result = get_features(store, ["iris_00001", "iris_00000"], ["petal_area"])

    assert list(result["entity_id"]) == ["iris_00001", "iris_00000"]
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(result["petal_area"], [6.75, 0.28])
    )
