"""
Stage 4: Data Validation.
Schema and statistical checks that must pass before data can
flow downstream to training. Raises on failure to hard-stop the pipeline.
"""

import argparse
import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("validate")

EXPECTED_COLUMNS = {
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
}

VALID_SPECIES = {"setosa", "versicolor", "virginica"}

RANGE_CHECKS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}


class DataValidationError(Exception):
    pass


def _read_input_frame(input_data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if not isinstance(input_data, str):
        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    path = Path(input_data)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(StringIO(input_data))


def validate(input_path: str | pd.DataFrame) -> pd.DataFrame:
    df = _read_input_frame(input_path)

    errors: list[str] = []

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Unexpected null values in columns: {null_cols}")

    invalid_species = set(df["species"].unique()) - VALID_SPECIES
    if invalid_species:
        errors.append(f"Unexpected species values: {invalid_species}")

    for col, (low, high) in RANGE_CHECKS.items():
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            errors.append(
                f"{len(out_of_range)} rows out of expected range for '{col}' ({low}-{high})"
            )

    if errors:
        for error in errors:
            logger.error(error)
        raise DataValidationError(f"Validation failed with {len(errors)} error(s)")

    logger.info(
        "Validation PASSED: %d rows, %d columns, all checks satisfied",
        len(df),
        df.shape[1],
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_features.csv")
    args = parser.parse_args()

    try:
        validate(args.input)
    except DataValidationError as exc:
        logger.error("Pipeline halted: %s", exc)
        sys.exit(1)