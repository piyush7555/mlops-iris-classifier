"""
Stage 2: Data Preprocessing.
Handles missing values, duplicate removal, and type correction.
"""

import argparse
import logging
from io import StringIO
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("preprocess")

NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def _read_input_frame(input_data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if not isinstance(input_data, str):
        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    path = Path(input_data)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(StringIO(input_data))


def preprocess(input_path: str | pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    df = _read_input_frame(input_path)

    initial_rows = len(df)

    df = df.drop_duplicates()
    logger.info("Dropped %d duplicate rows", initial_rows - len(df))

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(
                "Imputed %d missing values in '%s' with median=%.3f",
                n_missing,
                col,
                median_val,
            )

    df = df.dropna(subset=["species"])
    df.drop(columns=["collected_at"], inplace=True, errors="ignore")

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    logger.info("Preprocessed %d rows -> %s", len(df), output_path)
    return df


def preprocess_data(input_path: str | pd.DataFrame, output_path: str) -> pd.DataFrame:
    return preprocess(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/iris_raw.csv")
    parser.add_argument("--output", default="data/processed/iris_preprocessed.csv")

    args = parser.parse_args()
    preprocess(args.input, args.output)