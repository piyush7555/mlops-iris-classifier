"""
Stage 3: Feature Engineering.
Derives new, model-useful features from the raw measurements.
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

logger = logging.getLogger("features")


def _read_input_frame(input_data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()

    if not isinstance(input_data, str):
        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    path = Path(input_data)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(StringIO(input_data))


def engineer_features(input_path: str | pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    df = _read_input_frame(input_path)

    df["sepal_area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]
    df["sepal_to_petal_length_ratio"] = (
        df["sepal length (cm)"] / df["petal length (cm)"].replace(0, pd.NA)
    )
    df["petal_length_bin"] = pd.cut(
        df["petal length (cm)"],
        bins=[0, 2, 4.5, 7],
        labels=["short", "medium", "long"],
    )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

    logger.info("Engineered %d features -> %s", df.shape[1], output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_preprocessed.csv")
    parser.add_argument("--output", default="data/processed/iris_features.csv")

    args = parser.parse_args()
    engineer_features(args.input, args.output)