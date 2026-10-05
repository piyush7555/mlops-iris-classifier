# Mlops-iris-classifier

A sample ML project used to demonstrate Git-based version control
workflows in an MLOps context.

## Setup
\`\`\`bash
pip install -r requirements.txt
python src/train.py

## Feature store

The feature stage writes engineered features to `data/processed/iris_features.csv`.
The DVC `feature_store` stage loads them into the reproducible local SQLite store
at `data/feature_store/iris.db`. Each row has a stable `entity_id`, an
`event_timestamp`, and the four registered feature columns.

```bash
python src/feature_store.py
```

Feature values can be retrieved by entity ID with `get_features` from
`src.feature_store`.

## MLflow experiment tracking

The repo includes MLflow scripts for experiment tracking and model registration.
Install the project dependencies and start the tracking server before running them:

```bash
python -m pip install -r requirements.txt
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then, in a second terminal, run:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train_with_mlflow.py
python src/register_best_model.py
python src/load_registered_model.py
```

The training script logs each model configuration as a separate MLflow run under
`iris-classification-baseline`, stores confusion-matrix artifacts, and saves the
trained model for registration. The registration script selects the best run by
`f1_macro` and promotes the model to the `Staging` stage in the MLflow Model Registry.
