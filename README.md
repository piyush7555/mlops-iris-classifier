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
