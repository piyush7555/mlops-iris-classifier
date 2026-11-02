# Continuous Integration Pipeline

## Overview

The GitHub Actions workflow in `.github/workflows/ci.yml` runs for pushes and
pull requests targeting `main` or `develop`. It uses Python 3.11 and caches pip
packages using `requirements.txt` as the cache key.

## Jobs

| Job | Purpose |
| --- | --- |
| `build-and-lint` | Installs project dependencies, compiles `src/` and `tests/`, and runs Flake8. |
| `test-suite` | Builds the Iris data pipeline, runs all pytest tests with coverage, publishes a Pytest Results check, and uploads JUnit XML and coverage XML reports. |
| `model-quality-gate` | Runs the dedicated deterministic model-quality test after the test suite passes. |

The jobs are sequential: `test-suite` requires `build-and-lint`, and
`model-quality-gate` requires `test-suite`. A failure therefore blocks all
dependent jobs. The model-quality check uses a stratified Iris test split, the
project's `train_model` function, and a minimum accuracy of 0.90.

## Pull-request quality gate

To require CI before merging into `main`, configure a GitHub branch ruleset
(Settings > Rules > Rulesets) targeting only `main`. Enable pull requests,
require status checks, and select `build-and-lint`, `test-suite`, and
`model-quality-gate`. Enable the up-to-date branch requirement if appropriate
for the repository.

GitHub settings cannot be applied by this repository workflow; an administrator
must create and activate this ruleset in the repository settings.

## Reports and local verification

The `test-and-coverage-reports` artifact contains `test-results.xml` and
`coverage.xml`. The workflow also adds a `Pytest Results` check to the
pull-request checks when the repository token has permission to read workflow
artifacts and write checks. GitHub restricts these permissions for pull requests
from forks, so the report check may not be created for those contributions.

Run the same checks locally with:

```powershell
python -m pip install -r requirements.txt
python -m pip install flake8 pytest pytest-cov
python -m compileall -q src tests
python -m flake8 src/ --max-line-length=100 --extend-ignore=E203 --statistics
python src/pipeline/collect.py --output data/raw/iris_raw.csv
python src/pipeline/preprocess.py --input data/raw/iris_raw.csv --output data/processed/iris_preprocessed.csv
python src/pipeline/features.py --input data/processed/iris_preprocessed.csv --output data/processed/iris_features.csv
python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml:coverage.xml --junitxml=test-results.xml
python -m pytest tests/test_model_quality.py -v
```

## Safely demonstrating regression detection

On a temporary feature branch only, change
`MIN_ACCEPTABLE_ACCURACY` in `tests/test_model_quality.py` from `0.90` to
`0.999`, then push the branch and open a pull request. The model-quality test
should fail because the deterministic model score is approximately `0.9667`.
Since the complete test suite includes this same quality test, `test-suite`
will fail and the dependent `model-quality-gate` job will be skipped. This
proves that the test suite detects the regression without changing `main`.
Restore the threshold to `0.90` and discard the temporary branch; do not merge
the failing demonstration pull request.
