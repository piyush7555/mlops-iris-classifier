from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.train import train_model

MIN_ACCEPTABLE_ACCURACY = 0.90


def test_model_meets_minimum_accuracy_threshold():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.20,
        random_state=42,
        stratify=iris.target,
    )
    model = train_model(X_train, y_train, n_estimators=50, max_depth=3)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    assert accuracy >= MIN_ACCEPTABLE_ACCURACY, (
        f"Model accuracy {accuracy:.4f} is below the required "
        f"threshold {MIN_ACCEPTABLE_ACCURACY:.3f}"
    )
