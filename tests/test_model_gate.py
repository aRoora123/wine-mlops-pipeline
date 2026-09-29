import time

import mlflow.sklearn
from sklearn.metrics import f1_score

from src.data import load_and_split_data


MODEL_NAME = "WineClassifier"
MODEL_ALIAS = "champion"


def test_model_gate():
    X_train, X_test, y_train, y_test = load_and_split_data()

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions, average="macro")
    assert f1 >= 0.88

    start = time.perf_counter()
    model.predict(X_test)
    elapsed = (time.perf_counter() - start) * 1000

    assert elapsed <= 30

    assert set(predictions).issubset({0, 1, 2})
