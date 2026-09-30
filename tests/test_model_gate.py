import time

import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.data import load_and_split_data


MODEL_NAME = "WineClassifier"
MODEL_ALIAS = "champion"


def test_model_gate():
    _, X_test, _, _ = load_and_split_data()

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    run = client.get_run(model_version.run_id)

    validation_f1 = run.data.metrics["val_macro_f1"]
    assert validation_f1 >= 0.88

    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed = (time.perf_counter() - start) * 1000

    assert elapsed <= 30

    assert set(predictions).issubset({0, 1, 2})
