import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data import load_and_split_data


RANDOM_STATE = 42
EXPERIMENT_NAME = "Wine-Cultivar-Classification"


def get_model_configs():
    return [
        ("random_forest_1",
         RandomForestClassifier(n_estimators=100, random_state=42)),

        ("random_forest_2",
         RandomForestClassifier(
             n_estimators=200, max_depth=5, random_state=42)),

        ("random_forest_3",
         RandomForestClassifier(
             n_estimators=300, max_depth=10, random_state=42)),

        ("gradient_boosting_1",
         GradientBoostingClassifier(
             n_estimators=100, learning_rate=0.1,
             max_depth=3, random_state=42)),

        ("gradient_boosting_2",
         GradientBoostingClassifier(
             n_estimators=150, learning_rate=0.05,
             max_depth=3, random_state=42)),

        ("gradient_boosting_3",
         GradientBoostingClassifier(
             n_estimators=200, learning_rate=0.1,
             max_depth=2, random_state=42)),
    ]


def get_metrics(y_true, predictions, probabilities):
    return {
        "f1": f1_score(y_true, predictions, average="macro"),
        "accuracy": accuracy_score(y_true, predictions),
        "log_loss": log_loss(y_true, probabilities)
    }


def evaluate_model(model, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_true = []
    train_pred = []
    train_prob = []

    val_true = []
    val_pred = []
    val_prob = []

    for train_idx, val_idx in cv.split(X, y):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        model.fit(X_train, y_train)

        train_true.extend(y_train)
        train_pred.extend(model.predict(X_train))
        train_prob.extend(model.predict_proba(X_train))

        val_true.extend(y_val)
        val_pred.extend(model.predict(X_val))
        val_prob.extend(model.predict_proba(X_val))

    train_metrics = get_metrics(
        train_true, train_pred, train_prob
    )

    val_metrics = get_metrics(
        val_true, val_pred, val_prob
    )

    return train_metrics, val_metrics


def log_model(name, model, X_train, y_train):
    train_metrics, val_metrics = evaluate_model(
        model, X_train, y_train
    )

    model.fit(X_train, y_train)

    signature = infer_signature(
        X_train,
        model.predict(X_train)
    )

    mlflow.log_params(model.get_params())

    mlflow.log_metrics({
        "train_macro_f1": train_metrics["f1"],
        "train_accuracy": train_metrics["accuracy"],
        "train_log_loss": train_metrics["log_loss"],
        "val_macro_f1": val_metrics["f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_log_loss": val_metrics["log_loss"]
    })

    mlflow.set_tag("model_name", name)
    mlflow.set_tag("model_type", type(model).__name__)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:3]
    )

    return train_metrics, val_metrics


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split_data()

    best_run_id = None
    best_f1 = -1
    best_log_loss = float("inf")

    for name, model in get_model_configs():

        with mlflow.start_run(run_name=name) as run:
            train_metrics, val_metrics = log_model(
                name, model, X_train, y_train
            )

            val_f1 = val_metrics["f1"]
            val_log_loss = val_metrics["log_loss"]

            if val_f1 > best_f1 or (
                val_f1 == best_f1
                and val_log_loss < best_log_loss
            ):
                best_run_id = run.info.run_id
                best_f1 = val_f1
                best_log_loss = val_log_loss

        print(
            name,
            "Val F1:", round(val_f1, 4),
            "Val Accuracy:", round(val_metrics["accuracy"], 4),
            "Val Log Loss:", round(val_log_loss, 4)
        )

    client = MlflowClient()

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="WineClassifier"
    )

    client.set_registered_model_alias(
        name="WineClassifier",
        alias="champion",
        version=registered_model.version
    )

    print("\nBest model:", best_run_id)
    print("WineClassifier version:", registered_model.version)


if __name__ == "__main__":
    main()
