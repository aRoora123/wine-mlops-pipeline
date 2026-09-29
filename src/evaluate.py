import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_and_split_data


MODEL_NAME = "WineClassifier"
MODEL_ALIAS = "champion"


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    f1 = f1_score(y_test, predictions, average="macro")
    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    print("Test F1:", round(f1, 4))
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Log Loss:", round(loss, 4))


if __name__ == "__main__":
    main()
