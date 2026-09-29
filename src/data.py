from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_and_split_data():
    wine = load_wine()

    X = wine.data
    y = wine.target

    if X.shape[1] != 13:
        raise ValueError(f"Expected 13 features, got {X.shape[1]}.")

    if X.shape[0] != 178:
        raise ValueError(f"Expected 178 samples, got {X.shape[0]}.")

    if X.shape[0] != len(y):
        raise ValueError("Features and targets have different lengths.")

    if not X.size:
        raise ValueError("Feature data is empty.")

    if not y.size:
        raise ValueError("Target data is empty.")

    if not (X == X).all():
        raise ValueError("Feature data contains invalid values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
