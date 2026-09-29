import numpy as np

from src.data import load_and_split_data


def test_data_split_shape():
    X_train, X_test, y_train, y_test = load_and_split_data()

    assert X_train.shape == (142, 13)
    assert X_test.shape == (36, 13)
    assert y_train.shape == (142,)
    assert y_test.shape == (36,)


def test_data_has_no_missing_values():
    X_train, X_test, y_train, y_test = load_and_split_data()

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()


def test_train_test_sizes():
    X_train, X_test, y_train, y_test = load_and_split_data()

    assert len(X_train) + len(X_test) == 178
    assert len(y_train) + len(y_test) == 178
