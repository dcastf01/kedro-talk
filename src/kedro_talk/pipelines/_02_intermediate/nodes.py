import logging
from typing import Tuple

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[data.columns.drop('target')]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters['test_size'], random_state=parameters['random_state']
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = classifier.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    return {'balanced_accuracy': balanced_accuracy}
