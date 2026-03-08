"""
regression.py

Functions for fitting and evaluating linear regression models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def fit_linear_regression(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Fit a linear regression model and return the model and evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    return model, metrics


def get_coefficients(model: LinearRegression, feature_names: list) -> pd.DataFrame:
    """Return model coefficients as a DataFrame, with intercept as a separate row."""
    rows = [{"feature": "(intercept)", "coefficient": model.intercept_}]
    rows += [
        {"feature": name, "coefficient": coef}
        for name, coef in zip(feature_names, model.coef_)
    ]
    return pd.DataFrame(rows)
