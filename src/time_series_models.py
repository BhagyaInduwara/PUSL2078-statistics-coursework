"""
time_series_models.py

Functions for fitting ARIMA and SARIMA time series models.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(series: pd.Series, order: tuple = (1, 1, 1)):
    """Fit an ARIMA model to the given time series."""
    model = ARIMA(series, order=order)
    result = model.fit()
    return result


def fit_sarima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
):
    """Fit a SARIMA model to the given time series."""
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    return result


def forecast(result, steps: int = 10) -> pd.Series:
    """Generate a forecast for the given number of steps."""
    return result.forecast(steps=steps)
