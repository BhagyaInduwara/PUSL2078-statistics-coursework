"""
data_preprocessing.py

Utilities for loading, cleaning, and transforming raw datasets.
"""

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load a CSV dataset from the given file path."""
    return pd.read_csv(filepath)


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing values."""
    return df.dropna()


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Fill missing values using the specified strategy ('mean', 'median', or 'mode')."""
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        mode_values = df.mode()
        if mode_values.empty:
            return df
        return df.fillna(mode_values.iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def save_cleaned(df: pd.DataFrame, filepath: str) -> None:
    """Save the cleaned DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
