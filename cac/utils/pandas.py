"""Utility functions for pandas operations"""
import numpy as np
import pandas as pd


def apply_filters(df: pd.DataFrame, filters: dict, reset_index=False):
    """
    Filters df based on given filters (key-values pairs).
    """
    X = df.copy()

    for col, values in filters.items():
        if isinstance(values, (list, tuple, np.ndarray)):
            indices = X[col].isin(list(values))
        else:
            indices = X[col] == values
        X = X[indices]

    if reset_index:
        X = X.reset_index(drop=True)

    return X


def apply_antifilters(df: pd.DataFrame, filters: dict, reset_index=False):
    """
    Filters df removing rows for given filters (key-values pairs).
    """
    X = df.copy()

    for col, values in filters.items():
        if isinstance(values, (list, tuple, np.ndarray)):
            indices = X[col].isin(list(values))
        else:
            indices = X[col] == values
        X = X[~indices]

    if reset_index:
        X = X.reset_index(drop=True)

    return X