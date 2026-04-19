# -*- coding: utf-8 -*-
"""
Bootstrap-based uncertainty estimation for regression models.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone


def predict_with_uncertainty(model, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, n_bootstrap: int = 20,
                              random_seed: int = None,
                              return_bootstrap: bool = False):
    """Estimate prediction uncertainty via bootstrap resampling.

    For each bootstrap iteration the model is cloned, fitted on a resampled
    training set, and used to predict on the test set. The 2.5th and 97.5th
    percentiles across iterations form the 95 % confidence interval.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
        The best model returned by the CV training step.
    X_train, y_train : np.ndarray
        Training features and labels.
    X_test : np.ndarray
        Test features.
    n_bootstrap : int
        Number of bootstrap iterations.
    random_seed : int or None
        Seed for reproducibility.
    return_bootstrap : bool
        If True, also return the raw bootstrap prediction matrix.

    Returns
    -------
    df : pd.DataFrame
        Columns: 'co2_pred', 'co2_pred_lower', 'co2_pred_upper'.
    boot_preds : np.ndarray, shape (n_bootstrap, n_test)
        Only returned when return_bootstrap=True.
    """
    rng = np.random.RandomState(random_seed)
    boot_preds = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        m = clone(model)
        m.fit(X_train[idx], y_train[idx])
        boot_preds.append(m.predict(X_test))

    arr = np.vstack(boot_preds)  # (n_bootstrap, n_test)
    df = pd.DataFrame({
        'co2_pred':       arr.mean(axis=0),
        'co2_pred_lower': np.percentile(arr, 2.5,  axis=0),
        'co2_pred_upper': np.percentile(arr, 97.5, axis=0),
    })

    if return_bootstrap:
        return df, arr
    return df
