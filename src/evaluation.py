# -*- coding: utf-8 -*-
"""
Regression metrics and result export utilities.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone

from src.data import save_dataframe


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    Returns
    -------
    dict with keys: 'mse', 'rmse', 'mae', 'r2'
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mse':  mse,
        'rmse': np.sqrt(mse),
        'mae':  mean_absolute_error(y_true, y_pred),
        'r2':   r2_score(y_true, y_pred),
    }


def export_cv_folds(best_model, X_tr: np.ndarray, y_tr: np.ndarray,
                    out_dir: str, prefix: str, k_folds: int) -> None:
    """Re-run k-fold CV on the best model and save per-fold metrics."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    records = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr), start=1):
        m = clone(best_model)
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        pred = m.predict(X_tr[val_idx])
        metrics = compute_metrics(y_tr[val_idx], pred)
        metrics['fold'] = fold
        records.append(metrics)

    save_dataframe(pd.DataFrame(records),
                   os.path.join(out_dir, f"{prefix}_cv_fold_metrics.csv"))


def export_test_metrics(best_model, X_te: np.ndarray, y_te: np.ndarray,
                         out_dir: str, prefix: str) -> np.ndarray:
    """Predict on the test set, save metrics, and return predictions.

    Returns
    -------
    y_pred : np.ndarray
    """
    t0 = time.time()
    y_pred = best_model.predict(X_te)
    elapsed = time.time() - t0

    metrics = compute_metrics(y_te, y_pred)
    metrics['prediction_time_s'] = elapsed
    print(f"  -> Test prediction time: {elapsed:.2f}s")

    save_dataframe(pd.DataFrame([metrics]),
                   os.path.join(out_dir, f"{prefix}_test_metrics.csv"))
    return y_pred
