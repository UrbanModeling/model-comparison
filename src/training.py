# -*- coding: utf-8 -*-
"""
Core training pipeline: cross-validated grid search and per-model task runner.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

from src.data import save_dataframe
from src.evaluation import compute_metrics, export_cv_folds, export_test_metrics
from src.uncertainty import predict_with_uncertainty


def train_model_cv(regressor, param_grid: dict,
                   X: np.ndarray, y: np.ndarray,
                   k_folds: int = 5):
    """Fit a StandardScaler → regressor pipeline via grid-search cross-validation.

    Parameters
    ----------
    regressor : sklearn-compatible estimator
    param_grid : dict
        Parameter grid for GridSearchCV. Keys must be prefixed with 'reg__'.
    X, y : np.ndarray
        Training features and labels.
    k_folds : int
        Number of CV folds.

    Returns
    -------
    best_model : fitted Pipeline
    best_cv_mse : float
    cv_train_time : float
        Wall-clock seconds for the entire grid search.
    cv_results : dict
        Raw cv_results_ dict from GridSearchCV.
    """
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reg',   regressor),
    ])
    cv   = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    grid.fit(X, y)
    cv_train_time = time.time() - t0

    return grid.best_estimator_, -grid.best_score_, cv_train_time, grid.cv_results_


def run_single_model_task(name: str, train_fn,
                          X_tr: np.ndarray, y_tr: np.ndarray,
                          X_te: np.ndarray, y_te: np.ndarray,
                          out_root: str, k_folds: int,
                          do_uncertainty: bool = False,
                          feature_method: str = None,
                          n_features: int = None,
                          scale: str = 't',
                          ids: np.ndarray = None,
                          years: np.ndarray = None,
                          n_bootstrap: int = 20) -> None:
    """Train one model, evaluate it, and optionally run uncertainty analysis.

    All results are saved as CSV files under ``{out_root}/{name}/``.

    Parameters
    ----------
    name : str
        Short model identifier used in output filenames.
    train_fn : callable
        Model-specific training function from ``src.models``.
        Signature: ``(X, y, k_folds) -> (model, best_mse, train_time, cv_results)``.
    X_tr, y_tr : np.ndarray
        Pre-selected training features and labels.
    X_te, y_te : np.ndarray
        Pre-selected test features and labels.
    out_root : str
        Root output directory; a subdirectory ``name`` will be created inside.
    k_folds : int
        Number of cross-validation folds.
    do_uncertainty : bool
        Whether to run bootstrap uncertainty estimation.
    feature_method : str or None
        Label for the feature selection method (used in filenames only).
    n_features : int or None
        Number of selected features (used in filenames only).
    scale : str
        Data split type ('s' or 't'; used in filenames only).
    ids : np.ndarray or None
        uid values for the test set rows (prepended to uncertainty output).
    years : np.ndarray or None
        year values for the test set rows (prepended to uncertainty output).
    n_bootstrap : int
        Bootstrap iterations for uncertainty estimation.
    """
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    method_tag = feature_method or 'raw'
    prefix = f"{scale}_{name}_{n_features}feat_{method_tag}_{k_folds}folds"

    # ── 1. Cross-validated training ───────────────────────────────────────────
    print(f"\n[{name}] Training ...")
    t0 = time.time()
    model, best_mse, train_time, cv_results = train_fn(X_tr, y_tr, k_folds=k_folds)
    print(f"[{name}] CV best MSE={best_mse:.4f} | wall time={time.time()-t0:.1f}s")

    save_dataframe(
        pd.DataFrame([{'best_cv_mse': best_mse, 'train_time_s': train_time}]),
        os.path.join(out_dir, f"{prefix}_train_summary.csv"),
    )

    # Export per-fold CV metrics
    export_cv_folds(model, X_tr, y_tr, out_dir, prefix, k_folds)

    # Export grid-search result table
    try:
        save_dataframe(pd.DataFrame(cv_results),
                       os.path.join(out_dir, f"{prefix}_gridsearch_results.csv"))
    except Exception:
        pass

    # ── 2. Test-set evaluation ────────────────────────────────────────────────
    export_test_metrics(model, X_te, y_te, out_dir, prefix)

    # ── 3. Bootstrap uncertainty estimation ──────────────────────────────────
    if do_uncertainty:
        unc_df, boot_preds = predict_with_uncertainty(
            model, X_tr, y_tr, X_te,
            n_bootstrap=n_bootstrap,
            random_seed=0,
            return_bootstrap=True,
        )
        unc_df['y_true'] = y_te

        if ids is not None:
            unc_df.insert(0, 'uid', ids)
        if years is not None:
            col_idx = 1 if ids is not None else 0
            unc_df.insert(col_idx, 'year', years)

        save_dataframe(unc_df, os.path.join(out_dir, f"{prefix}_uncertainty.csv"))

        boot_records = [
            {**compute_metrics(y_te, preds), 'bootstrap': i}
            for i, preds in enumerate(boot_preds, start=1)
        ]
        save_dataframe(pd.DataFrame(boot_records),
                       os.path.join(out_dir, f"{prefix}_bootstrap_metrics.csv"))
    else:
        print(f"[{name}] Uncertainty analysis skipped.")
