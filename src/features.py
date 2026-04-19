# -*- coding: utf-8 -*-
"""
Feature selection methods used before model training.

Supported methods
-----------------
'mi'      : Mutual Information (mutual_info_regression)
'rf'      : Random Forest feature importance
'pearson' : Pearson correlation coefficient
None      : Use all available features without selection
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def select_mi_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        target: str, k: int = 10):
    """Select top-k features by Mutual Information with the target."""
    X_tr = df_train.drop(columns=[target])
    X_te = df_test.drop(columns=[target])

    def mi_scorer(X, y):
        return mutual_info_regression(X, y, random_state=0)

    selector = SelectKBest(score_func=mi_scorer, k=k)
    X_tr_sel = selector.fit_transform(X_tr, df_train[target].values)
    X_te_sel = selector.transform(X_te)

    feat_names = X_tr.columns[selector.get_support()].tolist()
    return X_tr_sel, X_te_sel, feat_names


def select_rf_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                        target: str, k: int = 10):
    """Select top-k features by Random Forest feature importance."""
    X_tr = df_train.drop(columns=[target])
    X_te = df_test.drop(columns=[target])

    rf = RandomForestRegressor(random_state=0, n_jobs=-1)
    rf.fit(X_tr, df_train[target].values)

    importances = pd.Series(rf.feature_importances_, index=X_tr.columns)
    top_feats = importances.nlargest(k).index.tolist()
    return X_tr[top_feats].values, X_te[top_feats].values, top_feats


def select_pearson_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                             target: str, k: int = 10):
    """Select top-k features by absolute Pearson correlation with the target."""
    corr = df_train.corr()[target].abs().drop(target)
    top_feats = corr.nlargest(k).index.tolist()
    return df_train[top_feats].values, df_test[top_feats].values, top_feats


def select_features(df_train: pd.DataFrame, df_test: pd.DataFrame,
                    target: str, method: str = 'mi', k: int = 10):
    """Unified entry point for feature selection.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        DataFrames that include the target column.
    target : str
        Name of the target column.
    method : str or None
        Feature selection method ('mi', 'rf', 'pearson', or None).
    k : int
        Number of features to select.

    Returns
    -------
    X_tr, X_te : np.ndarray
    feat_names : list[str]
    """
    if method is None or str(method).lower() == 'none':
        feat_names = df_train.drop(columns=[target]).columns.tolist()
        return (df_train[feat_names].values,
                df_test[feat_names].values,
                feat_names)

    method = method.lower()
    if method == 'mi':
        return select_mi_features(df_train, df_test, target, k=k)
    elif method == 'rf':
        return select_rf_features(df_train, df_test, target, k=k)
    elif method == 'pearson':
        return select_pearson_features(df_train, df_test, target, k=k)
    else:
        raise ValueError(f"Unknown feature selection method: '{method}'. "
                         "Choose from 'mi', 'rf', 'pearson', or None.")
