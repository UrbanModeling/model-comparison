# -*- coding: utf-8 -*-
"""
Model registry: one training function per algorithm.

Each function wraps ``train_model_cv`` with a model-specific parameter grid.
All functions share the same signature::

    train_<name>_cv(X, y, k_folds=5) -> (model, best_cv_mse, train_time, cv_results)

To enable or disable a model in the pipeline, comment/uncomment the
corresponding entry in the ``MODELS`` dict at the bottom of this file.
"""

import warnings
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from skelm import ELMRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDRegressor, HuberRegressor, TweedieRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,
    BaggingRegressor, HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from src.training import train_model_cv

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ── Linear models ─────────────────────────────────────────────────────────────

def train_linear_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__fit_intercept': [True, False]}
    return train_model_cv(LinearRegression(), param_grid, X, y, k_folds)


def train_ridge_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__alpha': [0.01, 0.1, 1.0, 10.0, 100],
        'reg__fit_intercept': [True, False],
    }
    return train_model_cv(Ridge(), param_grid, X, y, k_folds)


def train_glm_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__alpha': [0.01, 0.1, 1.0, 10.0, 100]}
    return train_model_cv(TweedieRegressor(max_iter=1000), param_grid, X, y, k_folds)


def train_sgd_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
        'reg__eta0':  [0.001, 0.01, 0.05, 0.1, 0.2],
    }
    return train_model_cv(SGDRegressor(random_state=0), param_grid, X, y, k_folds)


def train_huber_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],
        'reg__alpha':   [1e-4, 1e-3, 1e-2, 1e-1, 1],
    }
    return train_model_cv(HuberRegressor(max_iter=10000), param_grid, X, y, k_folds)


def train_bayesianridge_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__alpha_1': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        'reg__alpha_2': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    }
    return train_model_cv(BayesianRidge(), param_grid, X, y, k_folds)


# ── Instance-based / kernel models ────────────────────────────────────────────

def train_knn_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__n_neighbors': [3, 7, 15, 31, 63],
        'reg__weights':     ['uniform', 'distance'],
    }
    return train_model_cv(KNeighborsRegressor(n_jobs=-1), param_grid, X, y, k_folds)


def train_svr_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]}
    return train_model_cv(SVR(), param_grid, X, y, k_folds)


# ── Neural networks ───────────────────────────────────────────────────────────

def train_mlp_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    }
    return train_model_cv(MLPRegressor(random_state=0), param_grid, X, y, k_folds)


def train_elm_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__n_neurons': [50, 100, 200, 500, 1000],
        'reg__alpha':     [0.01, 0.1, 1.0, 10.0, 100],
    }
    return train_model_cv(ELMRegressor(random_state=0), param_grid, X, y, k_folds)


# ── Tree-based ensemble models ────────────────────────────────────────────────

def train_rf_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__n_estimators': [100, 200, 300, 500, 1000]}
    return train_model_cv(
        RandomForestRegressor(random_state=0, n_jobs=-1), param_grid, X, y, k_folds)


def train_adaboost_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__n_estimators':  [100, 200, 300, 500, 1000],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    }
    return train_model_cv(
        AdaBoostRegressor(random_state=0), param_grid, X, y, k_folds)


def train_gbdt_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]}
    return train_model_cv(
        GradientBoostingRegressor(random_state=0), param_grid, X, y, k_folds)


def train_bagging_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {'reg__n_estimators': [10, 50, 100, 200, 500]}
    return train_model_cv(
        BaggingRegressor(random_state=0, n_jobs=-1), param_grid, X, y, k_folds)


def train_histgb_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__max_iter':      [100, 200, 300, 500, 1000],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    }
    return train_model_cv(
        HistGradientBoostingRegressor(random_state=0), param_grid, X, y, k_folds)


# ── Third-party boosting libraries ────────────────────────────────────────────

def train_xgb_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__n_estimators':  [100, 200, 300, 500, 1000],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    }
    return train_model_cv(
        xgb.XGBRegressor(objective='reg:squarederror', random_state=0, n_jobs=-1),
        param_grid, X, y, k_folds)


def train_lgb_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__n_estimators':  [100, 200, 300, 500, 1000],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    }
    return train_model_cv(
        lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1),
        param_grid, X, y, k_folds)


def train_catboost_cv(X: np.ndarray, y: np.ndarray, k_folds: int = 5):
    param_grid = {
        'reg__iterations':    [100, 200, 300, 500, 1000],
        'reg__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    }
    return train_model_cv(
        CatBoostRegressor(random_state=0, verbose=0,
                          thread_count=1, allow_writing_files=False),
        param_grid, X, y, k_folds)


# ── Model registry ────────────────────────────────────────────────────────────
# Comment out any model to exclude it from the benchmark run.

MODELS = {
    # Linear models
    # 'linear':        train_linear_cv,
    # 'ridge':         train_ridge_cv,
    # 'bayesianridge': train_bayesianridge_cv,
    # 'sgd':           train_sgd_cv,
    # 'huber':         train_huber_cv,
    # 'glm':           train_glm_cv,

    # Tree / ensemble models
    # 'rf':       train_rf_cv,
    # 'gbdt':     train_gbdt_cv,
    # 'adaboost': train_adaboost_cv,
    'bagging':  train_bagging_cv,
    'histgb':   train_histgb_cv,

    # Other sklearn models
    'knn': train_knn_cv,
    'mlp': train_mlp_cv,
    'svr': train_svr_cv,

    # Third-party libraries
    'xgb':      train_xgb_cv,
    'lgb':      train_lgb_cv,
    'catboost': train_catboost_cv,
    'elm':      train_elm_cv,
}
