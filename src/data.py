# -*- coding: utf-8 -*-
"""
Data I/O utilities: loading train/test CSVs and saving result DataFrames.
"""

import os
import pandas as pd


def load_train_test(train_path: str, test_path: str):
    """Load train and test CSV files.

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the test CSV file.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test  = pd.read_csv(test_path,  encoding="utf-8-sig")
    return df_train, df_test


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to CSV (UTF-8 with BOM to avoid encoding issues).

    Parameters
    ----------
    df : pd.DataFrame
    filepath : str
        Destination file path. Parent directories are created automatically.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  -> Saved: {filepath}")
