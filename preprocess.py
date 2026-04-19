# -*- coding: utf-8 -*-
"""
Data splitting script.

Reads the raw dataset and produces two sets of train/test splits:
  - Spatial split  (merged9_s_train.csv / merged9_s_test.csv):
      80 % / 20 % random split by administrative unit (uid).
  - Temporal split (merged9_t_train.csv / merged9_t_test.csv):
      Training years 2001–2016, test years 2017–2020.

Run this script once before training:
    python preprocess.py
"""

import os
import pandas as pd
import numpy as np

def preprocess(
    data_path: str,
    spatial_dir: str,
    temporal_dir: str,
    seed: int = 42
):
    """
    Apply two splits to the same raw dataset:
      1) Shuffle by uid and split 80%/20% (spatial split)
      2) Split by year: 2001–2016 for training, 2017–2020 for testing (temporal split)
    Save results to the respective output directories.
    """
    # 1. Load the full dataset
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    print(f"[OK] Raw dataset: {len(df)} records, columns: {list(df.columns)}")

    # ========== Spatial split ==========
    # 2. Get all unique uids
    unique_uids = df["uid"].unique()
    # 3. Shuffle uid list and split 80% / 20%
    rng = np.random.default_rng(seed)
    shuffled_uids = rng.permutation(unique_uids)
    n_total = len(shuffled_uids)
    n_train = int(n_total * 0.8)
    train_uids = shuffled_uids[:n_train]
    test_uids  = shuffled_uids[n_train:]
    # 4. Filter rows by uid
    train_spatial = df[df["uid"].isin(train_uids)].reset_index(drop=True)
    test_spatial  = df[df["uid"].isin(test_uids)].reset_index(drop=True)
    # 5. Save
    os.makedirs(spatial_dir, exist_ok=True)
    s_train_path = os.path.join(spatial_dir, "merged9_s_train.csv")
    s_test_path  = os.path.join(spatial_dir, "merged9_s_test.csv")
    train_spatial.to_csv(s_train_path, index=False, encoding="utf-8-sig")
    test_spatial.to_csv(s_test_path,  index=False, encoding="utf-8-sig")
    print(f"[OK] Spatial train: {len(train_uids)} uids -> {s_train_path}")
    print(f"[OK] Spatial test : {len(test_uids)} uids -> {s_test_path}")

    # ========== Temporal split ==========
    # 6. Split into train set (2001–2016) and test set (2017–2020) by year
    train_temporal = df[df["year"].between(2001, 2016)].reset_index(drop=True)
    test_temporal  = df[df["year"].between(2017, 2020)].reset_index(drop=True)
    # 7. Save
    os.makedirs(temporal_dir, exist_ok=True)
    t_train_path = os.path.join(temporal_dir, "merged9_t_train.csv")
    t_test_path  = os.path.join(temporal_dir, "merged9_t_test.csv")
    train_temporal.to_csv(t_train_path, index=False, encoding="utf-8-sig")
    test_temporal.to_csv(t_test_path,  index=False, encoding="utf-8-sig")
    print(f"[OK] Temporal train (2001-2016) -> {t_train_path}")
    print(f"[OK] Temporal test  (2017-2020) -> {t_test_path}")

if __name__ == "__main__":
    _BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH    = os.path.join(_BASE, "data", "raw",       "merged9.5.csv")
    SPATIAL_DIR  = os.path.join(_BASE, "data", "processed")
    TEMPORAL_DIR = os.path.join(_BASE, "data", "processed")

    preprocess(DATA_PATH, SPATIAL_DIR, TEMPORAL_DIR, seed=42)
