# -*- coding: utf-8 -*-
"""
Global configuration for the CO2 emission prediction pipeline.
Modify this file to change data paths, model settings, or feature options.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATA    = os.path.join(BASE_DIR, "data", "raw",       "merged9.5.csv")
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "predict_ds")

# ── Data split scales ─────────────────────────────────────────────────────────
# 's' = spatial split (80/20 by uid), 't' = temporal split (2001-2016 / 2017-2020)
SCALES = ['t']

# ── Target variable ───────────────────────────────────────────────────────────
TARGET_COL = 'co2_ds'

# ── Feature selection ─────────────────────────────────────────────────────────
# Options: 'mi' (mutual information), 'rf' (random forest), 'pearson', None
FEATURE_METHODS = ['mi']
SELECT_K_LIST   = [20]

# Variable whitelist applied before feature selection.
# Set to [] or None to use all columns except the target.
SELECTED_FEATURES = [
    'prov_id', 'uid', 'year', 'dew', 'temp', 'pop',
    'gdp', 'area_crop', 'area_forest', 'area_urban', 'val_ind',
    'export', 'import', 'val_pri', 'val_sec', 'val_ter',
    'inc_urban', 'inc_rural', 'pass_vol', 'cars', 'coal_raw',
    'coal_clean', 'briquettes', 'coke', 'gas_coke', 'oil_crude',
    'gasoline', 'kerosene', 'diesel', 'oil_fuel', 'lpg',
    'gas_ref', 'gas_nat', 'heat', 'elec', 'evap',
    'wind', 'press', 'solar', 'precip', 'pm25',
]

# ── Training ──────────────────────────────────────────────────────────────────
K_FOLDS        = 5
N_BOOTSTRAP    = 20
DO_UNCERTAINTY = False
RANDOM_SEED    = 0
