"""
test_data_integrity.py - Validates dataset dimensions, hygiene, and missing values.
"""

import pytest
import pandas as pd
from scripts.data_utils import load_raw_data, COLUMN_NAMES, NUMERIC_COLUMNS

def test_raw_data_dimensions():
    train_df, test_df = load_raw_data()
    assert len(train_df) == 32561, f"Expected 32,561 training rows, got {len(train_df)}"
    assert len(test_df) == 16281, f"Expected 16,281 testing rows, got {len(test_df)}"
    assert train_df.shape[1] == 15, "Expected 15 columns in train"
    assert test_df.shape[1] == 15, "Expected 15 columns in test"
    assert len(train_df) + len(test_df) == 48842, "Expected 48,842 total rows before cleaning"

def test_target_label_hygiene():
    train_df, test_df = load_raw_data()
    train_targets = set(train_df['income'].unique())
    test_targets = set(test_df['income'].unique())
    assert train_targets == {'<=50K', '>50K'}, f"Unexpected train target labels: {train_targets}"
    assert test_targets == {'<=50K', '>50K'}, f"Unexpected test target labels: {test_targets}"

def test_missing_value_markers():
    train_df, test_df = load_raw_data()
    # Missing values should be NaNs in workclass, occupation, native-country
    assert train_df['workclass'].isna().sum() == 1836
    assert train_df['occupation'].isna().sum() == 1843
    assert train_df['native-country'].isna().sum() == 583
    assert test_df['workclass'].isna().sum() == 963
    assert test_df['occupation'].isna().sum() == 966
    assert test_df['native-country'].isna().sum() == 274
