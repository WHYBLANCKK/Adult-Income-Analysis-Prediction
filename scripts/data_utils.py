"""
data_utils.py - Reusable utilities for loading and preparing the Adult Census Income dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

NUMERIC_COLUMNS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
TARGET_COLUMN = 'income'


def get_project_root() -> Path:
    """Returns the project root directory reliably whether called from root, notebooks/, or scripts/."""
    current = Path.cwd().resolve()
    for p in [current, current.parent, current.parent.parent]:
        if (p / 'warehouse').exists() and (p / 'notebooks').exists():
            return p
    return current


def find_data_file(filename: str) -> Path:
    """Locates a dataset file in data/raw/ or adult/ directory."""
    root = get_project_root()
    candidates = [
        root / 'adult' / filename,
        root / 'data' / 'raw' / filename,
    ]
    for c in candidates:
        if c.exists() and not (c.is_symlink() and not c.resolve().exists()):
            return c
    raise FileNotFoundError(
        f"Dataset file '{filename}' not found. Please place it in either 'data/raw/' or 'adult/'. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def load_raw_data(data_path: str = None, test_path: str = None):
    """
    Loads adult.data and adult.test cleanly.
    - Resolves from adult/ or data/raw/
    - Handles leading spaces
    - Replaces '?' with NaN
    - Skips comment header in adult.test
    - Normalizes target labels by stripping trailing dots
    """
    if data_path is None:
        data_path = find_data_file('adult.data')
    if test_path is None:
        test_path = find_data_file('adult.test')

    train_df = pd.read_csv(
        data_path,
        names=COLUMN_NAMES,
        sep=',',
        skipinitialspace=True,
        na_values='?',
        engine='python'
    )

    test_df = pd.read_csv(
        test_path,
        names=COLUMN_NAMES,
        sep=',',
        skipinitialspace=True,
        skiprows=1,
        na_values='?',
        engine='python'
    )

    # Normalize target column in test set (strip trailing period)
    test_df['income'] = test_df['income'].astype(str).str.rstrip('.')

    return train_df, test_df


def load_combined_data():
    """Loads and concatenates train and test into a single unified dataframe with a split indicator."""
    train_df, test_df = load_raw_data()
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    return combined_df

