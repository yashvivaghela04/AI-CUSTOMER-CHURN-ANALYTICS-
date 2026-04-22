"""Data loading and validation utilities for churn analytics."""

from __future__ import annotations

from io import StringIO
from typing import Iterable

import pandas as pd


DEFAULT_DATA_URL = (
    "https://docs.google.com/spreadsheets/d/12O6M2gvjTdFDOUzWRy7rsyCb4aC1Fi__"
    "LPinkDInucE/export?format=csv&gid=982505339"
)
REQUIRED_COLUMNS = (
    "CustomerId",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
)


def load_data(url: str = DEFAULT_DATA_URL) -> pd.DataFrame:
    """Load raw dataset from a CSV URL."""
    try:
        df = pd.read_csv(url)
    except Exception as exc:
        raise ValueError(f"Failed to load dataset from URL: {url}") from exc

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def validate_binary_columns(
    df: pd.DataFrame, binary_columns: Iterable[str] = ("Exited", "HasCrCard", "IsActiveMember")
) -> None:
    """Ensure binary columns only contain 0/1 values (ignoring missing values)."""
    for column in binary_columns:
        if column not in df.columns:
            continue

        non_null_values = set(df[column].dropna().unique())
        if not non_null_values.issubset({0, 1}):
            raise ValueError(
                f"Column '{column}' contains non-binary values: "
                f"{sorted(non_null_values - {0, 1})}"
            )


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Ensure all required columns exist in the dataset."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_dataset(df: pd.DataFrame) -> None:
    """Run baseline validation checks before preprocessing."""
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("Dataset has no rows or columns.")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        print(f"Warning: Found {duplicate_rows} duplicate rows.")

    validate_required_columns(df)
    validate_binary_columns(df)


def print_preview(df: pd.DataFrame, head_rows: int = 5) -> None:
    """Print dataset head() and info() as requested."""
    print("\nDataset head():")
    print(df.head(head_rows))

    info_buffer = StringIO()
    df.info(buf=info_buffer)
    print("\nDataset info():")
    print(info_buffer.getvalue())


def load_and_validate_data(url: str = DEFAULT_DATA_URL, head_rows: int = 5) -> pd.DataFrame:
    """Load dataset from URL, validate it, and print quick diagnostics."""
    df = load_data(url)
    validate_dataset(df)
    print_preview(df, head_rows=head_rows)
    return df
