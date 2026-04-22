"""Preprocessing utilities for cleaning and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean input data by removing unnecessary columns and imputing missing values."""
    cleaned_df = df.copy()

    if "Surname" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["Surname"])

    numeric_columns = cleaned_df.select_dtypes(include="number").columns
    categorical_columns = cleaned_df.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        if cleaned_df[column].isna().any():
            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in categorical_columns:
        if cleaned_df[column].isna().any():
            mode_values = cleaned_df[column].mode(dropna=True)
            if not mode_values.empty:
                cleaned_df[column] = cleaned_df[column].fillna(mode_values.iloc[0])

    return cleaned_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create segmentation features used for churn analysis."""
    featured_df = df.copy()

    if "Age" in featured_df.columns:
        featured_df["AgeGroup"] = pd.cut(
            featured_df["Age"],
            bins=[float("-inf"), 29, 45, 60, float("inf")],
            labels=["Young", "Mid-Age", "Senior", "Elder"],
        )

    if "CreditScore" in featured_df.columns:
        featured_df["CreditScoreBand"] = pd.cut(
            featured_df["CreditScore"],
            bins=[float("-inf"), 499, 700, float("inf")],
            labels=["Low", "Medium", "High"],
        )

    if "Tenure" in featured_df.columns:
        featured_df["TenureGroup"] = pd.cut(
            featured_df["Tenure"],
            bins=[float("-inf"), 2, 7, float("inf")],
            labels=["New", "Mid", "Long"],
        )

    if "Balance" in featured_df.columns:
        balance = featured_df["Balance"]
        featured_df["BalanceSegment"] = np.select(
            [balance == 0, (balance > 0) & (balance < 50000), balance >= 50000],
            ["Zero", "Low", "High"],
            default=Unknown,
        )
    if {"Balance", "IsActiveMember"}.issubset(featured_df.columns):
        featured_df["Risk"] = np.where(
            (featured_df["Balance"] > 50000) & (featured_df["IsActiveMember"] == 0),
            "High Risk",
            "Normal",
        )

    return featured_df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run complete preprocessing pipeline."""
    cleaned_df = clean_data(df)
    processed_df = create_features(cleaned_df)
    return processed_df
