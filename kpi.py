"""KPI calculation utilities for churn and segmentation analytics."""

from __future__ import annotations

import pandas as pd


def _churn_rate_percent(series: pd.Series) -> float:
    """Calculate churn rate percentage from a binary churn series."""
    if series.empty:
        return float("nan")
    return float(series.eq(1).mean() * 100)


def calculate_overall_churn(df: pd.DataFrame, churn_column: str = "Exited") -> float:
    """Return overall churn rate as a percentage."""
    if churn_column not in df.columns:
        raise ValueError(f"Missing churn column: {churn_column}")
    return _churn_rate_percent(df[churn_column])


def calculate_segment_churn(
    df: pd.DataFrame, segment_column: str, churn_column: str = "Exited"
) -> pd.DataFrame:
    """Calculate churn rate percentage by a segment column."""
    if segment_column not in df.columns:
        raise ValueError(f"Missing segment column: {segment_column}")
    if churn_column not in df.columns:
        raise ValueError(f"Missing churn column: {churn_column}")

    grouped = df.groupby(segment_column, dropna=False)[churn_column]
    segment_churn = grouped.agg(
        ChurnRate=_churn_rate_percent,
        CustomerCount="size",
    ).reset_index()
    segment_churn = segment_churn.sort_values("ChurnRate", ascending=False, na_position="last")
    return segment_churn


def calculate_high_value_churn(
    df: pd.DataFrame,
    balance_column: str = "Balance",
    churn_column: str = "Exited",
    high_value_threshold: float = 50000,
) -> float:
    """Return churn rate percentage for high-value customers."""
    if balance_column not in df.columns:
        raise ValueError(f"Missing balance column: {balance_column}")
    if churn_column not in df.columns:
        raise ValueError(f"Missing churn column: {churn_column}")

    high_value_customers = df[df[balance_column] >= high_value_threshold]
    if high_value_customers.empty:
        return float("nan")
    return _churn_rate_percent(high_value_customers[churn_column])


def calculate_engagement_churn(
    df: pd.DataFrame, engagement_column: str = "IsActiveMember", churn_column: str = "Exited"
) -> pd.DataFrame:
    """Compare churn rates for active versus inactive members."""
    if engagement_column not in df.columns:
        raise ValueError(f"Missing engagement column: {engagement_column}")
    if churn_column not in df.columns:
        raise ValueError(f"Missing churn column: {churn_column}")

    engagement_map = {1: "Active", 0: "Inactive"}
    engagement_df = df.copy()
    engagement_df["EngagementStatus"] = engagement_df[engagement_column].map(engagement_map)
    engagement_df["EngagementStatus"] = engagement_df["EngagementStatus"].fillna("Unknown")

    comparison = (
        engagement_df.groupby("EngagementStatus", dropna=False)[churn_column]
        .apply(_churn_rate_percent)
        .reset_index(name="ChurnRate")
    )
    return comparison
