"""Plotly visualization utilities for churn analytics dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from kpi import calculate_engagement_churn, calculate_segment_churn


def _build_churn_bar_chart(
    data: pd.DataFrame, x_col: str, title: str, x_label: str, y_label: str = "Churn Rate (%)"
) -> Figure:
    """Create a clean bar chart for churn rate by a segment."""
    plot_data = data.sort_values("ChurnRate", ascending=False).copy()

    hover_columns = {}
    if "CustomerCount" in plot_data.columns:
        hover_columns["CustomerCount"] = True

    fig = px.bar(
        plot_data,
        x=x_col,
        y="ChurnRate",
        text="ChurnRate",
        color=x_col,
        hover_data=hover_columns,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    hover_template = (
        f"<b>{x_label}</b>: %{{x}}<br>"
        "<b>Churn Rate</b>: %{y:.2f}%<br>"
        "<b>Customers</b>: %{customdata[0]}<extra></extra>"
    )
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=hover_template if "CustomerCount" in plot_data.columns else None,
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        yaxis_tickformat=".2f",
        template="plotly_dark",
        legend_title_text=x_label,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    return fig


def plot_geography_churn(df: pd.DataFrame) -> Figure:
    """Plot churn rate by geography."""
    churn_data = calculate_segment_churn(df, "Geography")
    return _build_churn_bar_chart(
        churn_data,
        x_col="Geography",
        title="Geography vs Churn Rate",
        x_label="Geography",
    )


def plot_age_churn(df: pd.DataFrame) -> Figure:
    """Plot churn rate by age group."""
    churn_data = calculate_segment_churn(df, "AgeGroup")
    return _build_churn_bar_chart(
        churn_data,
        x_col="AgeGroup",
        title="Age Group vs Churn Rate",
        x_label="Age Group",
    )


def plot_balance_churn(df: pd.DataFrame) -> Figure:
    """Plot churn rate by balance segment."""
    churn_data = calculate_segment_churn(df, "BalanceSegment")
    return _build_churn_bar_chart(
        churn_data,
        x_col="BalanceSegment",
        title="Balance Segment vs Churn Rate",
        x_label="Balance Segment",
    )


def plot_gender_churn(df: pd.DataFrame) -> Figure:
    """Plot churn rate by gender."""
    churn_data = calculate_segment_churn(df, "Gender")
    return _build_churn_bar_chart(
        churn_data,
        x_col="Gender",
        title="Gender vs Churn Rate",
        x_label="Gender",
    )


def plot_engagement_churn(df: pd.DataFrame) -> Figure:
    """Plot churn rate by engagement status (active vs inactive)."""
    churn_data = calculate_engagement_churn(df)
    if "IsActiveMember" in df.columns:
        counts = (
            df["IsActiveMember"]
            .map({1: "Active", 0: "Inactive"})
            .fillna("Unknown")
            .value_counts(dropna=False)
            .rename_axis("EngagementStatus")
            .reset_index(name="CustomerCount")
        )
        churn_data = churn_data.merge(counts, on="EngagementStatus", how="left")

    return _build_churn_bar_chart(
        churn_data,
        x_col="EngagementStatus",
        title="Engagement Status vs Churn Rate",
        x_label="Engagement Status",
    )


def plot_geography_pie(df: pd.DataFrame) -> Figure:
    """Plot geography contribution as a churn-share pie chart."""
    churn_data = calculate_segment_churn(df, "Geography")
    plot_data = churn_data.sort_values("ChurnRate", ascending=False).copy()
    fig = px.pie(
        plot_data,
        names="Geography",
        values="ChurnRate",
        title="Geography Churn Share",
        hover_data=["CustomerCount"] if "CustomerCount" in plot_data.columns else None,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "<b>Geography</b>: %{label}<br>"
            "<b>Churn Share</b>: %{percent}<br>"
            "<b>Churn Rate Proxy</b>: %{value:.2f}%<br>"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
    )
    return fig


def plot_age_trend_line(df: pd.DataFrame) -> Figure:
    """Plot age-group churn as a trend-style line chart."""
    churn_data = calculate_segment_churn(df, "AgeGroup")
    age_order = ["Young", "Mid-Age", "Senior", "Elder"]
    plot_data = churn_data.copy()
    plot_data["AgeGroup"] = pd.Categorical(plot_data["AgeGroup"], categories=age_order, ordered=True)
    plot_data = plot_data.sort_values("AgeGroup")

    fig = px.line(
        plot_data,
        x="AgeGroup",
        y="ChurnRate",
        markers=True,
        title="Age Group Churn Trend",
        hover_data=["CustomerCount"] if "CustomerCount" in plot_data.columns else None,
    )
    fig.update_traces(
        line=dict(color="#60A5FA", width=3),
        marker=dict(size=10),
        hovertemplate=(
            "<b>Age Group</b>: %{x}<br>"
            "<b>Churn Rate</b>: %{y:.2f}%<br>"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Churn Rate (%)",
        yaxis_tickformat=".2f",
        template="plotly_dark",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
    )
    return fig
