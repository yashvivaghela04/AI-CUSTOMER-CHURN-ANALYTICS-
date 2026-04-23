"""Streamlit app for interactive customer segmentation and churn analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

from data_loader import load_and_validate_data
from kpi import (
    calculate_engagement_churn,
    calculate_high_value_churn,
    calculate_overall_churn,
    calculate_segment_churn,
)
from preprocessing import preprocess_data

@st.cache_data(show_spinner=False, ttl=0)
def _get_processed_data() -> pd.DataFrame:
    raw_df = load_and_validate_data()
    return preprocess_data(raw_df)

def _format_metric(value: float) -> str:
    """Format KPI values for display."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def _inject_custom_css() -> None:
    """Apply a modern, dark SaaS-style visual theme."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(135deg, #0F172A, #1E293B);
                color: #F1F5F9;
            }
            .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px; }
            .section-title {
                font-size: 1.45rem;
                font-weight: 800;
                margin-top: 1rem;
                margin-bottom: 0.9rem;
                color: #F1F5F9;
                letter-spacing: 0.01em;
            }
            .kpi-card {
                background: #1E293B;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 8px 18px rgba(2, 6, 23, 0.35);
                margin-bottom: 1rem;
            }
            .panel-card {
                background: #1E293B;
                border: 1px solid #1F2937;
                border-radius: 14px;
                padding: 12px 14px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
                margin-bottom: 1rem;
            }
            .kpi-label { font-size: 0.9rem; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.04em; }
            .kpi-value { font-size: 2.0rem; font-weight: 800; color: #F1F5F9; }
            .section-gap { margin-top: 1.3rem; }
            .helper-note {
                color: #38BDF8;
                font-size: 0.86rem;
                padding: 0.45rem 0.7rem;
                margin: 0.2rem 0 0.8rem 0;
                background: rgba(56, 189, 248, 0.08);
                border: 1px solid rgba(56, 189, 248, 0.24);
                border-radius: 10px;
            }
            [data-testid="stPlotlyChart"] {
                background: rgba(30, 41, 59, 0.55);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 14px;
                padding: 6px 8px;
                margin-bottom: 0.8rem;
            }
            [data-testid="stCaptionContainer"] p {
                color: #94A3B8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_card(label: str, value: str) -> None:
    """Render a KPI card with custom style."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_sidebar_selection(
    df: pd.DataFrame,
    column: str,
    key: str,
    label: str,
) -> tuple[list[str], list[str]]:
    """Render one multiselect filter and return options + selected values."""
    if column not in df.columns:
        return [], []

    options = sorted(df[column].dropna().astype(str).unique().tolist())
    if key not in st.session_state:
        st.session_state[key] = options

    selected = st.sidebar.multiselect(
        label=label,
        options=options,
        default=st.session_state[key],
        key=key,
    )
    return options, selected


def _reset_filters() -> None:
    """Reset all sidebar filter selections."""
    for key in [
        "flt_geography",
        "flt_gender",
        "flt_age_group",
        "flt_engagement",
        "flt_balance_segment",
        "flt_credit_band",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def _apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Build grouped sidebar filters and return filtered dataframe."""
    st.sidebar.header("Filters")
    if st.sidebar.button("Reset Filters", use_container_width=True):
        _reset_filters()
        st.rerun()
    st.sidebar.markdown("---")

    filtered_df = df.copy()

    st.sidebar.subheader("Demographics")
    _, sel_gender = _get_sidebar_selection(filtered_df, "Gender", "flt_gender", "Gender")
    _, sel_age = _get_sidebar_selection(filtered_df, "AgeGroup", "flt_age_group", "Age Group")
    engagement_options = ["Active", "Inactive"]
    if "flt_engagement" not in st.session_state:
        st.session_state["flt_engagement"] = engagement_options
    sel_engagement = st.sidebar.multiselect(
        label="Engagement Status",
        options=engagement_options,
        default=st.session_state["flt_engagement"],
        key="flt_engagement",
    )

    st.sidebar.subheader("Geography")
    _, sel_geo = _get_sidebar_selection(filtered_df, "Geography", "flt_geography", "Geography")

    st.sidebar.subheader("Financial")
    _, sel_balance = _get_sidebar_selection(
        filtered_df, "BalanceSegment", "flt_balance_segment", "Balance Segment"
    )
    _, sel_credit = _get_sidebar_selection(
        filtered_df, "CreditScoreBand", "flt_credit_band", "Credit Score Band"
    )

    filters = [
        ("Gender", sel_gender),
        ("AgeGroup", sel_age),
        ("Geography", sel_geo),
        ("BalanceSegment", sel_balance),
        ("CreditScoreBand", sel_credit),
    ]
    for column, selected_values in filters:
        if column in filtered_df.columns and selected_values:
            filtered_df = filtered_df[filtered_df[column].astype(str).isin(selected_values)]

    if "IsActiveMember" in filtered_df.columns and sel_engagement:
        engagement_map = {1: "Active", 0: "Inactive"}
        filtered_df = filtered_df[
            filtered_df["IsActiveMember"].map(engagement_map).isin(sel_engagement)
        ]

    return filtered_df


def _render_filter_feedback(base_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Show dynamic feedback for selected filters."""
    total_count = len(base_df)
    filtered_count = len(filtered_df)
    kept_pct = (filtered_count / total_count * 100) if total_count else 0.0
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Impact")
    st.sidebar.caption(f"Records: **{filtered_count:,} / {total_count:,}** ({kept_pct:.1f}%)")

    base_churn = calculate_overall_churn(base_df)
    filtered_churn = calculate_overall_churn(filtered_df)
    if not pd.isna(base_churn) and not pd.isna(filtered_churn):
        delta = filtered_churn - base_churn
        direction = "higher" if delta > 0 else "lower"
        st.sidebar.caption(f"Churn is **{abs(delta):.2f}% {direction}** than baseline.")


def _render_key_insights(filtered_df: pd.DataFrame) -> None:
    """Render compact key insights."""
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    top_geo = calculate_segment_churn(filtered_df, "Geography").head(1)
    top_age = calculate_segment_churn(filtered_df, "AgeGroup").head(1)
    engagement_churn = calculate_engagement_churn(filtered_df).set_index("EngagementStatus")

    if not top_geo.empty:
        st.info(
            f"Highest churn geography: **{top_geo.iloc[0]['Geography']}** at "
            f"**{_format_metric(float(top_geo.iloc[0]['ChurnRate']))}**."
        )
    if not top_age.empty:
        st.info(
            f"Highest churn age group: **{top_age.iloc[0]['AgeGroup']}** at "
            f"**{_format_metric(float(top_age.iloc[0]['ChurnRate']))}**."
        )
    if {"Active", "Inactive"}.issubset(engagement_churn.index):
        active_rate = float(engagement_churn.loc["Active", "ChurnRate"])
        inactive_rate = float(engagement_churn.loc["Inactive", "ChurnRate"])
        st.info(
            f"Inactive churn (**{_format_metric(inactive_rate)}**) exceeds active churn "
            f"(**{_format_metric(active_rate)}**) by **{_format_metric(inactive_rate - active_rate)}**."
        )


def _build_overview_churn_chart(
    segment_df: pd.DataFrame,
    x_col: str,
    title: str,
    height: int = 330,
) -> Figure:
    """Create a styled churn bar chart with highest segment highlight."""
    chart_df = segment_df.copy().sort_values("ChurnRate", ascending=False)
    if chart_df["CustomerCount"].isna().any():
        chart_df["CustomerCount"] = chart_df["CustomerCount"].fillna(0)
    highest_idx = chart_df["ChurnRate"].idxmax()
    chart_df["BarColor"] = "#6366F1"
    chart_df.loc[highest_idx, "BarColor"] = "#EF4444"

    fig = px.bar(
        chart_df,
        x=x_col,
        y="ChurnRate",
        color="BarColor",
        color_discrete_map="identity",
        text="ChurnRate",
        custom_data=["CustomerCount"],
        title=title,
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Churn: %{y:.2f}%<br>"
            "Customers: %{customdata[0]}<extra></extra>"
        ),
        marker_line_width=1.2,
        marker_line_color="#334155",
        hoverlabel=dict(font_size=12),
    )
    highest_row = chart_df.loc[highest_idx]
    fig.add_annotation(
        x=highest_row[x_col],
        y=float(highest_row["ChurnRate"]),
        text=f"Highest churn: {float(highest_row['ChurnRate']):.2f}%",
        showarrow=True,
        arrowhead=2,
        ax=35,
        ay=-35,
        font=dict(size=11, color="#EF4444"),
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(size=12, color="#E2E8F0"),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title=x_col,
        yaxis_title="Churn Rate (%)",
        yaxis_gridcolor="rgba(148, 163, 184, 0.12)",
        xaxis_gridcolor="rgba(148, 163, 184, 0.05)",
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#6366F1", font_size=12),
    )
    return fig


def _style_plotly_figure(fig: Figure, height: int = 320) -> Figure:
    """Apply consistent palette and readability settings across charts."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(size=12, color="#F1F5F9"),
        colorway=["#6366F1", "#38BDF8", "#22C55E", "#EF4444"],
        yaxis_gridcolor="rgba(148, 163, 184, 0.12)",
        xaxis_gridcolor="rgba(148, 163, 184, 0.05)",
        margin=dict(l=20, r=20, t=60, b=20),
        hoverlabel=dict(bgcolor="#1E293B", bordercolor="#38BDF8", font_size=12),
        height=height,
    )
    return fig


def _render_overview_tab(filtered_df: pd.DataFrame) -> None:
    """Overview with KPI cards, insights, and three core charts."""
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
    st.caption("ℹ️ KPI cards update instantly with filters and chart selections.")
    kpi_col_1, kpi_col_2 = st.columns(2)
    with kpi_col_1:
        _render_kpi_card("Overall Churn Rate", _format_metric(calculate_overall_churn(filtered_df)))
    with kpi_col_2:
        _render_kpi_card("High-Value Churn Rate", _format_metric(calculate_high_value_churn(filtered_df)))

    if "Risk" in filtered_df.columns:
        high_risk_pct = float((filtered_df["Risk"] == "High Risk").mean() * 100)
        st.warning(f"High-risk customer share: {_format_metric(high_risk_pct)}")

    _render_key_insights(filtered_df)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Core Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-note">ℹ️ Hover over chart elements to view churn rate and customer counts.</div>',
        unsafe_allow_html=True,
    )

    geography_churn = calculate_segment_churn(filtered_df, "Geography")
    age_churn = calculate_segment_churn(filtered_df, "AgeGroup")
    engagement_churn = calculate_engagement_churn(filtered_df)
    if (
        "EngagementStatus" in engagement_churn.columns
        and "CustomerCount" not in engagement_churn.columns
        and "IsActiveMember" in filtered_df.columns
    ):
        engagement_counts = (
            filtered_df["IsActiveMember"]
            .map({1: "Active", 0: "Inactive"})
            .value_counts()
            .rename_axis("EngagementStatus")
            .reset_index(name="CustomerCount")
        )
        engagement_churn = engagement_churn.merge(engagement_counts, on="EngagementStatus", how="left")

    visual_col_1, visual_col_2 = st.columns(2)
    with visual_col_1:
        geo_fig = _build_overview_churn_chart(geography_churn, "Geography", "Geography vs Churn", height=330)
        st.plotly_chart(geo_fig, use_container_width=True)
    with visual_col_2:
        age_fig = _build_overview_churn_chart(age_churn, "AgeGroup", "Age Group vs Churn", height=330)
        st.plotly_chart(age_fig, use_container_width=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    engagement_fig = _build_overview_churn_chart(
        engagement_churn, "EngagementStatus", "Engagement vs Churn", height=285
    )
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.plotly_chart(engagement_fig, use_container_width=True)


def _prepare_aggregated_data(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str | None,
    color_axis: str | None,
    agg: str,
) -> pd.DataFrame:
    """Aggregate data for dynamic chart generation."""
    group_cols = [x_axis] + ([color_axis] if color_axis and color_axis != x_axis else [])

    if agg == "count" or y_axis is None:
        grouped = df.groupby(group_cols, dropna=False).size().reset_index(name="Value")
        y_label = "Count"
    else:
        grouped = df.groupby(group_cols, dropna=False)[y_axis].mean().reset_index(name="Value")
        if y_axis == "Exited":
            grouped["Value"] = grouped["Value"] * 100
            y_label = "Exited Mean (%)"
        else:
            y_label = f"{y_axis} Mean"

    return grouped, y_label


def _render_visual_builder_tab(filtered_df: pd.DataFrame) -> None:
    """Full interactive visualization builder."""
    st.markdown('<div class="section-title">Visualization Builder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><b>Build custom visuals</b><br>'
        "Select dimensions, chart type, and aggregation to explore churn and segments interactively."
        "</div>",
        unsafe_allow_html=True,
    )

    categorical_cols = [
        c for c in filtered_df.columns if filtered_df[c].dtype == "object" or "Group" in c or "Band" in c
    ]
    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_axis = st.selectbox("X-axis *", options=categorical_cols, key="vb_x")
    with c2:
        y_options = ["(auto: count)"] + numeric_cols
        y_choice = st.selectbox("Y-axis (optional)", options=y_options, key="vb_y")
        y_axis = None if y_choice == "(auto: count)" else y_choice
    with c3:
        color_options = ["None"] + categorical_cols
        color_choice = st.selectbox("Color Grouping (optional)", options=color_options, key="vb_color")
        color_axis = None if color_choice == "None" else color_choice
    with c4:
        agg = st.selectbox("Aggregation", options=["mean", "count"], key="vb_agg")

    chart_type = st.selectbox(
        "Chart Type",
        options=["Bar", "Grouped Bar", "Pie", "Line", "Scatter", "Box Plot", "Histogram"],
        key="vb_chart",
    )

    chart_df, y_label = _prepare_aggregated_data(filtered_df, x_axis, y_axis, color_axis, agg)
    if chart_df.empty:
        st.warning("No data available for the selected chart settings.")
        return

    if chart_type == "Bar":
        fig = px.bar(chart_df, x=x_axis, y="Value", color=color_axis, hover_data={"Value": ":.2f"})
    elif chart_type == "Grouped Bar":
        fig = px.bar(chart_df, x=x_axis, y="Value", color=color_axis, barmode="group", hover_data={"Value": ":.2f"})
    elif chart_type == "Pie":
        pie_color = color_axis if color_axis else x_axis
        pie_df = chart_df.groupby(pie_color, dropna=False)["Value"].sum().reset_index()
        fig = px.pie(pie_df, names=pie_color, values="Value")
    elif chart_type == "Line":
        fig = px.line(chart_df, x=x_axis, y="Value", color=color_axis, markers=True, hover_data={"Value": ":.2f"})
    elif chart_type == "Scatter":
        scatter_y = y_axis if y_axis else "Exited"
        fig = px.scatter(filtered_df, x=x_axis, y=scatter_y, color=color_axis, hover_data=filtered_df.columns[:6])
    elif chart_type == "Box Plot":
        box_y = y_axis if y_axis else "Balance"
        fig = px.box(filtered_df, x=x_axis, y=box_y, color=color_axis)
    else:
        hist_x = y_axis if y_axis else "Balance"
        fig = px.histogram(filtered_df, x=hist_x, color=color_axis, marginal="box")

    fig.update_layout(
        title=f"{chart_type}: {x_axis}",
        yaxis_title=y_label if chart_type in {"Bar", "Grouped Bar", "Line"} else None,
    )
    fig = _style_plotly_figure(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


def _render_advanced_analysis_tab(filtered_df: pd.DataFrame) -> None:
    """Dynamic multi-dimension analysis with configurable axes."""
    st.markdown('<div class="section-title">Advanced Analysis</div>', unsafe_allow_html=True)
    categorical_cols = [
        c for c in filtered_df.columns if filtered_df[c].dtype == "object" or "Group" in c or "Band" in c
    ]
    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_axis = st.selectbox("X-axis", options=categorical_cols, key="adv_x")
    with c2:
        color_axis = st.selectbox("Group/Color", options=["None"] + categorical_cols, key="adv_color")
    with c3:
        facet_axis = st.selectbox("Facet (optional)", options=["None"] + categorical_cols, key="adv_facet")
    with c4:
        chart_type = st.selectbox(
            "Chart Type",
            options=["Grouped bar", "Stacked bar", "100% stacked", "Line", "Heatmap", "Scatter", "Bubble"],
            key="adv_chart",
        )

    color_col = None if color_axis == "None" else color_axis
    facet_col = None if facet_axis == "None" else facet_axis
    if color_col == x_axis:
        color_col = None
    if facet_col in {x_axis, color_col}:
        st.info("Facet should be different from X-axis and Group/Color. Please select another facet.")
        return

    group_cols = [x_axis]
    if color_col:
        group_cols.append(color_col)
    if facet_col:
        group_cols.append(facet_col)

    if chart_type in {"Grouped bar", "Stacked bar", "100% stacked", "Line"}:
        grouped = (
            filtered_df.groupby(group_cols, dropna=False)["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Value")
        )
        if chart_type == "Line":
            fig = px.line(
                grouped,
                x=x_axis,
                y="Value",
                color=color_col,
                facet_col=facet_col,
                markers=True,
                title=f"Line Churn Trend by {x_axis}",
                hover_data={"Value": ":.2f"},
            )
        else:
            barmode = "group" if chart_type == "Grouped bar" else "stack"
            fig = px.bar(
                grouped,
                x=x_axis,
                y="Value",
                color=color_col,
                facet_col=facet_col,
                barmode=barmode,
                title=f"{chart_type} Churn by {x_axis}",
                hover_data={"Value": ":.2f"},
            )
            if chart_type == "100% stacked":
                fig.update_layout(barnorm="percent")
                fig.update_yaxes(title="Percent Share")
            else:
                fig.update_yaxes(title="Churn Rate (%)")
    elif chart_type == "Heatmap":
        if color_col is None:
            st.info("Select a Group/Color dimension to render heatmap.")
            return
        grouped = (
            filtered_df.groupby(group_cols, dropna=False)["Exited"]
            .mean()
            .mul(100)
            .reset_index(name="Value")
        )
        fig = px.density_heatmap(
            grouped,
            x=x_axis,
            y=color_col,
            z="Value",
            facet_col=facet_col,
            color_continuous_scale="Blues",
            title=f"Heatmap: {x_axis} x {color_col}",
            hover_data={"Value": ":.2f"},
        )
    elif chart_type == "Bubble":
        y_axis = st.selectbox("Bubble Y-axis", options=numeric_cols, key="adv_bubble_y")
        bubble_size = st.selectbox("Bubble Size", options=numeric_cols, key="adv_bubble_size")
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_col,
            size=bubble_size,
            facet_col=facet_col,
            size_max=40,
            title=f"Bubble: {x_axis} vs {y_axis}",
            hover_data=["Geography", "AgeGroup", "BalanceSegment", "Risk"] if "Risk" in filtered_df.columns else None,
        )
    else:  # Scatter
        y_axis = st.selectbox("Scatter Y-axis", options=numeric_cols, key="adv_scatter_y")
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_col,
            facet_col=facet_col,
            title=f"Scatter: {x_axis} vs {y_axis}",
            hover_data=["Geography", "AgeGroup", "BalanceSegment", "Risk"] if "Risk" in filtered_df.columns else None,
        )

    fig = _style_plotly_figure(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)


def _render_story_tab(filtered_df: pd.DataFrame) -> None:
    """Short dynamic storytelling with findings, risk, and recommendations."""
    st.markdown('<div class="section-title">Story</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-card"><b>Executive Story</b><br>'
        "Top findings, risk signal, and one clear recommended action."
        "</div>",
        unsafe_allow_html=True,
    )

    top_geo = calculate_segment_churn(filtered_df, "Geography").head(1)
    top_age = calculate_segment_churn(filtered_df, "AgeGroup").head(1)
    engagement = calculate_engagement_churn(filtered_df).set_index("EngagementStatus")

    if not top_geo.empty:
        st.success(
            f"Top finding 1: **{top_geo.iloc[0]['Geography']}** leads churn at "
            f"**{_format_metric(float(top_geo.iloc[0]['ChurnRate']))}**."
        )
    if not top_age.empty:
        st.success(
            f"Top finding 2: **{top_age.iloc[0]['AgeGroup']}** is the highest-risk age cohort "
            f"at **{_format_metric(float(top_age.iloc[0]['ChurnRate']))}** churn."
        )
    if {"Active", "Inactive"}.issubset(engagement.index):
        active_rate = float(engagement.loc["Active", "ChurnRate"])
        inactive_rate = float(engagement.loc["Inactive", "ChurnRate"])
        st.success(
            f"Top finding 3: inactive members churn **{_format_metric(inactive_rate - active_rate)}** more than active members."
        )

    st.markdown("")
    if "Risk" in filtered_df.columns:
        high_risk_pct = float((filtered_df["Risk"] == "High Risk").mean() * 100)
        st.warning(f"Risk summary: **{_format_metric(high_risk_pct)}** of customers are currently high risk.")

    st.markdown("")
    st.info(
        "Recommendation: prioritize retention campaigns for high-risk inactive customers in top-churn geographies, "
        "then monitor uplift in active membership and product engagement."
    )


def main() -> None:
    """Run Streamlit app."""
    st.set_page_config(page_title="Customer Segmentation & Churn Analytics", layout="wide")
    _inject_custom_css()
    st.title("📊 Interactive Customer Segmentation & Churn Analytics Dashboard")

    processed_df = _get_processed_data()

    # 👇 ADD THIS LINE
    st.write("DEBUG - Columns:", processed_df.columns)

    filtered_df = _apply_sidebar_filters(processed_df)
    _render_filter_feedback(processed_df, filtered_df)

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your filter choices.")
        return

    tab_overview, tab_builder, tab_advanced, tab_story = st.tabs(
        ["Overview", "Visualization Builder", "Advanced Analysis", "Story"]
    )
    with tab_overview:
        _render_overview_tab(filtered_df)
    with tab_builder:
        _render_visual_builder_tab(filtered_df)
    with tab_advanced:
        _render_advanced_analysis_tab(filtered_df)
    with tab_story:
        _render_story_tab(filtered_df)


if __name__ == "__main__":
    main()
