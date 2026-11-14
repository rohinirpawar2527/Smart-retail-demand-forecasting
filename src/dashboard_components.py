"""
Dashboard Components & Professional SaaS Visualizations for Smart Retail Demand Forecasting System.

Provides reusable UI components, SaaS CSS styling, a centralized Plotly theme engine,
Plotly chart builders, inventory risk calculators, and cached data access functions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# SAAS BRAND PALETTE & COLOR TOKENS
# ==============================================================================
PRIMARY_NAVY = "#0F172A"       # Slate 900
SECONDARY_NAVY = "#1E293B"     # Slate 800
ACCENT_TEAL = "#0D9488"        # Teal 600
LIGHT_TEAL = "#14B8A6"         # Teal 500
BRAND_BLUE = "#2563EB"         # Blue 600
BG_LIGHT = "#F8FAFC"           # Slate 50
CARD_BORDER = "#E2E8F0"        # Slate 200
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#64748B"         # Slate 500
SUCCESS_GREEN = "#10B981"      # Emerald 500
WARNING_AMBER = "#F59E0B"      # Amber 500
DANGER_RED = "#EF4444"         # Red 500

# Consistent Category Color Palette across the entire application
CATEGORY_COLOR_MAP: Dict[str, str] = {
    "Grocery": "#0D9488",        # Teal
    "Dairy & Bakery": "#2563EB", # Royal Blue
    "Beverages": "#0F172A",      # Deep Navy
    "Personal Care": "#8B5CF6",  # Purple
    "Home & Kitchen": "#F59E0B", # Amber
}

# Standardized clean Plotly modebar configuration (hides unnecessary clutter)
PLOTLY_CONFIG: Dict[str, Any] = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


# ==============================================================================
# SAAS CSS INJECTION
# ==============================================================================
def apply_custom_css():
    """Inject premium SaaS styling into Streamlit with refined spacing and card hierarchy."""
    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {TEXT_DARK};
        }}
        
        .stApp {{
            background-color: {BG_LIGHT};
        }}
        
        /* Modern Header styling */
        .dashboard-header {{
            background: linear-gradient(135deg, {PRIMARY_NAVY} 0%, {SECONDARY_NAVY} 100%);
            padding: 24px 32px;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .dashboard-header h1 {{
            font-size: 24px;
            font-weight: 700;
            color: #FFFFFF !important;
            margin: 0;
            padding: 0;
            letter-spacing: -0.02em;
        }}
        .dashboard-header p {{
            font-size: 13.5px;
            color: #94A3B8;
            margin: 4px 0 0 0;
        }}
        
        /* KPI Cards */
        .kpi-card {{
            background: #FFFFFF;
            border: 1px solid {CARD_BORDER};
            border-radius: 12px;
            padding: 20px 22px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            position: relative;
            overflow: hidden;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 14px -2px rgba(0, 0, 0, 0.08);
            border-color: {LIGHT_TEAL};
        }}
        .kpi-title {{
            font-size: 12.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_MUTED};
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            color: {PRIMARY_NAVY};
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 8px;
        }}
        .kpi-subtext {{
            font-size: 12px;
            font-weight: 500;
        }}
        .badge-positive {{
            color: {SUCCESS_GREEN};
            background-color: #ECFDF5;
            padding: 3px 9px;
            border-radius: 12px;
            display: inline-block;
        }}
        .badge-neutral {{
            color: {BRAND_BLUE};
            background-color: #EFF6FF;
            padding: 3px 9px;
            border-radius: 12px;
            display: inline-block;
        }}
        .badge-warning {{
            color: {WARNING_AMBER};
            background-color: #FFFBEB;
            padding: 3px 9px;
            border-radius: 12px;
            display: inline-block;
        }}
        .badge-danger {{
            color: {DANGER_RED};
            background-color: #FEF2F2;
            padding: 3px 9px;
            border-radius: 12px;
            display: inline-block;
        }}

        /* Section Cards & Visual Containers */
        .chart-container-card {{
            background: #FFFFFF;
            border: 1px solid {CARD_BORDER};
            border-radius: 12px;
            padding: 16px 20px 8px 20px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        }}

        /* Sidebar customizations */
        section[data-testid="stSidebar"] {{
            background-color: #FFFFFF !important;
            border-right: 1px solid {CARD_BORDER};
        }}
        
        /* Modern Buttons */
        .stButton>button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .stButton>button:hover {{
            border-color: {ACCENT_TEAL};
            color: {ACCENT_TEAL};
        }}
        
        /* Dataframes & Tables */
        .stDataFrame {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid {CARD_BORDER};
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_kpi_card(
    title: str,
    value: str,
    subtext: str = "",
    badge_type: str = "neutral",
    icon: str = "📊",
):
    """Render a premium HTML KPI card."""
    badge_class = f"badge-{badge_type}"
    subtext_html = f'<div class="kpi-subtext"><span class="{badge_class}">{subtext}</span></div>' if subtext else ""
    html = f"""
    <div class="kpi-card">
        <div class="kpi-title">
            <span>{title}</span>
            <span>{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & ARTIFACT LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_default_dataset(file_path: str = "data/processed_retail_data.csv") -> pd.DataFrame:
    """Load and cache preprocessed retail sales dataset."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path: str = "models/best_forecasting_model.joblib") -> Dict[str, Any]:
    """Load and cache the trained ML model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path.resolve()}")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_comparison_results(csv_path: str = "models/model_comparison_results.csv") -> Optional[pd.DataFrame]:
    """Load model evaluation comparison results."""
    path = Path(csv_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_test_predictions(csv_path: str = "models/test_predictions.csv") -> Optional[pd.DataFrame]:
    """Load out-of-sample test predictions for model evaluation."""
    path = Path(csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ==============================================================================
# REUSABLE PLOTLY STYLING FUNCTION (CENTRAL THEME ENGINE)
# ==============================================================================
def apply_saas_chart_theme(
    fig: go.Figure,
    title: str = "",
    subtitle: str = "",
    height: int = 440,
    show_legend: bool = True,
    legend_position: str = "top",  # "top" (below title) | "bottom" | "none"
    x_title: str = "",
    y_title: str = "",
    is_date_axis: bool = False,
    hovermode: str = "closest",
) -> go.Figure:
    """Reusable Plotly theme engine ensuring consistent typography, spacing, and zero overlap.

    Key Features:
    - Consistent clean white card backgrounds.
    - Title placed cleanly at the top-left with optional descriptive subtitle.
    - Legend placed in a dedicated row BELOW the title with generous top margin (90-95px)
      to eliminate title/legend collisions on any screen size.
    - Subtle horizontal & vertical gridlines (#F1F5F9).
    - Readable axis labels and ticks with proper date formatting.
    - Standard height of 400-460px to prevent crowding.
    """
    # 1. Format Title + Subtitle
    if subtitle:
        title_html = (
            f"<span style='font-size:15px; font-weight:700; color:{PRIMARY_NAVY}; letter-spacing:-0.01em;'>"
            f"{title}</span><br>"
            f"<span style='font-size:12px; font-weight:400; color:{TEXT_MUTED};'>{subtitle}</span>"
        )
    elif title:
        title_html = (
            f"<span style='font-size:15px; font-weight:700; color:{PRIMARY_NAVY}; letter-spacing:-0.01em;'>"
            f"{title}</span>"
        )
    else:
        title_html = ""

    title_config = (
        dict(
            text=title_html,
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            pad=dict(b=8),
        )
        if title_html
        else None
    )

    # 2. Margins and Legend Positioning (Prevents all overlapping)
    if show_legend and legend_position == "top":
        # Generous top margin guarantees space for title + legend row
        top_margin = 92 if subtitle else 82
        bottom_margin = 48
        legend_config = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(family="Inter, sans-serif", size=11, color=TEXT_MUTED),
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="rgba(226, 232, 240, 0.7)",
            borderwidth=1,
        )
    elif show_legend and legend_position == "bottom":
        top_margin = 72 if subtitle else 62
        bottom_margin = 68
        legend_config = dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter, sans-serif", size=11, color=TEXT_MUTED),
        )
    else:
        top_margin = 72 if subtitle else 62
        bottom_margin = 48
        legend_config = None

    # 3. Apply Unified Layout
    fig.update_layout(
        title=title_config,
        template="plotly_white",
        height=height,
        hovermode=hovermode,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=top_margin, b=bottom_margin, l=60, r=25),
        showlegend=show_legend if legend_position != "none" else False,
        legend=legend_config if (show_legend and legend_position != "none") else None,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color=TEXT_MUTED, family="Inter, sans-serif")),
            tickfont=dict(size=11, color=TEXT_MUTED, family="Inter, sans-serif"),
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            linecolor="#E2E8F0",
            linewidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color=TEXT_MUTED, family="Inter, sans-serif")),
            tickfont=dict(size=11, color=TEXT_MUTED, family="Inter, sans-serif"),
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            linecolor="#E2E8F0",
            linewidth=1,
            zeroline=False,
        ),
    )

    # 4. Date formatting on X-axis if applicable
    if is_date_axis:
        fig.update_xaxes(
            tickformat="%b %d, %Y",
            nticks=10,
            tickangle=-20,
            showgrid=True,
            gridcolor="#F1F5F9",
        )

    return fig


# ==============================================================================
# PLOTLY CHART BUILDERS (PAGE-BY-PAGE)
# ==============================================================================

# --- 1. OVERVIEW CHARTS ---

def create_daily_sales_chart(df: pd.DataFrame) -> go.Figure:
    """1. Full-width Daily Sales Trend chart with 7-day and 30-day moving averages."""
    daily = df.groupby("Date")["Units_Sold"].sum().reset_index()
    daily["MA7"] = daily["Units_Sold"].rolling(7, min_periods=1).mean()
    daily["MA30"] = daily["Units_Sold"].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["Units_Sold"],
        mode="lines",
        name="Daily Demand",
        line=dict(color="#CBD5E1", width=1.1),
        opacity=0.75,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Daily Sales: %{y:,} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["MA7"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color=BRAND_BLUE, width=2.2),
        hovertemplate="<b>7-Day MA:</b> %{y:,.1f} units<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["Date"],
        y=daily["MA30"],
        mode="lines",
        name="30-Day Trendline",
        line=dict(color=ACCENT_TEAL, width=2.4, dash="dot"),
        hovertemplate="<b>30-Day Trend:</b> %{y:,.1f} units<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title="System-Wide Daily Demand Trend (2024 - 2025)",
        subtitle="Total sales volume aggregated across 5 stores and 20 products with smoothing trends",
        height=450,
        show_legend=True,
        legend_position="top",
        x_title="Timeline",
        y_title="Total Units Sold",
        is_date_axis=True,
        hovermode="x unified",
    )
    return fig


def create_category_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Category demand distribution donut chart with standardized colors."""
    cat_df = df.groupby("Category")["Units_Sold"].sum().reset_index()
    cat_df = cat_df.sort_values("Units_Sold", ascending=False)

    colors = [CATEGORY_COLOR_MAP.get(c, BRAND_BLUE) for c in cat_df["Category"]]

    fig = go.Figure(data=[go.Pie(
        labels=cat_df["Category"],
        values=cat_df["Units_Sold"],
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter, sans-serif", size=11),
        hovertemplate="<b>%{label}</b><br>Total Units Sold: %{value:,}<br>Share: %{percent}<extra></extra>",
    )])

    apply_saas_chart_theme(
        fig,
        title="Demand Share by Product Category",
        subtitle="Volume contribution across all 5 retail departments",
        height=440,
        show_legend=True,
        legend_position="bottom",
    )
    return fig


def create_top_products_bar_chart(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Top 10 Best-Selling Products: horizontal bar chart with separate category traces and zero overlap."""
    top_df = (
        df.groupby(["Product_ID", "Category"])["Units_Sold"]
        .sum()
        .reset_index()
        .sort_values("Units_Sold", ascending=True)
        .tail(n)
    )

    fig = go.Figure()
    # Add traces by category to enable clean, consistent category colors in the legend
    for category in sorted(top_df["Category"].unique()):
        cat_slice = top_df[top_df["Category"] == category]
        fig.add_trace(go.Bar(
            y=cat_slice["Product_ID"],
            x=cat_slice["Units_Sold"],
            name=category,
            orientation="h",
            marker=dict(
                color=CATEGORY_COLOR_MAP.get(category, BRAND_BLUE),
                line=dict(color="#FFFFFF", width=1),
            ),
            text=cat_slice["Units_Sold"].map("{:,}".format),
            textposition="outside",
            textfont=dict(family="Inter, sans-serif", size=10.5, color=PRIMARY_NAVY),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Category: " + category + "<br>Total Units Sold: %{x:,}<extra></extra>",
        ))

    max_val = top_df["Units_Sold"].max()
    fig.update_xaxes(range=[0, max_val * 1.18])

    apply_saas_chart_theme(
        fig,
        title=f"Top {n} Best-Selling Products",
        subtitle="Ranked by cumulative units sold across all stores (Category grouped)",
        height=440,
        show_legend=True,
        legend_position="top",
        x_title="Total Units Sold",
        y_title="Product ID",
    )
    # Ensure product label spacing on Y-axis
    fig.update_layout(margin=dict(l=85, r=40, t=92, b=48))
    return fig


def create_monthly_sales_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly Sales Volume Clustered Bar: 2024 vs 2025 seasonality."""
    monthly = df.groupby(["Year", "Month"])["Units_Sold"].sum().reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["Month_Name"] = monthly["Month"].apply(lambda m: month_names[m - 1])

    m24 = monthly[monthly["Year"] == 2024].sort_values("Month")
    m25 = monthly[monthly["Year"] == 2025].sort_values("Month")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=m24["Month_Name"],
        y=m24["Units_Sold"],
        name="Year 2024",
        marker_color=BRAND_BLUE,
        hovertemplate="<b>%{x} 2024:</b> %{y:,} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=m25["Month_Name"],
        y=m25["Units_Sold"],
        name="Year 2025",
        marker_color=ACCENT_TEAL,
        hovertemplate="<b>%{x} 2025:</b> %{y:,} units<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title="Monthly Sales Volume Comparison (2024 vs 2025)",
        subtitle="Year-over-year seasonality, Q4 holiday surges, and annual retail expansion",
        height=420,
        show_legend=True,
        legend_position="top",
        x_title="Calendar Month",
        y_title="Total Units Sold",
    )
    fig.update_layout(barmode="group")
    return fig


# --- 2. FORECASTING CHARTS ---

def create_forecast_plot(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    store_id: str,
    product_id: str,
) -> go.Figure:
    """Autoregressive Demand Forecast Chart with clear visual hierarchy and zero overlap."""
    fig = go.Figure()

    # Historical Actuals (last 45 days)
    hist_tail = historical_df.tail(45)
    fig.add_trace(go.Scatter(
        x=hist_tail["Date"],
        y=hist_tail["Units_Sold"],
        mode="lines+markers",
        name="Historical Actual Demand",
        line=dict(color=BRAND_BLUE, width=2.2),
        marker=dict(size=4.5, color=BRAND_BLUE),
        hovertemplate="<b>%{x|%b %d, %Y} (Actual)</b><br>Demand: %{y} units<extra></extra>",
    ))

    # Connector line
    last_hist_date = hist_tail["Date"].iloc[-1]
    last_hist_val = hist_tail["Units_Sold"].iloc[-1]
    first_fc_date = pd.to_datetime(forecast_df["Date"].iloc[0])
    first_fc_val = forecast_df["Forecasted_Demand"].iloc[0]

    fig.add_trace(go.Scatter(
        x=[last_hist_date, first_fc_date],
        y=[last_hist_val, first_fc_val],
        mode="lines",
        line=dict(color=ACCENT_TEAL, width=2, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Forecast trace
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["Date"]),
        y=forecast_df["Forecasted_Demand"],
        mode="lines+markers",
        name="XGBoost Recursive Forecast",
        line=dict(color=ACCENT_TEAL, width=2.5, dash="dash"),
        marker=dict(size=6, symbol="circle", color=ACCENT_TEAL),
        hovertemplate="<b>%{x|%b %d, %Y} (%{text})</b><br>Forecast: %{y:.1f} units<extra></extra>",
        text=forecast_df["Day_of_Week"],
    ))

    # Highlight promotions in forecast
    promo_days = forecast_df[forecast_df["Promotion"] == 1]
    if not promo_days.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(promo_days["Date"]),
            y=promo_days["Forecasted_Demand"],
            mode="markers",
            name="Planned Promotion (Lift)",
            marker=dict(size=11, symbol="star", color=DANGER_RED, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="<b>Planned Promotion</b><br>Lifted Demand: %{y:.1f} units<extra></extra>",
        ))

    apply_saas_chart_theme(
        fig,
        title=f"Autoregressive Demand Forecast: {product_id} @ {store_id}",
        subtitle=f"{len(forecast_df)}-Day ahead recursive multi-step horizon with planned merchandising & calendar drivers",
        height=460,
        show_legend=True,
        legend_position="top",
        x_title="Timeline",
        y_title="Units Sold (Daily)",
        is_date_axis=True,
        hovermode="x unified",
    )
    return fig


# --- 3. PRODUCT & CATEGORY ANALYTICS CHARTS ---

def create_product_wise_demand_chart(df: pd.DataFrame) -> go.Figure:
    """Full-width horizontal bar chart for all 20 products sorted by total volume."""
    prod_df = (
        df.groupby(["Product_ID", "Category"])["Units_Sold"]
        .sum()
        .reset_index()
        .sort_values("Units_Sold", ascending=True)
    )

    fig = go.Figure()
    for cat in sorted(prod_df["Category"].unique()):
        cat_slice = prod_df[prod_df["Category"] == cat]
        fig.add_trace(go.Bar(
            y=cat_slice["Product_ID"],
            x=cat_slice["Units_Sold"],
            name=cat,
            orientation="h",
            marker=dict(color=CATEGORY_COLOR_MAP.get(cat, BRAND_BLUE)),
            text=cat_slice["Units_Sold"].map("{:,}".format),
            textposition="outside",
            textfont=dict(family="Inter, sans-serif", size=10),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Category: " + cat + "<br>Units Sold: %{x:,}<extra></extra>",
        ))

    max_v = prod_df["Units_Sold"].max()
    fig.update_xaxes(range=[0, max_v * 1.18])

    apply_saas_chart_theme(
        fig,
        title="Product-Level Total Demand Breakdown (All 20 SKUs)",
        subtitle="Cumulative sales volume across all retail locations over the 2-year observation window",
        height=520,
        show_legend=True,
        legend_position="top",
        x_title="Total Units Sold",
        y_title="Product ID",
    )
    fig.update_layout(margin=dict(l=85, r=40, t=92, b=48))
    return fig


def create_category_promo_lift_chart(df: pd.DataFrame) -> go.Figure:
    """Category promotional demand comparison: Regular Price vs On Promotion."""
    promo_stats = df.groupby(["Category", "Promotion"])["Units_Sold"].mean().reset_index()
    reg = promo_stats[promo_stats["Promotion"] == 0].set_index("Category")["Units_Sold"]
    prm = promo_stats[promo_stats["Promotion"] == 1].set_index("Category")["Units_Sold"]

    cat_list = sorted(df["Category"].unique().tolist())
    reg_vals = [reg.get(c, 0) for c in cat_list]
    prm_vals = [prm.get(c, 0) for c in cat_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_list,
        y=reg_vals,
        name="Regular Price",
        marker_color="#94A3B8",
        hovertemplate="<b>%{x} (Regular):</b> %{y:.1f} units/day<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=cat_list,
        y=prm_vals,
        name="On Promotion",
        marker_color=DANGER_RED,
        hovertemplate="<b>%{x} (Promotion):</b> %{y:.1f} units/day<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title="Promotional Demand Impact by Category",
        subtitle="Average daily sales comparing regular pricing vs promotional campaign periods",
        height=420,
        show_legend=True,
        legend_position="top",
        x_title="Category",
        y_title="Mean Daily Units Sold",
    )
    fig.update_layout(barmode="group")
    return fig


def create_store_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Store-wise Sales Volume and Revenue performance."""
    store_df = df.groupby("Store_ID").agg(
        Total_Units=("Units_Sold", "sum"),
        Total_Revenue=("Revenue", "sum"),
    ).reset_index().sort_values("Total_Units", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=store_df["Store_ID"],
        y=store_df["Total_Units"],
        name="Units Sold",
        marker_color=ACCENT_TEAL,
        text=store_df["Total_Units"].map("{:,}".format),
        textposition="outside",
        hovertemplate="<b>%{x}:</b> %{y:,} units sold<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title="Store-Wise Sales Volume Performance",
        subtitle="Comparative sales distribution across all 5 retail locations",
        height=420,
        show_legend=False,
        x_title="Store ID",
        y_title="Total Units Sold",
    )
    return fig


# --- 4. INVENTORY ANALYSIS CHARTS ---

def create_inventory_risk_donut(inv_df: pd.DataFrame) -> go.Figure:
    """Inventory Risk Classification Donut Chart."""
    status_counts = inv_df["Stock_Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    color_map = {
        "CRITICAL RISK": DANGER_RED,
        "REORDER REQUIRED": WARNING_AMBER,
        "HEALTHY": SUCCESS_GREEN,
        "OVERSTOCKED": BRAND_BLUE,
    }
    colors = [color_map.get(s, "#94A3B8") for s in status_counts["Status"]]

    fig = go.Figure(data=[go.Pie(
        labels=status_counts["Status"],
        values=status_counts["Count"],
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter, sans-serif", size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value} SKUs (%{percent})<extra></extra>",
    )])

    apply_saas_chart_theme(
        fig,
        title="Portfolio Inventory Risk Distribution",
        subtitle="Stock health classification across all 100 store-product inventory positions",
        height=430,
        show_legend=True,
        legend_position="bottom",
    )
    return fig


def create_top_at_risk_inventory_bar(inv_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Current Stock vs Reorder Point for the top N SKUs closest to stockout."""
    at_risk = inv_df.sort_values("Days_of_Supply", ascending=True).head(top_n).copy()
    at_risk["Label"] = at_risk["Store_ID"] + " - " + at_risk["Product_ID"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=at_risk["Label"],
        x=at_risk["Current_Stock"],
        name="Current On-Hand Stock",
        orientation="h",
        marker_color=DANGER_RED,
        hovertemplate="<b>%{y}</b><br>Current Stock: %{x} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=at_risk["Label"],
        x=at_risk["Reorder_Point"],
        name="Minimum Reorder Point (ROP)",
        orientation="h",
        marker_color=SECONDARY_NAVY,
        hovertemplate="<b>%{y}</b><br>Reorder Point: %{x} units<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title=f"Top {top_n} SKUs Nearing Stockout Risk",
        subtitle="Comparing on-hand inventory vs calculated replenishment safety threshold",
        height=430,
        show_legend=True,
        legend_position="top",
        x_title="Units",
        y_title="Store - Product",
    )
    fig.update_layout(barmode="group", margin=dict(l=130, r=40, t=92, b=48))
    return fig


# --- 5. MODEL BENCHMARK & EVALUATION CHARTS ---

def create_model_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Model comparison bar chart for Test MAE, RMSE, and WAPE."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["Test_MAE"],
        name="Test MAE (Units)",
        marker_color=BRAND_BLUE,
        text=comparison_df["Test_MAE"].round(2),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Test MAE: %{y:.3f} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["Test_RMSE"],
        name="Test RMSE",
        marker_color="#6366F1",
        text=comparison_df["Test_RMSE"].round(2),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Test RMSE: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=comparison_df["Model"],
        y=comparison_df["Test_WAPE(%)"],
        name="Test WAPE (%)",
        marker_color=ACCENT_TEAL,
        text=comparison_df["Test_WAPE(%)"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Test WAPE: %{y:.2f}%<extra></extra>",
    ))

    apply_saas_chart_theme(
        fig,
        title="Competitive Model Benchmark (Out-of-Sample Test Set)",
        subtitle="Comparing Linear Regression, Random Forest, and XGBoost on Q4 test data (Lower is Better)",
        height=440,
        show_legend=True,
        legend_position="top",
        x_title="Candidate Model",
        y_title="Error Score",
    )
    fig.update_layout(barmode="group")
    return fig


def create_feature_importance_plot(
    model: Any,
    preprocessor: Any,
    top_n: int = 12,
) -> Optional[go.Figure]:
    """Plot feature importance from winning trained model."""
    if not hasattr(model, "feature_importances_"):
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
        clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
        importances = model.feature_importances_

        imp_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(top_n)

        fig = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale=[[0, "#94A3B8"], [0.5, ACCENT_TEAL], [1.0, PRIMARY_NAVY]],
                line=dict(color="#FFFFFF", width=1),
            ),
            text=imp_df["Importance"].map(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(family="Inter, sans-serif", size=10),
            cliponaxis=False,
            hovertemplate="<b>%{y}:</b> %{x:.4f} relative gain<extra></extra>",
        ))

        apply_saas_chart_theme(
            fig,
            title=f"Top {top_n} Feature Importance Weights (Winning XGBoost Model)",
            subtitle="Relative information gain per engineered lag, rolling average, and calendar driver",
            height=440,
            show_legend=False,
            x_title="Relative Feature Importance",
            y_title="Predictor Feature",
        )
        fig.update_layout(margin=dict(l=110, r=40, t=92, b=48))
        return fig
    except Exception:
        return None


def create_actual_vs_predicted_scatter(test_preds_df: pd.DataFrame, sample_size: int = 500) -> go.Figure:
    """Actual vs Predicted demand scatter plot on out-of-sample test set."""
    sample = test_preds_df.sample(min(sample_size, len(test_preds_df)), random_state=42)

    fig = go.Figure()

    # 45-degree perfect prediction reference line
    max_val = max(sample["Actual_Units"].max(), sample["Predicted_Units"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect Fit (y = x)",
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    # Scatter points grouped by Category
    for cat in sorted(sample["Category"].unique()):
        cat_slice = sample[sample["Category"] == cat]
        fig.add_trace(go.Scatter(
            x=cat_slice["Actual_Units"],
            y=cat_slice["Predicted_Units"],
            mode="markers",
            name=cat,
            marker=dict(
                size=6.5,
                color=CATEGORY_COLOR_MAP.get(cat, BRAND_BLUE),
                opacity=0.75,
                line=dict(color="#FFFFFF", width=0.5),
            ),
            hovertemplate="<b>Category:</b> " + cat + "<br>Actual: %{x} units<br>Predicted: %{y:.1f} units<extra></extra>",
        ))

    apply_saas_chart_theme(
        fig,
        title="Actual vs Predicted Demand (Out-of-Sample Test Set: Q4 2025)",
        subtitle="Scatter evaluation of XGBoost test predictions against ground truth sales",
        height=440,
        show_legend=True,
        legend_position="top",
        x_title="Actual Units Sold",
        y_title="Predicted Units Sold",
    )
    return fig


# ==============================================================================
# INVENTORY CALCULATION ENGINE
# ==============================================================================
def compute_inventory_health(
    df: pd.DataFrame,
    lead_time_days: int = 2,
    safety_stock_days: int = 3,
    target_days: int = 7,
) -> pd.DataFrame:
    """Compute store-product inventory status, stockout risks, and suggested reorder quantities.

    Assumptions & Supply Chain Logic:
    - Average Daily Demand (D_avg): mean daily sales per store-product over the last 30 days.
    - Current Stock: latest recorded Inventory_Level.
    - Days of Supply = Current Stock / D_avg.
    - Reorder Point (ROP) = D_avg * (lead_time_days + safety_stock_days).
    - Target Stock Level = D_avg * target_days.
    - Suggested Reorder Quantity = max(0, Target Stock Level - Current Stock).
    - Risk Classification:
      * CRITICAL RISK: Days of Supply <= lead_time_days (cannot survive lead time).
      * REORDER REQUIRED: Current Stock <= ROP.
      * OVERSTOCKED: Days of Supply >= 14.0 days.
      * HEALTHY: Operating securely between ROP and Target.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    cutoff_30d = latest_date - pd.Timedelta(days=30)

    # 30-day average demand
    recent_df = df[df["Date"] >= cutoff_30d]
    demand_30d = (
        recent_df.groupby(["Store_ID", "Product_ID"])["Units_Sold"]
        .mean()
        .reset_index()
        .rename(columns={"Units_Sold": "Avg_Daily_Demand"})
    )

    # Latest inventory record for every store-product series
    sorted_df = df.sort_values("Date")
    latest_records = (
        sorted_df.groupby(["Store_ID", "Product_ID"])
        .last()
        .reset_index()[["Store_ID", "Product_ID", "Category", "Price", "Inventory_Level"]]
        .rename(columns={"Inventory_Level": "Current_Stock"})
    )

    merged = pd.merge(latest_records, demand_30d, on=["Store_ID", "Product_ID"], how="left")
    merged["Avg_Daily_Demand"] = merged["Avg_Daily_Demand"].fillna(1.0).clip(lower=0.5)

    merged["Days_of_Supply"] = (merged["Current_Stock"] / merged["Avg_Daily_Demand"]).round(1)
    merged["Reorder_Point"] = (merged["Avg_Daily_Demand"] * (lead_time_days + safety_stock_days)).round(1)
    merged["Target_Stock"] = (merged["Avg_Daily_Demand"] * target_days).round(1)

    # Reorder Quantity
    merged["Suggested_Reorder"] = np.maximum(
        0, np.round(merged["Target_Stock"] - merged["Current_Stock"])
    ).astype(int)

    # Risk Status
    conditions = [
        merged["Days_of_Supply"] <= lead_time_days,
        merged["Current_Stock"] <= merged["Reorder_Point"],
        merged["Days_of_Supply"] >= 14.0,
    ]
    choices = ["CRITICAL RISK", "REORDER REQUIRED", "OVERSTOCKED"]
    merged["Stock_Status"] = np.select(conditions, choices, default="HEALTHY")

    # Estimated Reorder Cost
    merged["Est_Reorder_Cost"] = (merged["Suggested_Reorder"] * merged["Price"]).round(2)

    return merged.sort_values(by=["Days_of_Supply", "Avg_Daily_Demand"], ascending=[True, False]).reset_index(drop=True)
