"""
Retail Exploratory Data Analysis (EDA) Module.

This script loads the preprocessed retail sales dataset and performs comprehensive
exploratory data analysis covering 10 key retail dimensions:
1. Daily Sales Trend (with 7-day rolling average)
2. Monthly Sales Trend
3. Product-wise Total Demand
4. Store-wise Total Sales and Revenue
5. Category-wise Demand Breakdown
6. Promotion vs Non-Promotion Demand Impact
7. Holiday vs Non-Holiday Demand Lift
8. Day-of-Week Demand Seasonality
9. Correlation Heatmap
10. Top 10 Best-Selling Products

Visualizations are saved in assets/eda/ using Matplotlib, Seaborn, and Plotly.
Statistical insights and summaries are printed to the console.
"""

import os
from pathlib import Path
import tempfile
from typing import Optional, Union

# Configure headless matplotlib environment cleanly
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Professional styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "figure.titlesize": 16,
    "figure.dpi": 300,
})

ASSETS_EDA_DIR = Path("assets/eda")


def load_processed_data(file_path: Union[str, Path] = "data/processed_retail_data.csv") -> pd.DataFrame:
    """Load the preprocessed retail dataset.

    Parameters:
    -----------
    file_path : str or Path
        Path to processed CSV.

    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run preprocessing.py first!")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_daily_sales_trend(df: pd.DataFrame, output_dir: Path) -> None:
    """1. Daily Sales Trend with 7-day moving average."""
    daily_sales = df.groupby("Date")["Units_Sold"].sum().reset_index()
    daily_sales["Rolling_7D"] = daily_sales["Units_Sold"].rolling(window=7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_sales["Date"], daily_sales["Units_Sold"], color="#90caf9", alpha=0.55, label="Daily Total Sales")
    ax.plot(daily_sales["Date"], daily_sales["Rolling_7D"], color="#1565c0", linewidth=2.2, label="7-Day Moving Average")

    ax.set_title("Overall Daily Sales Demand Trend (2024 - 2025)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    png_path = output_dir / "01_daily_sales_trend.png"
    plt.savefig(png_path)
    plt.close()

    # Interactive Plotly chart
    fig_plotly = px.line(
        daily_sales,
        x="Date",
        y=["Units_Sold", "Rolling_7D"],
        labels={"value": "Units Sold", "Date": "Date", "variable": "Metric"},
        title="Interactive Daily Sales Demand Trend (7-Day Rolling Average)",
        color_discrete_map={"Units_Sold": "#90caf9", "Rolling_7D": "#1565c0"},
    )
    fig_plotly.update_layout(template="plotly_white", hovermode="x unified")
    html_path = output_dir / "01_daily_sales_trend_interactive.html"
    fig_plotly.write_html(html_path.as_posix())


def plot_monthly_sales_trend(df: pd.DataFrame, output_dir: Path) -> None:
    """2. Monthly Sales Trend."""
    monthly_sales = df.groupby(["Year", "Month"])["Units_Sold"].sum().reset_index()
    monthly_sales["Period"] = (
        monthly_sales["Year"].astype(str) + "-" + monthly_sales["Month"].astype(str).str.zfill(2)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#1976d2" if yr == 2024 else "#388e3c" for yr in monthly_sales["Year"]]
    bars = ax.bar(monthly_sales["Period"], monthly_sales["Units_Sold"], color=palette, width=0.65, edgecolor="black", linewidth=0.5)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=45,
        )

    ax.set_title("Monthly Sales Volume (2024 vs 2025)")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Total Units Sold")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    plt.savefig(output_dir / "02_monthly_sales_trend.png")
    plt.close()


def plot_product_wise_demand(df: pd.DataFrame, output_dir: Path) -> None:
    """3. Product-wise Total Demand."""
    prod_sales = df.groupby(["Product_ID", "Category"])["Units_Sold"].sum().reset_index()
    prod_sales = prod_sales.sort_values("Units_Sold", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(
        data=prod_sales,
        x="Units_Sold",
        y="Product_ID",
        hue="Category",
        dodge=False,
        palette="Set2",
        ax=ax,
    )

    ax.set_title("Total Product Demand across All Stores")
    ax.set_xlabel("Total Units Sold")
    ax.set_ylabel("Product ID")
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(output_dir / "03_product_wise_demand.png")
    plt.close()


def plot_store_wise_sales(df: pd.DataFrame, output_dir: Path) -> None:
    """4. Store-wise Total Sales & Revenue."""
    store_stats = df.groupby("Store_ID").agg({
        "Units_Sold": "sum",
        "Revenue": "sum",
    }).reset_index().sort_values("Units_Sold", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Units Sold Bar Chart
    sns.barplot(
        data=store_stats,
        x="Store_ID",
        y="Units_Sold",
        hue="Store_ID",
        legend=False,
        palette="Blues_r",
        ax=ax1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Total Sales Volume by Store")
    ax1.set_xlabel("Store ID")
    ax1.set_ylabel("Total Units Sold")
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    # Revenue Bar Chart
    sns.barplot(
        data=store_stats,
        x="Store_ID",
        y="Revenue",
        hue="Store_ID",
        legend=False,
        palette="Greens_r",
        ax=ax2,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_title("Total Revenue by Store ($)")
    ax2.set_xlabel("Store ID")
    ax2.set_ylabel("Total Revenue ($)")
    for p in ax2.patches:
        ax2.annotate(f"${int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "04_store_wise_sales.png")
    plt.close()


def plot_category_wise_demand(df: pd.DataFrame, output_dir: Path) -> None:
    """5. Category-wise Demand Breakdown."""
    cat_sales = df.groupby("Category")["Units_Sold"].sum().reset_index().sort_values("Units_Sold", ascending=False)
    cat_sales["Share"] = (cat_sales["Units_Sold"] / cat_sales["Units_Sold"].sum()) * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(cat_sales["Category"], cat_sales["Units_Sold"], color=sns.color_palette("mako", len(cat_sales)), edgecolor="black", linewidth=0.5)

    for bar, share in zip(bars, cat_sales["Share"]):
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({share:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )

    ax.set_title("Total Demand Breakdown by Retail Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Total Units Sold")
    ax.set_ylim(0, cat_sales["Units_Sold"].max() * 1.15)
    plt.tight_layout()

    plt.savefig(output_dir / "05_category_wise_demand.png")
    plt.close()

    # Interactive Plotly Donut Chart
    fig_donut = px.pie(
        cat_sales,
        names="Category",
        values="Units_Sold",
        hole=0.45,
        title="Category Demand Share (Interactive Donut)",
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    fig_donut.write_html((output_dir / "05_category_demand_donut.html").as_posix())


def plot_promotion_effect(df: pd.DataFrame, output_dir: Path) -> None:
    """6. Promotion vs Non-Promotion Demand Impact."""
    promo_stats = df.groupby("Promotion")["Units_Sold"].mean().reset_index()
    promo_stats["Status"] = promo_stats["Promotion"].map({0: "Regular Price", 1: "On Promotion"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Mean comparison bar chart
    sns.barplot(
        data=promo_stats,
        x="Status",
        y="Units_Sold",
        hue="Status",
        legend=False,
        palette=["#78909c", "#e53935"],
        ax=ax1,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_title("Average Daily Units Sold per Record")
    ax1.set_xlabel("Pricing Status")
    ax1.set_ylabel("Mean Units Sold")
    diff_pct = ((promo_stats.loc[promo_stats["Promotion"] == 1, "Units_Sold"].values[0] /
                 promo_stats.loc[promo_stats["Promotion"] == 0, "Units_Sold"].values[0]) - 1) * 100
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f} units", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax1.text(0.5, 0.9, f"Promotional Lift: +{diff_pct:.1f}%", transform=ax1.transAxes, ha="center",
             fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#e53935"))

    # Distribution Boxplot
    df_box = df.copy()
    df_box["Pricing_Status"] = df_box["Promotion"].map({0: "Regular Price", 1: "On Promotion"})
    sns.boxplot(
        data=df_box,
        x="Pricing_Status",
        y="Units_Sold",
        hue="Pricing_Status",
        legend=False,
        palette=["#90a4ae", "#ef5350"],
        ax=ax2,
        showfliers=False,
    )
    ax2.set_title("Sales Distribution (Promotion vs Regular)")
    ax2.set_xlabel("Pricing Status")
    ax2.set_ylabel("Units Sold (excl. outliers)")

    plt.tight_layout()
    plt.savefig(output_dir / "06_promotion_vs_non_promotion.png")
    plt.close()


def plot_holiday_effect(df: pd.DataFrame, output_dir: Path) -> None:
    """7. Holiday vs Non-Holiday Demand Lift."""
    holiday_stats = df.groupby("Holiday")["Units_Sold"].mean().reset_index()
    holiday_stats["Status"] = holiday_stats["Holiday"].map({0: "Regular Day", 1: "Retail Holiday"})

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(
        data=holiday_stats,
        x="Status",
        y="Units_Sold",
        hue="Status",
        legend=False,
        palette=["#42a5f5", "#ff7043"],
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    lift = ((holiday_stats.loc[holiday_stats["Holiday"] == 1, "Units_Sold"].values[0] /
             holiday_stats.loc[holiday_stats["Holiday"] == 0, "Units_Sold"].values[0]) - 1) * 100

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f} units/day", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax.set_title(f"Holiday Demand Surge (Lift: +{lift:.1f}%)")
    ax.set_xlabel("Calendar Day Type")
    ax.set_ylabel("Mean Units Sold per Record")
    plt.tight_layout()

    plt.savefig(output_dir / "07_holiday_vs_non_holiday.png")
    plt.close()


def plot_day_of_week_demand(df: pd.DataFrame, output_dir: Path) -> None:
    """8. Day-of-Week Demand Seasonality."""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = df.groupby("Day_of_Week")["Units_Sold"].mean().reindex(day_order).reset_index()

    # Colors: subtle for weekdays, accented for weekend
    palette = ["#90caf9" if d not in ["Saturday", "Sunday"] else "#1e88e5" for d in day_order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(dow_stats["Day_of_Week"], dow_stats["Units_Sold"], color=palette, edgecolor="black", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Average Demand by Day of Week (Weekend Peak Highlighted)")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Units Sold")
    ax.set_ylim(0, dow_stats["Units_Sold"].max() * 1.15)
    plt.tight_layout()

    plt.savefig(output_dir / "08_day_of_week_demand.png")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """9. Correlation Heatmap."""
    numeric_cols = [
        "Units_Sold",
        "Price",
        "Promotion",
        "Holiday",
        "Inventory_Level",
        "Is_Weekend",
        "Month",
        "Day_of_Week_Num",
        "Revenue",
    ]
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(output_dir / "09_correlation_heatmap.png")
    plt.close()


def plot_top_10_products(df: pd.DataFrame, output_dir: Path) -> None:
    """10. Top 10 Best-Selling Products."""
    top_10 = (
        df.groupby(["Product_ID", "Category"])
        .agg({"Units_Sold": "sum", "Revenue": "sum", "Price": "mean"})
        .reset_index()
        .sort_values("Units_Sold", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = sns.barplot(
        data=top_10,
        x="Units_Sold",
        y="Product_ID",
        hue="Category",
        dodge=False,
        palette="viridis",
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )

    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(
                f"{int(width):,}",
                (width, p.get_y() + p.get_height() / 2.),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9.5,
            )

    ax.set_title("Top 10 Best-Selling Products by Total Sales Volume")
    ax.set_xlabel("Total Units Sold")
    ax.set_ylabel("Product ID")
    ax.legend(title="Category", loc="lower right")
    ax.set_xlim(0, top_10["Units_Sold"].max() * 1.15)
    plt.tight_layout()

    plt.savefig(output_dir / "10_top_10_best_selling_products.png")
    plt.close()

    # Interactive Plotly chart
    fig_plotly = px.bar(
        top_10,
        x="Product_ID",
        y="Units_Sold",
        color="Category",
        hover_data={"Price": ":$.2f", "Revenue": ":$,.2f", "Units_Sold": ":,"},
        title="Top 10 Best-Selling Products (Interactive)",
        text="Units_Sold",
    )
    fig_plotly.update_traces(textposition="outside")
    fig_plotly.update_layout(template="plotly_white")
    fig_plotly.write_html((output_dir / "10_top_10_products_interactive.html").as_posix())


def print_statistical_insights(df: pd.DataFrame) -> None:
    """Compute and print key business and statistical insights."""
    total_volume = df["Units_Sold"].sum()
    total_revenue = df["Revenue"].sum()
    avg_price = df["Price"].mean()

    # Promotion lift
    promo_avg = df.loc[df["Promotion"] == 1, "Units_Sold"].mean()
    regular_avg = df.loc[df["Promotion"] == 0, "Units_Sold"].mean()
    promo_lift = ((promo_avg / regular_avg) - 1) * 100

    # Holiday lift
    holiday_avg = df.loc[df["Holiday"] == 1, "Units_Sold"].mean()
    normal_avg = df.loc[df["Holiday"] == 0, "Units_Sold"].mean()
    holiday_lift = ((holiday_avg / normal_avg) - 1) * 100

    # Weekend vs Weekday
    weekend_avg = df.loc[df["Is_Weekend"] == 1, "Units_Sold"].mean()
    weekday_avg = df.loc[df["Is_Weekend"] == 0, "Units_Sold"].mean()
    weekend_lift = ((weekend_avg / weekday_avg) - 1) * 100

    # Top Store and Category
    store_summary = df.groupby("Store_ID")["Units_Sold"].sum()
    top_store = store_summary.idxmax()
    top_store_vol = store_summary.max()

    cat_summary = df.groupby("Category")["Units_Sold"].sum()
    top_cat = cat_summary.idxmax()
    top_cat_vol = cat_summary.max()

    print("\n" + "=" * 70)
    print("KEY RETAIL BUSINESS & STATISTICAL INSIGHTS")
    print("=" * 70)
    print(f"Total Sales Volume (Units Sold): {total_volume:,}")
    print(f"Total Gross Revenue:             ${total_revenue:,.2f}")
    print(f"Average Unit Selling Price:      ${avg_price:.2f}")
    print(f"Top Performing Store:            {top_store} ({top_store_vol:,} units sold)")
    print(f"Leading Product Category:        {top_cat} ({top_cat_vol:,} units sold, {(top_cat_vol/total_volume)*100:.1f}%)")
    print(f"Promotional Demand Lift:         +{promo_lift:.2f}% (Avg {promo_avg:.1f} vs {regular_avg:.1f} units/day)")
    print(f"Holiday Demand Surge:            +{holiday_lift:.2f}% (Avg {holiday_avg:.1f} vs {normal_avg:.1f} units/day)")
    print(f"Weekend Demand Premium:          +{weekend_lift:.2f}% (Avg {weekend_avg:.1f} vs {weekday_avg:.1f} units/day)")
    print("=" * 70)

    # Top 10 Best Sellers Table
    top_10 = (
        df.groupby(["Product_ID", "Category"])
        .agg(
            Total_Units_Sold=("Units_Sold", "sum"),
            Total_Revenue=("Revenue", "sum"),
            Avg_Price=("Price", "mean"),
        )
        .reset_index()
        .sort_values("Total_Units_Sold", ascending=False)
        .head(10)
    )
    print("\nTOP 10 BEST-SELLING PRODUCTS:")
    print(top_10.to_string(index=False))


def run_eda(
    data_path: Union[str, Path] = "data/processed_retail_data.csv",
    output_dir: Union[str, Path] = ASSETS_EDA_DIR,
) -> None:
    """Execute complete EDA pipeline and generate all artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
    print("=" * 70)

    # Load preprocessed dataset
    df = load_processed_data(data_path)
    print(f"[DATA LOADED] {len(df):,} rows from {Path(data_path).as_posix()}")
    print(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Stores: {df['Store_ID'].nunique()} | Total Products: {df['Product_ID'].nunique()} | Categories: {df['Category'].nunique()}\n")

    # Generate 10 Visualizations
    print("[VISUALIZATION 1/10] Generating Daily Sales Trend...")
    plot_daily_sales_trend(df, out_dir)

    print("[VISUALIZATION 2/10] Generating Monthly Sales Trend...")
    plot_monthly_sales_trend(df, out_dir)

    print("[VISUALIZATION 3/10] Generating Product-wise Demand...")
    plot_product_wise_demand(df, out_dir)

    print("[VISUALIZATION 4/10] Generating Store-wise Sales & Revenue...")
    plot_store_wise_sales(df, out_dir)

    print("[VISUALIZATION 5/10] Generating Category-wise Demand Breakdown...")
    plot_category_wise_demand(df, out_dir)

    print("[VISUALIZATION 6/10] Generating Promotion vs Non-Promotion Demand...")
    plot_promotion_effect(df, out_dir)

    print("[VISUALIZATION 7/10] Generating Holiday vs Non-Holiday Demand...")
    plot_holiday_effect(df, out_dir)

    print("[VISUALIZATION 8/10] Generating Day-of-Week Demand Seasonality...")
    plot_day_of_week_demand(df, out_dir)

    print("[VISUALIZATION 9/10] Generating Correlation Heatmap...")
    plot_correlation_heatmap(df, out_dir)

    print("[VISUALIZATION 10/10] Generating Top 10 Best-Selling Products...")
    plot_top_10_products(df, out_dir)

    print(f"\n[SUCCESS] All visualizations successfully saved to: {out_dir.as_posix()}/")

    # Statistical Insights
    print_statistical_insights(df)


def main():
    """CLI execution for EDA module."""
    data_file = "data/processed_retail_data.csv"
    run_eda(data_path=data_file, output_dir=ASSETS_EDA_DIR)


if __name__ == "__main__":
    main()
