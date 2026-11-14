"""
Feature Engineering Module for Retail Demand Forecasting.

This module provides reusable functions to transform raw/preprocessed retail sales data
into ML-ready feature matrices while strictly preventing data leakage:
- Chronological sorting by Store_ID, Product_ID, and Date.
- Date and calendar feature extraction (Year, Month, Day, Day_of_Week, Is_Weekend).
- Cyclical calendar encodings (sine and cosine transforms for month and day of week).
- Autoregressive lag features (1, 7, 14, and 28 days).
- Rolling window mean features (7 and 28 days) strictly computed on shifted target
  values to prevent lookahead data leakage.
- Encoding of store, product, and category contextual attributes.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Default lag and rolling window configurations
LAG_DAYS: List[int] = [1, 7, 14, 28]
ROLLING_WINDOWS: List[int] = [7, 28]

FEATURE_COLUMNS: List[str] = [
    # Contextual and Pricing
    "Price",
    "Promotion",
    "Holiday",
    # Calendar & Seasonality
    "Month",
    "Day_of_Week_Num",
    "Is_Weekend",
    "Month_Sin",
    "Month_Cos",
    "Day_Sin",
    "Day_Cos",
    # Autoregressive Lags
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    # Rolling Statistics
    "Rolling_Mean_7",
    "Rolling_Mean_28",
    # Categorical identifiers (encoded during model pipeline)
    "Store_ID",
    "Product_ID",
    "Category",
]

CATEGORICAL_COLUMNS: List[str] = ["Store_ID", "Product_ID", "Category"]
NUMERICAL_COLUMNS: List[str] = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date features and cyclical encodings.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Date' column.

    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted calendar features.
    """
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"])

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["Day_of_Week_Num"] = data["Date"].dt.day_of_week
    data["Is_Weekend"] = data["Day_of_Week_Num"].isin([5, 6]).astype(int)

    # Cyclical sine and cosine encodings to preserve circular calendar continuity
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12.0)
    data["Day_Sin"] = np.sin(2 * np.pi * data["Day_of_Week_Num"] / 7.0)
    data["Day_Cos"] = np.cos(2 * np.pi * data["Day_of_Week_Num"] / 7.0)

    return data


def create_lag_features(
    df: pd.DataFrame,
    lag_days: Optional[List[int]] = None,
    target_col: str = "Units_Sold",
) -> pd.DataFrame:
    """Generate autoregressive lag features for each Store-Product time series.

    Parameters:
    -----------
    df : pd.DataFrame
        Sorted DataFrame containing Store_ID, Product_ID, and target_col.
    lag_days : list of int, optional
        Lag shifts in days (default: [1, 7, 14, 28]).
    target_col : str
        Target column name to lag.

    Returns:
    --------
    pd.DataFrame
        DataFrame with appended lag columns.
    """
    if lag_days is None:
        lag_days = LAG_DAYS

    data = df.copy()

    for lag in lag_days:
        col_name = f"Lag_{lag}"
        data[col_name] = data.groupby(["Store_ID", "Product_ID"])[target_col].shift(lag)

    return data


def create_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    target_col: str = "Units_Sold",
) -> pd.DataFrame:
    """Generate rolling mean features strictly on shifted data to prevent data leakage.

    NOTE: The target column is shifted by 1 before calculating the rolling mean.
    This ensures that for day t, the rolling statistics use ONLY observations up to
    day t-1 and never include day t's sales.

    Parameters:
    -----------
    df : pd.DataFrame
        Sorted DataFrame containing Store_ID, Product_ID, and target_col.
    windows : list of int, optional
        Window sizes in days (default: [7, 28]).
    target_col : str
        Target column name to compute rolling statistics on.

    Returns:
    --------
    pd.DataFrame
        DataFrame with appended rolling mean columns.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    data = df.copy()

    # Shift by 1 first to prevent lookahead leakage
    for w in windows:
        col_name = f"Rolling_Mean_{w}"
        # Group by Store and Product, shift by 1, then apply rolling mean
        data[col_name] = (
            data.groupby(["Store_ID", "Product_ID"])[target_col]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
        )

    return data


def build_feature_dataset(
    df: pd.DataFrame,
    target_col: str = "Units_Sold",
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    """Orchestrate complete feature engineering pipeline on retail dataset.

    Steps:
    1. Ensure chronological sort by Store_ID, Product_ID, Date.
    2. Extract date features & cyclical encodings.
    3. Generate 1, 7, 14, 28-day lags.
    4. Generate 7 and 28-day rolling means (strictly shifted).
    5. Optionally drop initial rows containing NaN from lag warm-up period.

    Parameters:
    -----------
    df : pd.DataFrame
        Input preprocessed dataset.
    target_col : str
        Target column name.
    drop_na_lags : bool
        Whether to drop rows with NaN in lag/rolling features (e.g. first 28 days).

    Returns:
    --------
    pd.DataFrame
        Enriched feature dataset.
    """
    data = df.copy()

    # 1. Ensure Chronological Sorting
    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(by=["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)

    # 2. Date features & Cyclical Encodings
    data = extract_date_features(data)

    # 3. Lag features
    data = create_lag_features(data, lag_days=LAG_DAYS, target_col=target_col)

    # 4. Rolling mean features
    data = create_rolling_features(data, windows=ROLLING_WINDOWS, target_col=target_col)

    # 5. Handle initial NaN lag rows
    if drop_na_lags:
        lag_cols = [f"Lag_{l}" for l in LAG_DAYS] + [f"Rolling_Mean_{w}" for w in ROLLING_WINDOWS]
        initial_count = len(data)
        data = data.dropna(subset=lag_cols).reset_index(drop=True)
        dropped_count = initial_count - len(data)
        print(f"[FEATURE ENG] Dropped {dropped_count:,} warm-up lag rows (first 28 days of each series).")
        print(f"[FEATURE ENG] Remaining ML-ready records: {len(data):,}")

    return data


def compute_lag_and_rolling_single_step(
    history_sales: List[float],
) -> dict:
    """Compute lag and rolling features for a single future day given sales history.

    Used by the recursive multi-step forecasting engine in `predict.py`.

    Parameters:
    -----------
    history_sales : list of float
        Chronological list of sales up to day t-1. Most recent day is at index -1.

    Returns:
    --------
    dict
        Dictionary containing Lag_1, Lag_7, Lag_14, Lag_28, Rolling_Mean_7, Rolling_Mean_28.
    """
    n = len(history_sales)
    if n < 28:
        raise ValueError(f"Sales history must contain at least 28 days for feature extraction, got {n} days.")

    # In 0-indexed list where history_sales[-1] is day t-1:
    lag_1 = float(history_sales[-1])
    lag_7 = float(history_sales[-7])
    lag_14 = float(history_sales[-14])
    lag_28 = float(history_sales[-28])

    rolling_7 = float(np.mean(history_sales[-7:]))
    rolling_28 = float(np.mean(history_sales[-28:]))

    return {
        "Lag_1": lag_1,
        "Lag_7": lag_7,
        "Lag_14": lag_14,
        "Lag_28": lag_28,
        "Rolling_Mean_7": rolling_7,
        "Rolling_Mean_28": rolling_28,
    }


def main():
    """CLI test execution for feature engineering."""
    data_path = Path("data/processed_retail_data.csv")
    if not data_path.exists():
        print(f"Data file not found at {data_path}. Run preprocessing.py first!")
        return

    print("Loading preprocessed retail dataset...")
    df = pd.read_csv(data_path)
    featured_df = build_feature_dataset(df)

    print("\nFeature Engineering Verification:")
    print(f"Dataset Shape: {featured_df.shape}")
    print("\nFeature Columns Available:")
    for col in FEATURE_COLUMNS:
        print(f"  - {col}: dtype {featured_df[col].dtype}")

    print("\nSample Engineered Features (first 3 rows):")
    cols_to_preview = ["Date", "Store_ID", "Product_ID", "Units_Sold", "Lag_1", "Lag_7", "Rolling_Mean_7", "Rolling_Mean_28"]
    print(featured_df[cols_to_preview].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
