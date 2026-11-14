"""
Demand Prediction & Recursive Multi-Step Forecasting Engine.

This module provides reusable inference and forecasting functions:
- Loads the trained model bundle and fitted preprocessors from models/.
- Performs recursive multi-step autoregressive demand forecasting for any (Store_ID, Product_ID).
- Dynamically updates lag features (Lag 1, 7, 14, 28) and rolling means (7, 28 days)
  across future horizons using prior step predictions.
- Enforces retail demand constraints (strictly non-negative outputs).
- Supports user-specified planned promotions, custom prices, and flexible forecast horizons.
- Documents all forecasting assumptions and provides error handling for out-of-scope inputs.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_COLUMNS,
    compute_lag_and_rolling_single_step,
    extract_date_features,
)

DEFAULT_MODEL_PATH = Path("models/best_forecasting_model.joblib")
DEFAULT_DATA_PATH = Path("data/processed_retail_data.csv")

# Standard US retail holidays for upcoming forecast horizons (2026 calendar)
HOLIDAY_CALENDAR_2026 = {
    "2026-01-01": "New Year's Day",
    "2026-01-19": "Martin Luther King Jr. Day",
    "2026-02-14": "Valentine's Day",
    "2026-02-16": "Presidents' Day",
    "2026-04-05": "Easter Sunday",
    "2026-05-10": "Mother's Day",
    "2026-05-25": "Memorial Day",
    "2026-06-21": "Father's Day",
    "2026-07-04": "Independence Day",
    "2026-09-07": "Labor Day",
    "2026-10-31": "Halloween",
    "2026-11-26": "Thanksgiving",
    "2026-11-27": "Black Friday",
    "2026-11-28": "Cyber Weekend",
    "2026-12-24": "Christmas Eve",
    "2026-12-25": "Christmas Day",
    "2026-12-31": "New Year's Eve",
}


def load_model_bundle(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Load the serialized model bundle from disk.

    Parameters:
    -----------
    model_path : str or Path
        Path to the saved .joblib file.

    Returns:
    --------
    dict
        Dictionary containing trained model, fitted preprocessor, feature names, and metadata.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecasting model bundle not found at {path.resolve()}. "
            "Please run 'python src/train_model.py' first!"
        )

    bundle = joblib.load(path)
    required_keys = ["model", "preprocessor", "feature_columns", "model_name"]
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise ValueError(f"Corrupted model bundle. Missing required keys: {missing}")

    return bundle


def get_available_entities(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Retrieve list of available Store IDs, Product IDs, and Product->Category mapping.

    Parameters:
    -----------
    data_path : str or Path
        Path to processed retail CSV.

    Returns:
    --------
    Tuple[list, list, dict]
        Available stores, products, and product-category mapping.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path.resolve()}")

    df = pd.read_csv(path, usecols=["Store_ID", "Product_ID", "Category"])
    stores = sorted(df["Store_ID"].unique().tolist())
    products = sorted(df["Product_ID"].unique().tolist())
    product_categories = (
        df.drop_duplicates(subset=["Product_ID"])
        .set_index("Product_ID")["Category"]
        .to_dict()
    )

    return stores, products, product_categories


def get_series_history(
    store_id: str,
    product_id: str,
    data_source: Union[str, Path, pd.DataFrame] = DEFAULT_DATA_PATH,
    min_days: int = 35,
) -> Tuple[List[float], str, float, str]:
    """Extract historical sales and attributes for a specific store and product series.

    Parameters:
    -----------
    store_id : str
        Target store identifier.
    product_id : str
        Target product identifier.
    data_source : str, Path, or pd.DataFrame
        Path to processed dataset or loaded DataFrame.
    min_days : int
        Minimum days of history required for lag warm-up (must be >= 28).

    Returns:
    --------
    Tuple[List[float], str, float, str]
        sales_history (list of float), latest_date_str, latest_price, category
    """
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {path.resolve()}")
        df = pd.read_csv(path)

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    series_df = df[(df["Store_ID"] == store_id) & (df["Product_ID"] == product_id)]
    if series_df.empty:
        raise ValueError(f"No historical records found for Store '{store_id}' and Product '{product_id}'")

    series_df = series_df.sort_values("Date").reset_index(drop=True)

    if len(series_df) < min_days:
        raise ValueError(
            f"Series has {len(series_df)} days of history; at least {min_days} days are required "
            "to construct 28-day lags and rolling statistics."
        )

    sales_history = series_df["Units_Sold"].astype(float).tolist()
    latest_row = series_df.iloc[-1]
    latest_date_str = latest_row["Date"].strftime("%Y-%m-%d")
    latest_price = float(latest_row["Price"])
    category = str(latest_row["Category"])

    return sales_history, latest_date_str, latest_price, category


def forecast_demand_recursive(
    store_id: str,
    product_id: str,
    forecast_days: int = 14,
    start_date: Optional[str] = None,
    planned_promotions: Optional[Dict[str, int]] = None,
    custom_prices: Optional[Dict[str, float]] = None,
    data_source: Union[str, Path, pd.DataFrame] = DEFAULT_DATA_PATH,
    model_bundle_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Generate recursive multi-step retail demand forecasts.

    Forecasting Mechanism:
    -----------------------
    For each day t in horizon (1..H):
    1. Extract calendar features (Month, Day, Day_of_Week, Is_Weekend, Cyclical Sin/Cos).
    2. Check holiday status from the retail calendar.
    3. Look up promotional plan and pricing (planned or defaulted).
    4. Construct autoregressive lags (t-1, t-7, t-14, t-28) and rolling means (7, 28)
       from the dynamic history buffer.
    5. Transform features using the fitted ColumnTransformer.
    6. Predict demand using the trained model and enforce non-negativity:
       forecast_t = max(0.0, raw_prediction).
    7. RECURSIVE UPDATE: Append forecast_t to the sales history buffer so that day t+1
       automatically consumes forecast_t as its Lag_1, and subsequent rolling averages
       adapt smoothly!

    Parameters:
    -----------
    store_id : str
        Target store identifier (e.g., 'STORE_001').
    product_id : str
        Target product identifier (e.g., 'PROD_001').
    forecast_days : int
        Number of future days to forecast (e.g. 7, 14, 30).
    start_date : str, optional
        Forecast start date (YYYY-MM-DD). If None, begins the day after historical data ends.
    planned_promotions : dict, optional
        Mapping of date strings ('YYYY-MM-DD') to promotion flags (1 or 0).
    custom_prices : dict, optional
        Mapping of date strings ('YYYY-MM-DD') to price overrides.
    data_path : str or Path
        Path to processed historical CSV.
    model_bundle_path : str or Path
        Path to serialized model bundle.

    Returns:
    --------
    pd.DataFrame
        DataFrame of forecasted daily demand and inputs.
    """
    if forecast_days <= 0:
        raise ValueError(f"forecast_days must be a positive integer, got {forecast_days}")

    # 1. Load model bundle
    bundle = load_model_bundle(model_bundle_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_columns = bundle["feature_columns"]

    # 2. Extract historical series
    sales_history, latest_date_str, latest_price, category = get_series_history(
        store_id=store_id,
        product_id=product_id,
        data_source=data_source,
        min_days=35,
    )

    # 3. Determine start date
    if start_date is None:
        start_dt = datetime.strptime(latest_date_str, "%Y-%m-%d") + timedelta(days=1)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    planned_promotions = planned_promotions or {}
    custom_prices = custom_prices or {}

    # Dynamic sales history buffer that will be updated recursively
    dynamic_history = list(sales_history)
    forecast_records: List[Dict[str, Any]] = []

    # 4. Recursive Step-by-Step Forecasting
    for step in range(forecast_days):
        current_dt = start_dt + timedelta(days=step)
        date_str = current_dt.strftime("%Y-%m-%d")

        # Calendar features
        month = current_dt.month
        day = current_dt.day
        dow_num = current_dt.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if dow_num in [5, 6] else 0
        dow_name = current_dt.strftime("%A")

        # Cyclical calendar encodings
        month_sin = float(np.sin(2 * np.pi * month / 12.0))
        month_cos = float(np.cos(2 * np.pi * month / 12.0))
        day_sin = float(np.sin(2 * np.pi * dow_num / 7.0))
        day_cos = float(np.cos(2 * np.pi * dow_num / 7.0))

        # Holiday flag
        holiday_flag = 1 if date_str in HOLIDAY_CALENDAR_2026 else 0

        # Promotion flag
        promo_flag = planned_promotions.get(date_str, 0)

        # Price determination
        if date_str in custom_prices:
            price = float(custom_prices[date_str])
        elif promo_flag == 1:
            # Standard promotional discount (15% off regular price)
            price = round(latest_price * 0.85, 2)
        else:
            price = latest_price

        # Autoregressive lag & rolling features from dynamic history buffer
        lag_roll_features = compute_lag_and_rolling_single_step(dynamic_history)

        # Construct single-step feature dictionary
        row_dict: Dict[str, Any] = {
            # Contextual & Pricing
            "Price": price,
            "Promotion": promo_flag,
            "Holiday": holiday_flag,
            # Calendar & Seasonality
            "Month": month,
            "Day_of_Week_Num": dow_num,
            "Is_Weekend": is_weekend,
            "Month_Sin": month_sin,
            "Month_Cos": month_cos,
            "Day_Sin": day_sin,
            "Day_Cos": day_cos,
            # Autoregressive Lags
            "Lag_1": lag_roll_features["Lag_1"],
            "Lag_7": lag_roll_features["Lag_7"],
            "Lag_14": lag_roll_features["Lag_14"],
            "Lag_28": lag_roll_features["Lag_28"],
            # Rolling Statistics
            "Rolling_Mean_7": lag_roll_features["Rolling_Mean_7"],
            "Rolling_Mean_28": lag_roll_features["Rolling_Mean_28"],
            # Categorical Identifiers
            "Store_ID": store_id,
            "Product_ID": product_id,
            "Category": category,
        }

        # Create 1-row DataFrame aligned with training feature columns
        step_df = pd.DataFrame([row_dict])[feature_columns]

        # Transform and Predict
        X_trans = preprocessor.transform(step_df)
        raw_pred = float(model.predict(X_trans)[0])

        # Enforce non-negative retail demand constraint
        forecast_units = max(0.0, raw_pred)

        # RECURSIVE UPDATE: Append forecast to history for subsequent steps!
        dynamic_history.append(forecast_units)

        forecast_records.append({
            "Date": date_str,
            "Day_of_Week": dow_name,
            "Store_ID": store_id,
            "Product_ID": product_id,
            "Category": category,
            "Price": price,
            "Promotion": promo_flag,
            "Holiday": holiday_flag,
            "Forecasted_Demand": round(forecast_units, 2),
            "Rounded_Demand": int(round(forecast_units)),
        })

    forecast_df = pd.DataFrame(forecast_records)
    return forecast_df


def explain_forecasting_assumptions() -> str:
    """Return a comprehensive explanation of retail forecasting assumptions."""
    explanation = """
================================================================================
FORECASTING METHODOLOGY & KEY ASSUMPTIONS
================================================================================
1. Autoregressive Recursive Multi-Step Horizon:
   - For day t = 1, the model uses actual historical sales observations for Lag_1,
     Lag_7, Lag_14, Lag_28, and rolling averages.
   - For day t >= 2, prior model predictions are appended to the historical buffer
     and dynamically feed into subsequent lag and rolling statistics.
   - This recursive approach enables arbitrary multi-week forecast horizons while
     preserving seasonal and autoregressive momentum.

2. Known Exogenous Calendar Drivers:
   - Calendar attributes (Year, Month, Day, Day of Week, Weekends, and Holidays)
     are deterministic and known with 100% precision in advance.
   - Cyclical sine/cosine encodings maintain continuous seasonal curves across month
     and week boundaries.

3. Commercial and Merchandising Inputs:
   - Retail promotions and pricing are planned management decisions. The user can
     specify planned promotional dates and custom pricing overrides.
   - When unspecified, prices default to the latest observed regular price, and
     promotions during promotional days apply a standard 15% discount.

4. Retail Supply-Chain Constraints:
   - Physical customer demand cannot be negative. Any raw prediction < 0 is clipped
     to 0.0 units.
   - Both continuous demand (for aggregate revenue and safety-stock planning) and
     rounded integer units (for inventory purchase orders) are provided.
================================================================================
"""
    return explanation.strip()


def main():
    """CLI test execution for demand forecasting."""
    print("=" * 70)
    print("SMART RETAIL DEMAND FORECASTING - PREDICTION MODULE")
    print("=" * 70)

    # 1. Print Assumptions
    print(explain_forecasting_assumptions())
    print("\n" + "=" * 70)

    # 2. Check available entities
    stores, products, categories = get_available_entities()
    print(f"Available Stores ({len(stores)}): {', '.join(stores)}")
    print(f"Available Products ({len(products)}): {', '.join(products[:6])} ...")

    # 3. Test Recursive Forecast on sample store and product
    sample_store = "STORE_001"
    sample_prod = "PROD_001"  # Organic Milk (Dairy & Bakery)
    sample_horizon = 14

    # Simulate a planned promotion on Day 5 and Day 6 (e.g. weekend promo)
    planned_promos = {
        "2026-01-05": 1,
        "2026-01-06": 1,
    }

    print(f"\n[FORECAST] Generating {sample_horizon}-day recursive forecast for {sample_store} - {sample_prod}...")
    forecast_df = predict_future_demand(
        store_id=sample_store,
        product_id=sample_prod,
        forecast_days=sample_horizon,
        planned_promotions=planned_promos,
    )

    print("\nFORECAST RESULTS TABLE:")
    print("-" * 80)
    print(forecast_df.to_string(index=False))
    print("-" * 80)

    total_forecasted_units = forecast_df["Rounded_Demand"].sum()
    avg_daily_demand = forecast_df["Forecasted_Demand"].mean()
    print(f"\nSummary for {sample_store} | {sample_prod}:")
    print(f" - Horizon:                {sample_horizon} days")
    print(f" - Total Projected Demand: {total_forecasted_units:,} units")
    print(f" - Average Daily Demand:   {avg_daily_demand:.1f} units/day")

    # 4. Error Handling Verification
    print("\n[VALIDATION] Testing error handling for invalid Store ID...")
    try:
        predict_future_demand(store_id="STORE_999", product_id=sample_prod)
    except ValueError as err:
        print(f" -> Successfully caught expected error: {err}")

    print("\n" + "=" * 70)
    print("PREDICTION TESTING COMPLETE")
    print("=" * 70)


# Provide alias function name matching prompt requirements
predict_future_demand = forecast_demand_recursive


if __name__ == "__main__":
    main()
