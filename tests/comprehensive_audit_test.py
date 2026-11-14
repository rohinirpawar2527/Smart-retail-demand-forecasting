"""
Comprehensive End-to-End Functionality Audit & Verification Suite.

Automated verification covering:
1. Dashboard Pages & Component Rendering
2. Data Upload & Validation Integrity
3. Machine Learning Training, Zero Leakage & Metrics
4. Demand Forecasting Engine & Multistep Horizons
5. Inventory Health & Stockout Early-Warning Logic
6. System Error Handling & Edge Cases
"""

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.dashboard_components import (
    CATEGORY_COLOR_MAP,
    PLOTLY_CONFIG,
    compute_inventory_health,
    create_actual_vs_predicted_scatter,
    create_category_donut_chart,
    create_category_promo_lift_chart,
    create_daily_sales_chart,
    create_feature_importance_plot,
    create_forecast_plot,
    create_inventory_risk_donut,
    create_model_comparison_bar,
    create_monthly_sales_chart,
    create_product_wise_demand_chart,
    create_store_performance_chart,
    create_top_at_risk_inventory_bar,
    create_top_products_bar_chart,
    load_comparison_results,
    load_default_dataset,
    load_model_bundle,
    load_test_predictions,
)
from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_feature_dataset,
    create_lag_features,
    create_rolling_features,
)
from src.predict import (
    forecast_demand_recursive,
    get_available_entities,
    get_series_history,
)
from src.preprocessing import (
    check_data_quality,
    engineer_date_features,
    load_raw_data,
    validate_schema,
)
from src.train_model import calculate_metrics, chronological_train_test_split


def test_section_1_dashboard():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 1: DASHBOARD PAGES & KPI VERIFICATION")
    print("=" * 70)

    df = load_default_dataset()
    assert len(df) == 73100, f"Expected 73,100 records, got {len(df)}"

    # KPI values verification
    total_units = int(df["Units_Sold"].sum())
    total_revenue = float(df["Revenue"].sum())
    total_stores = df["Store_ID"].nunique()
    total_products = df["Product_ID"].nunique()

    assert total_units == 2475169, f"KPI Units Sold mismatch: {total_units}"
    assert round(total_revenue, 2) == 13950066.42, f"KPI Revenue mismatch: {total_revenue}"
    assert total_stores == 5, f"KPI Stores mismatch: {total_stores}"
    assert total_products == 20, f"KPI Products mismatch: {total_products}"

    print(f" [PASS] Total Units Sold: {total_units:,} (Verified against dataset)")
    print(f" [PASS] Total Gross Revenue: ${total_revenue:,.2f} (Verified against dataset)")
    print(f" [PASS] Store Footprint: {total_stores} Stores | {total_products} Products")

    # Chart creation & collision-free layout verification
    comp_df = load_comparison_results()
    bundle = load_model_bundle()
    test_preds = load_test_predictions()

    charts = [
        ("Daily Sales Trend", create_daily_sales_chart(df)),
        ("Category Donut", create_category_donut_chart(df)),
        ("Top 10 Products", create_top_products_bar_chart(df, n=10)),
        ("Monthly Sales", create_monthly_sales_chart(df)),
        ("Product Demand", create_product_wise_demand_chart(df)),
        ("Promo Lift", create_category_promo_lift_chart(df)),
        ("Store Performance", create_store_performance_chart(df)),
        ("Model Comparison", create_model_comparison_bar(comp_df)),
        ("Feature Importance", create_feature_importance_plot(bundle["model"], bundle["preprocessor"])),
        ("Actual vs Predicted", create_actual_vs_predicted_scatter(test_preds, sample_size=300)),
    ]

    for name, chart in charts:
        assert chart is not None, f"Chart '{name}' failed to build."
        layout = chart.layout
        assert layout.height >= 400, f"Chart '{name}' height {layout.height} < 400px"
        assert layout.margin.t >= 60, f"Chart '{name}' top margin {layout.margin.t} < 60px (Risk of title collision)"
        print(f" [PASS] Chart '{name}': Height={layout.height}px, TopMargin={layout.margin.t}px (Collision-Free)")


def test_section_2_data_upload():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 2: DATA UPLOAD & SCHEMA INTEGRITY AUDIT")
    print("=" * 70)

    # 1. Valid CSV Load & Quality
    raw_df = load_raw_data("data/sample_retail_data.csv")
    assert validate_schema(raw_df) is True
    clean_df, metrics = check_data_quality(raw_df)
    assert metrics["total_missing"] == 0
    assert metrics["duplicate_count"] == 0
    print(" [PASS] Valid CSV Schema and Integrity Check Passed.")

    # 2. Missing Columns Rejection
    bad_df_missing = raw_df.drop(columns=["Price"])
    try:
        validate_schema(bad_df_missing)
        assert False, "Failed to catch missing 'Price' column!"
    except ValueError as e:
        print(f" [PASS] Correctly rejected dataset with missing column: {e}")

    # 3. Negative Price Rejection
    bad_df_price = raw_df.copy()
    bad_df_price.loc[0, "Price"] = -5.0
    try:
        check_data_quality(bad_df_price)
        assert False, "Failed to catch negative price!"
    except ValueError as e:
        print(f" [PASS] Correctly rejected dataset with negative price: {e}")

    # 4. Negative Units Sold Rejection
    bad_df_sales = raw_df.copy()
    bad_df_sales.loc[0, "Units_Sold"] = -10
    try:
        check_data_quality(bad_df_sales)
        assert False, "Failed to catch negative Units_Sold!"
    except ValueError as e:
        print(f" [PASS] Correctly rejected dataset with negative Units_Sold: {e}")

    # 5. Invalid Promotion Flag Rejection
    bad_df_promo = raw_df.copy()
    bad_df_promo.loc[0, "Promotion"] = 5
    try:
        check_data_quality(bad_df_promo)
        assert False, "Failed to catch invalid Promotion flag!"
    except ValueError as e:
        print(f" [PASS] Correctly rejected dataset with non-binary Promotion flag: {e}")


def test_section_3_machine_learning():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 3: MACHINE LEARNING, ZERO-LEAKAGE & METRICS")
    print("=" * 70)

    # 1. Model Bundle Verification
    bundle = load_model_bundle()
    assert bundle["model_name"] == "XGBoost Regressor"
    assert "preprocessor" in bundle
    assert "feature_columns" in bundle
    print(" [PASS] Serialized Winning Model Bundle Verified (XGBoost Regressor).")

    # 2. Benchmark Comparison Table Verification
    comp_df = load_comparison_results()
    assert len(comp_df) == 3
    models_evaluated = comp_df["Model"].tolist()
    assert "Linear Regression" in models_evaluated
    assert "Random Forest Regressor" in models_evaluated
    assert "XGBoost Regressor" in models_evaluated

    xgb_metrics = comp_df[comp_df["Model"] == "XGBoost Regressor"].iloc[0]
    assert xgb_metrics["Test_WAPE(%)"] == 14.70, f"Expected 14.70% WAPE, got {xgb_metrics['Test_WAPE(%)']}"
    assert xgb_metrics["Test_MAE"] == 5.686, f"Expected 5.686 MAE, got {xgb_metrics['Test_MAE']}"
    assert xgb_metrics["Test_R2"] == 0.8791, f"Expected 0.8791 R2, got {xgb_metrics['Test_R2']}"
    print(f" [PASS] Model Comparison Table Verified (XGBoost: WAPE={xgb_metrics['Test_WAPE(%)']}%, MAE={xgb_metrics['Test_MAE']}).")

    # 3. Mathematical Metric Function Verification
    y_true = np.array([100.0, 50.0, 20.0, 0.0])
    y_pred = np.array([90.0, 55.0, 25.0, 5.0])
    calc = calculate_metrics(y_true, y_pred)

    expected_mae = (10 + 5 + 5 + 5) / 4.0  # 6.25
    expected_wape = (25.0 / 170.0) * 100.0  # 14.7058%
    assert abs(calc["MAE"] - expected_mae) < 1e-2
    assert abs(calc["WAPE(%)"] - expected_wape) < 1e-1
    print(f" [PASS] Metric Calculation Verified: MAE={calc['MAE']}, WAPE={calc['WAPE(%)']}%.")

    # 4. Strict Chronological Train/Test Split Verification
    df = load_default_dataset()
    fdf = build_feature_dataset(df, drop_na_lags=True)
    train_df, test_df = chronological_train_test_split(fdf, split_date="2025-10-01")

    assert train_df["Date"].max() < pd.to_datetime("2025-10-01")
    assert test_df["Date"].min() >= pd.to_datetime("2025-10-01")
    assert len(train_df) == 61100
    assert len(test_df) == 9200
    print(f" [PASS] Chronological Split Verified: Train={len(train_df):,} rows, Test={len(test_df):,} rows.")

    # 5. Formal Data Leakage Proof Test
    # Verify that changing day t's sales has ZERO effect on day t's lag or rolling features
    sample_series = df[(df["Store_ID"] == "STORE_001") & (df["Product_ID"] == "PROD_001")].copy()
    sample_series = sample_series.sort_values("Date").reset_index(drop=True)

    # Compute baseline rolling features
    base_feats = create_rolling_features(create_lag_features(sample_series))

    # Artificially spike day 50 sales by 10,000 units
    tampered_series = sample_series.copy()
    tampered_series.loc[50, "Units_Sold"] = 10000

    tampered_feats = create_rolling_features(create_lag_features(tampered_series))

    # Day 50 features MUST remain 100% IDENTICAL because day 50 sales cannot leak into day 50 features!
    assert base_feats.loc[50, "Lag_1"] == tampered_feats.loc[50, "Lag_1"]
    assert base_feats.loc[50, "Rolling_Mean_7"] == tampered_feats.loc[50, "Rolling_Mean_7"]
    assert base_feats.loc[50, "Rolling_Mean_28"] == tampered_feats.loc[50, "Rolling_Mean_28"]
    print(" [PASS] Data Leakage Verification Passed: Zero lookahead bias detected in rolling statistics.")


def test_section_4_demand_forecasting():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 4: DEMAND FORECASTING & MULTI-STEP HORIZONS")
    print("=" * 70)

    stores, products, _ = get_available_entities()

    # 1. Product & Store Selection Verification
    assert "STORE_001" in stores
    assert "PROD_001" in products

    # 2. Horizons: 7, 14, 30 Days Verification
    for horizon in [7, 14, 30]:
        fc = forecast_demand_recursive("STORE_001", "PROD_001", forecast_days=horizon)
        assert len(fc) == horizon, f"Expected {horizon} days, got {len(fc)}"

        # Check non-negative constraint
        assert (fc["Forecasted_Demand"] >= 0.0).all(), "Negative demand found!"
        assert (fc["Rounded_Demand"] >= 0).all(), "Negative rounded units found!"

        # Check chronological continuity of future dates
        dates = pd.to_datetime(fc["Date"]).tolist()
        for i in range(1, len(dates)):
            diff_days = (dates[i] - dates[i - 1]).days
            assert diff_days == 1, f"Forecast dates not strictly consecutive at step {i}!"

        print(f" [PASS] Horizon {horizon:2d} Days: {len(fc)} consecutive daily forecasts. Range: {fc['Date'].min()} to {fc['Date'].max()}. (Zero negatives).")

    # 3. Planned Promotion Lift Verification
    promo_days = {"2026-01-05": 1, "2026-01-06": 1}
    fc_promo = forecast_demand_recursive("STORE_001", "PROD_001", forecast_days=14, planned_promotions=promo_days)
    p_demand = fc_promo[fc_promo["Date"] == "2026-01-05"]["Forecasted_Demand"].values[0]

    fc_normal = forecast_demand_recursive("STORE_001", "PROD_001", forecast_days=14)
    n_demand = fc_normal[fc_normal["Date"] == "2026-01-05"]["Forecasted_Demand"].values[0]

    assert p_demand > n_demand, "Promotional lift was not reflected in forecast!"
    print(f" [PASS] Promotional Lift Verified: Day with promo = {p_demand:.1f} units vs regular = {n_demand:.1f} units.")

    # 4. CSV Download Generation Verification
    buf = io.StringIO()
    fc.to_csv(buf, index=False)
    csv_str = buf.getvalue()
    assert len(csv_str) > 100
    assert "Forecasted_Demand" in csv_str
    print(" [PASS] Forecast CSV Export Buffer Verified.")


def test_section_5_inventory_insights():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 5: INVENTORY INSIGHTS & REORDER LOGIC")
    print("=" * 70)

    df = load_default_dataset()
    inv_df = compute_inventory_health(df, lead_time_days=2, safety_stock_days=3, target_days=7)

    # 1. Series count
    assert len(inv_df) == 100, f"Expected 100 series, got {len(inv_df)}"

    # 2. Required columns
    required_inv_cols = [
        "Store_ID", "Product_ID", "Category", "Price", "Current_Stock",
        "Avg_Daily_Demand", "Days_of_Supply", "Reorder_Point", "Target_Stock",
        "Suggested_Reorder", "Stock_Status", "Est_Reorder_Cost"
    ]
    for col in required_inv_cols:
        assert col in inv_df.columns, f"Missing inventory column: {col}"

    # 3. Supply chain logic integrity
    # Days of Supply = Current_Stock / Avg_Daily_Demand
    expected_dos = (inv_df["Current_Stock"] / inv_df["Avg_Daily_Demand"]).round(1)
    assert np.allclose(inv_df["Days_of_Supply"], expected_dos, atol=0.1)

    # Reorder Point = Avg_Daily_Demand * (2 + 3)
    expected_rop = (inv_df["Avg_Daily_Demand"] * 5).round(1)
    assert np.allclose(inv_df["Reorder_Point"], expected_rop, atol=0.1)

    # Critical risk triggers when Days of Supply <= lead time (2 days)
    crit_items = inv_df[inv_df["Stock_Status"] == "CRITICAL RISK"]
    assert (crit_items["Days_of_Supply"] <= 2.0).all(), "Critical status condition violated!"

    print(f" [PASS] Inventory Health Computed for {len(inv_df)} SKUs.")
    print(f" [PASS] Supply Chain Logic Verified (Days of Supply & ROP).")
    print(f" [PASS] Stockout Early-Warning Status Breakdown: {inv_df['Stock_Status'].value_counts().to_dict()}")


def test_section_6_error_handling():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 6: SYSTEM ERROR HANDLING & BOUNDARY CHECKS")
    print("=" * 70)

    # 1. Missing Model File Handling
    try:
        load_model_bundle("models/non_existent_model.joblib")
        assert False, "Failed to catch missing model file!"
    except FileNotFoundError as e:
        print(f" [PASS] Missing model file handled cleanly: FileNotFoundError")

    # 2. Invalid Store ID Handling
    try:
        forecast_demand_recursive("STORE_999", "PROD_001")
        assert False, "Failed to catch invalid store ID!"
    except ValueError as e:
        print(f" [PASS] Invalid Store ID handled cleanly: ValueError")

    # 3. Invalid Product ID Handling
    try:
        forecast_demand_recursive("STORE_001", "PROD_999")
        assert False, "Failed to catch invalid product ID!"
    except ValueError as e:
        print(f" [PASS] Invalid Product ID handled cleanly: ValueError")

    # 4. Invalid Horizon Handling
    try:
        forecast_demand_recursive("STORE_001", "PROD_001", forecast_days=-5)
        assert False, "Failed to catch non-positive forecast horizon!"
    except ValueError as e:
        print(f" [PASS] Invalid horizon handled cleanly: ValueError")

    # 5. In-Memory Custom DataFrame Forecasting
    df = load_default_dataset()
    custom_slice = df[df["Date"] >= "2025-06-01"].copy()
    fc_custom = forecast_demand_recursive(
        store_id="STORE_002",
        product_id="PROD_005",
        forecast_days=7,
        data_source=custom_slice,
    )
    assert len(fc_custom) == 7
    print(" [PASS] In-Memory Custom Dataset Forecasting Verified.")


def main():
    print("*" * 70)
    print("STARTING FULL SYSTEM FUNCTIONALITY AUDIT & VERIFICATION")
    print("*" * 70)

    test_section_1_dashboard()
    test_section_2_data_upload()
    test_section_3_machine_learning()
    test_section_4_demand_forecasting()
    test_section_5_inventory_insights()
    test_section_6_error_handling()

    print("\n" + "*" * 70)
    print("ALL 6 AUDIT SECTIONS PASSED WITH 100% SUCCESS RATE!")
    print("*" * 70)


if __name__ == "__main__":
    main()
