"""
Synthetic Retail Sales Dataset Generator for Smart Retail Demand Forecasting.

NOTE: This script generates synthetic demonstration data designed to simulate
realistic retail patterns (weekly seasonality, weekend surges, monthly trends,
promotional price discounts and demand lifts, holiday spikes, store-level differences,
and dynamic inventory replenishment). It is intended solely for demonstration,
prototyping, and model development in this project.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_retail_dataset(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic retail sales dataset over a 2-year period.

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    random_seed : int
        Seed for NumPy random generator to ensure reproducibility.

    Returns:
    --------
    pd.DataFrame
        Synthetic retail dataset with daily sales, pricing, and inventory.
    """
    np.random.seed(random_seed)

    # 1. Date Range & Calendar
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(dates)

    # 2. Stores Definition (5 Stores with varying footfall multipliers)
    stores = [
        {"Store_ID": "STORE_001", "store_mult": 1.25},  # Downtown Flagship
        {"Store_ID": "STORE_002", "store_mult": 1.15},  # Suburban Supercenter
        {"Store_ID": "STORE_003", "store_mult": 0.90},  # Urban Express
        {"Store_ID": "STORE_004", "store_mult": 1.05},  # Regional Mall Outlet
        {"Store_ID": "STORE_005", "store_mult": 0.80},  # Community Market
    ]

    # 3. Products Definition (20 Products across 5 Categories)
    products = [
        {"Product_ID": "PROD_001", "Category": "Dairy & Bakery", "base_price": 3.49, "base_demand": 48},
        {"Product_ID": "PROD_002", "Category": "Dairy & Bakery", "base_price": 4.29, "base_demand": 38},
        {"Product_ID": "PROD_003", "Category": "Dairy & Bakery", "base_price": 4.99, "base_demand": 26},
        {"Product_ID": "PROD_004", "Category": "Dairy & Bakery", "base_price": 5.49, "base_demand": 22},
        {"Product_ID": "PROD_005", "Category": "Grocery", "base_price": 9.99, "base_demand": 25},
        {"Product_ID": "PROD_006", "Category": "Grocery", "base_price": 12.99, "base_demand": 18},
        {"Product_ID": "PROD_007", "Category": "Grocery", "base_price": 2.29, "base_demand": 42},
        {"Product_ID": "PROD_008", "Category": "Grocery", "base_price": 2.79, "base_demand": 34},
        {"Product_ID": "PROD_009", "Category": "Grocery", "base_price": 4.49, "base_demand": 52},
        {"Product_ID": "PROD_010", "Category": "Beverages", "base_price": 5.99, "base_demand": 32},
        {"Product_ID": "PROD_011", "Category": "Beverages", "base_price": 3.99, "base_demand": 36},
        {"Product_ID": "PROD_012", "Category": "Beverages", "base_price": 10.99, "base_demand": 24},
        {"Product_ID": "PROD_013", "Category": "Beverages", "base_price": 6.49, "base_demand": 20},
        {"Product_ID": "PROD_014", "Category": "Personal Care", "base_price": 8.49, "base_demand": 16},
        {"Product_ID": "PROD_015", "Category": "Personal Care", "base_price": 7.99, "base_demand": 15},
        {"Product_ID": "PROD_016", "Category": "Personal Care", "base_price": 3.99, "base_demand": 28},
        {"Product_ID": "PROD_017", "Category": "Home & Kitchen", "base_price": 3.49, "base_demand": 27},
        {"Product_ID": "PROD_018", "Category": "Home & Kitchen", "base_price": 6.99, "base_demand": 14},
        {"Product_ID": "PROD_019", "Category": "Home & Kitchen", "base_price": 19.99, "base_demand": 9},
        {"Product_ID": "PROD_020", "Category": "Home & Kitchen", "base_price": 29.99, "base_demand": 6},
    ]

    # 4. Calendar Multipliers & Holidays
    # Day-of-week multipliers (0: Mon, 6: Sun) - Retail surges on Friday & Weekends
    dow_multipliers = {
        0: 0.88,  # Monday
        1: 0.85,  # Tuesday
        2: 0.90,  # Wednesday
        3: 0.96,  # Thursday
        4: 1.18,  # Friday
        5: 1.38,  # Saturday
        6: 1.25,  # Sunday
    }

    # Month seasonality multipliers (1: Jan, 12: Dec)
    month_multipliers = {
        1: 0.92,
        2: 0.90,
        3: 0.95,
        4: 0.97,
        5: 1.02,
        6: 1.05,
        7: 1.08,
        8: 1.04,
        9: 0.98,
        10: 1.02,
        11: 1.15,
        12: 1.25,
    }

    # Major retail holidays across 2024 and 2025
    holiday_dates = {
        # 2024
        "2024-01-01", "2024-01-15", "2024-02-14", "2024-03-31", "2024-05-12",
        "2024-05-27", "2024-07-04", "2024-09-02", "2024-10-31", "2024-11-28",
        "2024-11-29", "2024-11-30", "2024-12-24", "2024-12-25", "2024-12-31",
        # 2025
        "2025-01-01", "2025-01-20", "2025-02-14", "2025-04-20", "2025-05-11",
        "2025-05-26", "2025-07-04", "2025-09-01", "2025-10-31", "2025-11-27",
        "2025-11-28", "2025-11-29", "2025-12-24", "2025-12-25", "2025-12-31",
    }

    date_strings = dates.strftime("%Y-%m-%d").tolist()
    is_holiday = [1 if d in holiday_dates else 0 for d in date_strings]

    records = []

    # 5. Generate Simulation by Store and Product
    for store in stores:
        store_id = store["Store_ID"]
        s_mult = store["store_mult"]

        for prod in products:
            prod_id = prod["Product_ID"]
            category = prod["Category"]
            base_price = prod["base_price"]
            base_demand = prod["base_demand"]

            # Category-specific seasonal nuance (e.g. beverages higher in summer)
            is_beverage = (category == "Beverages")
            is_grocery = (category in ["Grocery", "Dairy & Bakery"])

            # Generate promotional schedule (campaigns of 3 to 6 days length)
            promo_array = np.zeros(total_days, dtype=int)
            idx = int(np.random.randint(5, 15))
            while idx < total_days - 7:
                campaign_length = int(np.random.randint(3, 7))
                promo_array[idx : min(idx + campaign_length, total_days)] = 1
                # Next promotion in 12 to 28 days
                idx += campaign_length + int(np.random.randint(12, 28))

            # Inventory simulation parameters
            avg_daily = base_demand * s_mult
            reorder_point = int(avg_daily * 2.5)
            target_stock = int(avg_daily * 6.0)
            order_qty = target_stock - reorder_point
            current_inventory = int(avg_daily * 5.0) + int(np.random.randint(-5, 10))
            pending_orders = []  # list of (arrival_day_index, quantity)

            for day_idx, current_date in enumerate(dates):
                # Process incoming deliveries
                deliveries_today = [qty for arr_day, qty in pending_orders if arr_day == day_idx]
                if deliveries_today:
                    current_inventory += sum(deliveries_today)
                    pending_orders = [(arr_day, qty) for arr_day, qty in pending_orders if arr_day > day_idx]

                dow = current_date.day_of_week
                month = current_date.month
                holiday_flag = is_holiday[day_idx]
                promo_flag = int(promo_array[day_idx])

                # Multipliers
                dow_mult = dow_multipliers[dow]
                month_mult = month_multipliers[month]

                # Seasonal adjustments by category
                if is_beverage and month in [6, 7, 8]:
                    month_mult *= 1.15
                elif is_grocery and month in [11, 12]:
                    month_mult *= 1.10

                # Subtle overall retail trend (~3% over 2 years)
                trend_mult = 1.0 + 0.03 * (day_idx / total_days)

                # Holiday multiplier
                holiday_mult = 1.0
                if holiday_flag:
                    holiday_mult = 1.45 if is_grocery else 1.30

                # Promotional price & lift
                if promo_flag:
                    discount_rate = np.random.uniform(0.12, 0.22)
                    price = round(base_price * (1.0 - discount_rate), 2)
                    promo_mult = 1.50
                else:
                    price = base_price
                    promo_mult = 1.0

                # Expected demand
                expected_demand = (
                    base_demand
                    * s_mult
                    * dow_mult
                    * month_mult
                    * trend_mult
                    * promo_mult
                    * holiday_mult
                )

                # Generate integer demand using Poisson distribution
                raw_demand = int(np.random.poisson(lam=max(0.5, expected_demand)))

                # Sales constrained by available inventory
                units_sold = min(raw_demand, current_inventory)

                # Inventory Level at end of day
                current_inventory -= units_sold
                inventory_level = current_inventory

                # Reorder logic: schedule delivery in 1-2 days if stock drops below reorder point
                if current_inventory <= reorder_point and not pending_orders:
                    lead_time = int(np.random.choice([1, 2], p=[0.7, 0.3]))
                    pending_orders.append((day_idx + lead_time, order_qty))

                records.append({
                    "Date": date_strings[day_idx],
                    "Store_ID": store_id,
                    "Product_ID": prod_id,
                    "Category": category,
                    "Price": price,
                    "Units_Sold": units_sold,
                    "Promotion": promo_flag,
                    "Inventory_Level": max(0, inventory_level),
                    "Holiday": holiday_flag,
                })

    df = pd.DataFrame(records)

    # Ensure required column order
    column_order = [
        "Date",
        "Store_ID",
        "Product_ID",
        "Category",
        "Price",
        "Units_Sold",
        "Promotion",
        "Inventory_Level",
        "Holiday",
    ]
    df = df[column_order]

    return df


def main():
    print("=" * 70)
    print("SMART RETAIL DEMAND FORECASTING - SYNTHETIC DATA GENERATOR")
    print("=" * 70)
    print("NOTE: Generating synthetic demonstration data for development & validation.\n")

    # Generate data
    df = generate_retail_dataset(
        start_date="2024-01-01",
        end_date="2025-12-31",
        random_seed=42,
    )

    # Output path
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sample_retail_data.csv"

    # Save CSV
    df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Dataset successfully saved to: {output_file.as_posix()}\n")

    # 1. Dataset Shape
    print("-" * 50)
    print("1. DATASET SHAPE:")
    print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

    # 2. First Five Rows
    print("\n" + "-" * 50)
    print("2. FIRST FIVE ROWS:")
    print(df.head(5).to_string(index=False))

    # 3. Column Names
    print("\n" + "-" * 50)
    print("3. COLUMN NAMES:")
    for col in df.columns:
        print(f" - {col} ({df[col].dtype})")

    # 4. Missing Values
    print("\n" + "-" * 50)
    print("4. MISSING VALUES PER COLUMN:")
    missing = df.isnull().sum()
    print(missing.to_string())
    print(f"Total missing values across dataset: {missing.sum()}")

    # Check for duplicates
    dup_count = df.duplicated().sum()
    print(f"Duplicate records found: {dup_count}")

    # 5. Basic Statistical Summary
    print("\n" + "-" * 50)
    print("5. BASIC STATISTICAL SUMMARY (NUMERICAL):")
    print(df.describe().round(2).to_string())

    print("\n" + "-" * 50)
    print("CATEGORICAL SUMMARY:")
    print(df[["Store_ID", "Product_ID", "Category"]].describe().to_string())

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
