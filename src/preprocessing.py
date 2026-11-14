"""
Retail Data Preprocessing Module.

This module loads raw retail sales data, validates schema and data integrity,
checks for missing values, duplicates, and invalid numeric/categorical entries,
engineers calendar and date-based features, and outputs a clean, production-ready
dataset for Exploratory Data Analysis (EDA) and Machine Learning forecasting models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Default Required Columns in raw retail sales dataset
REQUIRED_COLUMNS: List[str] = [
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


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load the raw retail dataset from CSV file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the raw CSV file.

    Returns:
    --------
    pd.DataFrame
        Loaded raw dataset (copy to preserve original).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Retail data file not found at: {path.resolve()}")

    try:
        df = pd.read_csv(path)
        print(f"[LOAD] Successfully loaded {len(df):,} rows from {path.as_posix()}")
        return df.copy()
    except Exception as exc:
        raise IOError(f"Failed to read CSV at {path}: {exc}") from exc


def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """Validate that the dataset contains all expected columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    required_columns : list of str, optional
        List of required column names.

    Returns:
    --------
    bool
        True if all required columns are present.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed! Missing required columns: {missing_cols}")

    print(f"[VALIDATION] Schema check passed ({len(required_columns)} required columns present).")
    return True


def check_data_quality(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
    """Inspect and validate data quality: missing values, duplicates, numeric constraints.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to validate.

    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        Cleaned/validated DataFrame copy and quality audit metrics.
    """
    clean_df = df.copy()
    metrics: Dict[str, Union[int, float]] = {}

    # 1. Missing Values
    missing_counts = clean_df.isnull().sum()
    total_missing = int(missing_counts.sum())
    metrics["total_missing"] = total_missing
    if total_missing > 0:
        print(f"[WARNING] Found {total_missing} missing values across columns:")
        for col, cnt in missing_counts[missing_counts > 0].items():
            print(f"  - {col}: {cnt} missing")
        # Fill or drop if necessary (dataset is synthetic and 0 missing, but for robustness):
        clean_df = clean_df.dropna()
    else:
        print("[QUALITY] Zero missing values detected.")

    # 2. Duplicate Records
    duplicate_count = int(clean_df.duplicated().sum())
    metrics["duplicate_count"] = duplicate_count
    if duplicate_count > 0:
        print(f"[WARNING] Found {duplicate_count} duplicate rows. Removing duplicates...")
        clean_df = clean_df.drop_duplicates()
    else:
        print("[QUALITY] Zero duplicate records detected.")

    # 3. Numeric Column Types & Range Validations
    # Price check
    invalid_prices = clean_df[clean_df["Price"] <= 0]
    metrics["invalid_prices"] = len(invalid_prices)
    if not invalid_prices.empty:
        raise ValueError(f"Found {len(invalid_prices)} records with non-positive Price!")

    # Units Sold check
    invalid_sales = clean_df[clean_df["Units_Sold"] < 0]
    metrics["invalid_sales"] = len(invalid_sales)
    if not invalid_sales.empty:
        raise ValueError(f"Found {len(invalid_sales)} records with negative Units_Sold!")

    # Inventory Level check
    invalid_inventory = clean_df[clean_df["Inventory_Level"] < 0]
    metrics["invalid_inventory"] = len(invalid_inventory)
    if not invalid_inventory.empty:
        raise ValueError(f"Found {len(invalid_inventory)} records with negative Inventory_Level!")

    # 4. Binary Flags Validation
    invalid_promotions = clean_df[~clean_df["Promotion"].isin([0, 1])]
    metrics["invalid_promotions"] = len(invalid_promotions)
    if not invalid_promotions.empty:
        raise ValueError(f"Found {len(invalid_promotions)} records with invalid Promotion flag (must be 0 or 1)!")

    invalid_holidays = clean_df[~clean_df["Holiday"].isin([0, 1])]
    metrics["invalid_holidays"] = len(invalid_holidays)
    if not invalid_holidays.empty:
        raise ValueError(f"Found {len(invalid_holidays)} records with invalid Holiday flag (must be 0 or 1)!")

    print("[QUALITY] All numeric and categorical integrity checks passed successfully.")
    return clean_df, metrics


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date into datetime format and engineer calendar features.

    Features generated:
    - Year: 4-digit calendar year
    - Month: Calendar month (1-12)
    - Day: Day of the month (1-31)
    - Day_of_Week: Name of the day (e.g., 'Monday', 'Tuesday')
    - Day_of_Week_Num: Day of week index (0=Monday, 6=Sunday)
    - Week: ISO calendar week number (1-53)
    - Is_Weekend: Binary flag (1 for Saturday/Sunday, 0 otherwise)
    - Revenue: Calculated retail revenue (Units_Sold * Price)

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with 'Date' column.

    Returns:
    --------
    pd.DataFrame
        DataFrame with enriched date features.
    """
    enriched = df.copy()

    # Convert Date to datetime
    enriched["Date"] = pd.to_datetime(enriched["Date"])

    # Extract date features
    enriched["Year"] = enriched["Date"].dt.year
    enriched["Month"] = enriched["Date"].dt.month
    enriched["Day"] = enriched["Date"].dt.day
    enriched["Day_of_Week"] = enriched["Date"].dt.day_name()
    enriched["Day_of_Week_Num"] = enriched["Date"].dt.day_of_week
    enriched["Week"] = enriched["Date"].dt.isocalendar().week.astype(int)
    enriched["Is_Weekend"] = enriched["Day_of_Week_Num"].isin([5, 6]).astype(int)

    # Calculate Revenue as an additional feature
    enriched["Revenue"] = (enriched["Units_Sold"] * enriched["Price"]).round(2)

    # Format Date back to string ISO format for clean CSV serialization
    enriched["Date"] = enriched["Date"].dt.strftime("%Y-%m-%d")

    print(f"[ENGINEERING] Generated date features: Year, Month, Day, Day_of_Week, Day_of_Week_Num, Week, Is_Weekend, Revenue.")
    return enriched


def preprocess_pipeline(
    input_file: Union[str, Path] = "data/sample_retail_data.csv",
    output_file: Optional[Union[str, Path]] = "data/processed_retail_data.csv",
) -> pd.DataFrame:
    """Run the complete end-to-end preprocessing pipeline.

    Parameters:
    -----------
    input_file : str or Path
        Path to raw retail dataset.
    output_file : str or Path, optional
        Path where the processed dataset will be saved.

    Returns:
    --------
    pd.DataFrame
        Fully preprocessed and feature-engineered DataFrame.
    """
    print("=" * 70)
    print("RETAIL DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_raw_data(input_file)

    # 2. Schema Validation
    validate_schema(raw_df)

    # 3. Data Quality & Constraint Checks
    clean_df, metrics = check_data_quality(raw_df)

    # 4. Feature Engineering
    processed_df = engineer_date_features(clean_df)

    # 5. Save Output
    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(out_path, index=False)
        print(f"[SAVE] Processed dataset saved to: {out_path.as_posix()}")

    print("=" * 70)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 70)

    return processed_df


def main():
    """CLI execution for preprocessing module."""
    input_path = "data/sample_retail_data.csv"
    output_path = "data/processed_retail_data.csv"

    processed_df = preprocess_pipeline(input_path, output_path)

    # Print Summary Information
    print("\n" + "-" * 50)
    print("PROCESSED DATASET OVERVIEW:")
    print(f"Shape: {processed_df.shape[0]:,} rows x {processed_df.shape[1]} columns")
    print("\nColumns:")
    for col in processed_df.columns:
        print(f"  - {col} ({processed_df[col].dtype})")

    print("\nFirst 5 Rows:")
    print(processed_df.head(5).to_string(index=False))

    print("\nSummary Statistics:")
    print(processed_df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
