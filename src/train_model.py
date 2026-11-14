"""
Model Training & Evaluation Module for Smart Retail Demand Forecasting.

This script executes the complete machine learning training pipeline:
1. Loads preprocessed retail sales data and executes feature engineering.
2. Performs a strictly chronological train/test split (never random).
3. Fits feature preprocessors (StandardScaler and OneHotEncoder) strictly on training data.
4. Trains and evaluates three competitive regressors:
   - Linear Regression (Baseline)
   - Random Forest Regressor
   - XGBoost Regressor
5. Evaluates model performance using MAE, RMSE, WAPE (%), and R2 Score.
6. Compares all models in a structured results table.
7. Selects the winning model based on test WAPE/MAE.
8. Serializes the best model bundle using Joblib inside models/.
9. Exports comparison results to models/model_comparison_results.csv.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Ensure project root is in sys.path for robust modular imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.feature_engineering import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
    build_feature_dataset,
)

MODELS_DIR = Path("models")
PROCESSED_DATA_PATH = Path("data/processed_retail_data.csv")
DEFAULT_SPLIT_DATE = "2025-10-01"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression evaluation metrics: MAE, RMSE, WAPE (%), and R2.

    WAPE (Weighted Absolute Percentage Error):
        WAPE = (sum(|y_true - y_pred|) / sum(y_true)) * 100
    This is the retail supply-chain industry standard metric, as it avoids
    division by zero and scales errors proportionally to total demand volume.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth actual sales quantities.
    y_pred : np.ndarray
        Predicted sales quantities.

    Returns:
    --------
    dict
        Dictionary of calculated evaluation metrics.
    """
    # Clip negative predictions to 0 for retail validity
    clipped_pred = np.maximum(0.0, y_pred)

    mae = float(mean_absolute_error(y_true, clipped_pred))
    rmse = float(root_mean_squared_error(y_true, clipped_pred))
    total_actual = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - clipped_pred)) / total_actual) * 100.0) if total_actual > 0 else 0.0
    r2 = float(r2_score(y_true, clipped_pred))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE(%)": round(wape, 2),
        "R2": round(r2, 4),
    }


def chronological_train_test_split(
    df: pd.DataFrame,
    split_date: str = DEFAULT_SPLIT_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological split on the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Engineered dataset with 'Date' column.
    split_date : str
        Cutoff date (e.g., '2025-10-01'). All records strictly before this date
        belong to the training set; all records on or after belong to the test set.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    """
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    cutoff = pd.to_datetime(split_date)
    train_df = df[df["Date"] < cutoff].copy()
    test_df = df[df["Date"] >= cutoff].copy()

    print(f"[SPLIT] Chronological Split Date: {split_date}")
    print(f" - Train period: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df):,} records)")
    print(f" - Test period:  {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')} ({len(test_df):,} records)")

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer pipeline.

    - Standardizes numerical features using StandardScaler.
    - One-hot encodes nominal categoricals (Store_ID, Product_ID, Category).

    Returns:
    --------
    ColumnTransformer
        Unfitted transformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, ColumnTransformer]:
    """Fit preprocessor and train competitive regressors, then compute comparative metrics.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Testing data.

    Returns:
    --------
    Tuple[dict, pd.DataFrame, ColumnTransformer]
        Dictionary of trained model objects, comparison DataFrame, and fitted preprocessor.
    """
    X_train_raw = train_df[FEATURE_COLUMNS]
    y_train = train_df["Units_Sold"].values

    X_test_raw = test_df[FEATURE_COLUMNS]
    y_test = test_df["Units_Sold"].values

    # Fit preprocessor strictly on training data
    print("\n[PREPROCESSING] Fitting ColumnTransformer strictly on training set...")
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    print(f" - Transformed Feature Space: {X_train.shape[1]} features (from {len(FEATURE_COLUMNS)} raw inputs)")

    # Define Candidate Models
    models: Dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=75,
            max_depth=15,
            min_samples_split=6,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("MODEL TRAINING & COMPARATIVE EVALUATION")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        duration = (datetime.now() - start_time).total_seconds()

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, train_preds)
        test_metrics = calculate_metrics(y_test, test_preds)

        print(f"  Training finished in {duration:.2f}s")
        print(f"  Train: MAE={train_metrics['MAE']} | RMSE={train_metrics['RMSE']} | WAPE={train_metrics['WAPE(%)']}% | R2={train_metrics['R2']}")
        print(f"  Test:  MAE={test_metrics['MAE']} | RMSE={test_metrics['RMSE']} | WAPE={test_metrics['WAPE(%)']}% | R2={test_metrics['R2']}")

        results.append({
            "Model": name,
            "Train_MAE": train_metrics["MAE"],
            "Test_MAE": test_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Train_WAPE(%)": train_metrics["WAPE(%)"],
            "Test_WAPE(%)": test_metrics["WAPE(%)"],
            "Train_R2": train_metrics["R2"],
            "Test_R2": test_metrics["R2"],
            "Fit_Time_Sec": round(duration, 2),
        })

    comparison_df = pd.DataFrame(results)
    return models, comparison_df, preprocessor


def run_training_pipeline(
    data_path: Union[str, Path] = PROCESSED_DATA_PATH,
    split_date: str = DEFAULT_SPLIT_DATE,
    models_dir: Union[str, Path] = MODELS_DIR,
) -> Tuple[str, Dict[str, Any]]:
    """Execute the end-to-end training, model selection, and artifact saving pipeline."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SMART RETAIL DEMAND FORECASTING - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    csv_path = Path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed retail data not found at {csv_path}. Run preprocessing.py first!")

    print(f"[DATA LOAD] Loading data from {csv_path.as_posix()}...")
    raw_df = pd.read_csv(csv_path)

    # 2. Feature Engineering
    print("[FEATURE ENG] Generating autoregressive lags and rolling features...")
    featured_df = build_feature_dataset(raw_df, target_col="Units_Sold", drop_na_lags=True)

    # 3. Chronological Train/Test Split
    train_df, test_df = chronological_train_test_split(featured_df, split_date=split_date)

    # 4. Train and Evaluate
    trained_models, comparison_df, preprocessor = train_and_evaluate_models(train_df, test_df)

    # 5. Display Comparison Table
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON RESULTS TABLE")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # 6. Save Comparison Results CSV
    csv_results_path = out_dir / "model_comparison_results.csv"
    comparison_df.to_csv(csv_results_path, index=False)
    print(f"\n[SAVE] Model comparison table saved to: {csv_results_path.as_posix()}")

    # 7. Select Best Model (ranked by lowest Test WAPE)
    best_row = comparison_df.sort_values(by="Test_WAPE(%)").iloc[0]
    best_model_name = str(best_row["Model"])
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 70)
    print(f"WINNING MODEL SELECTED: {best_model_name}")
    print(f"Test WAPE: {best_row['Test_WAPE(%)']}% | Test MAE: {best_row['Test_MAE']} | Test R2: {best_row['Test_R2']}")
    print("=" * 70)

    # 8. Save Model Bundle using Joblib
    model_bundle: Dict[str, Any] = {
        "model": best_model,
        "model_name": best_model_name,
        "preprocessor": preprocessor,
        "feature_columns": FEATURE_COLUMNS,
        "numerical_columns": NUMERICAL_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "best_metrics": best_row.to_dict(),
        "split_date": split_date,
        "trained_at": datetime.now().isoformat(),
    }

    bundle_path = out_dir / "best_forecasting_model.joblib"
    joblib.dump(model_bundle, bundle_path)
    print(f"[SAVE] Serialized winning model bundle saved to: {bundle_path.as_posix()}\n")

    return best_model_name, model_bundle


def main():
    """CLI execution for model training."""
    run_training_pipeline()


if __name__ == "__main__":
    main()
