"""
Demand forecasting model — training, evaluation, and recursive multi-step prediction.
Uses XGBoost when available, falls back to Random Forest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    RANDOM_SEED, FEATURES, TARGET, WAREHOUSES, PRODUCTS,
    XGBOOST_AVAILABLE, SIMULATION_HORIZON_DAYS,
)

if XGBOOST_AVAILABLE:
    from xgboost import XGBRegressor


def train_forecast_model(df):
    """
    Train the demand forecasting model on the historical dataset.
    
    Uses an 80/20 chronological train/test split. Selects XGBoost if available,
    otherwise Random Forest.
    
    Args:
        df: Feature-engineered DataFrame.
    
    Returns:
        tuple: (model, test_df_with_predictions, mae, rmse, r2)
    """
    model_df = df.dropna(subset=FEATURES).sort_values("Date").reset_index(drop=True)
    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx]
    test_df = model_df.iloc[split_idx:]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(f"Training rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"Train period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"Test period:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")

    if XGBOOST_AVAILABLE:
        forecast_model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED,
        )
    else:
        forecast_model = RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        )

    forecast_model.fit(X_train, y_train)
    test_predictions = forecast_model.predict(X_test)
    test_predictions = np.clip(test_predictions, 0, None)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = mean_squared_error(y_test, test_predictions) ** 0.5
    r2 = r2_score(y_test, test_predictions)

    print("\nDemand Forecasting Model Performance (hold-out test set):")
    print(f"  MAE  = {mae:.2f} units")
    print(f"  RMSE = {rmse:.2f} units")
    print(f"  R^2  = {r2:.3f}")

    test_df = test_df.copy()
    test_df["Predicted_Demand"] = test_predictions

    return forecast_model, test_df, mae, rmse, r2


def recursive_forecast(history_df, wh, product, future_dates, model):
    """
    Recursively forecast demand for future dates using the trained model.
    
    For each future day, builds features from the most recent known + already
    predicted demand values, predicts, then rolls forward.
    
    Args:
        history_df: Historical DataFrame with features.
        wh: Warehouse ID string.
        product: Product ID string.
        future_dates: DatetimeIndex of future dates to forecast.
        model: Trained forecasting model.
    
    Returns:
        list: Predicted demand values for each future date.
    """
    series = history_df[(history_df["Warehouse"] == wh) & (history_df["Product"] == product)] \
                .sort_values("Date")
    demand_history = list(series["Demand"].values)  # will be extended with predictions
    wh_code = series["WarehouseCode"].iloc[-1]
    prod_code = series["ProductCode"].iloc[-1]
    day_index_start = series["DayIndex"].iloc[-1]
    predictions = []

    for step, d in enumerate(future_dates, start=1):
        lag_1 = demand_history[-1]
        lag_7 = demand_history[-7] if len(demand_history) >= 7 else np.mean(demand_history)
        lag_14 = demand_history[-14] if len(demand_history) >= 14 else np.mean(demand_history)
        roll_7 = np.mean(demand_history[-7:])
        roll_28 = np.mean(demand_history[-28:]) if len(demand_history) >= 28 else np.mean(demand_history)
        feature_row = pd.DataFrame([{
            "DayOfWeek": d.dayofweek, "Month": d.month,
            "WeekOfYear": int(d.isocalendar().week), "IsWeekend": int(d.dayofweek >= 5),
            "DayIndex": day_index_start + step,
            "Demand_Lag_1": lag_1, "Demand_Lag_7": lag_7, "Demand_Lag_14": lag_14,
            "Demand_RollMean_7": roll_7, "Demand_RollMean_28": roll_28,
            "WarehouseCode": wh_code, "ProductCode": prod_code,
        }])[FEATURES]
        pred = max(0, model.predict(feature_row)[0])
        predictions.append(pred)
        demand_history.append(pred)

    return predictions


def generate_future_demand(df, forecast_model):
    """
    Generate predicted demand for the simulation horizon.
    
    Args:
        df: Feature-engineered historical DataFrame.
        forecast_model: Trained forecasting model.
    
    Returns:
        pd.DataFrame: Predicted demand with columns Date, Warehouse, Product, Predicted_Demand.
    """
    last_hist_date = df["Date"].max()
    future_dates = pd.date_range(last_hist_date + pd.Timedelta(days=1),
                                  periods=SIMULATION_HORIZON_DAYS, freq="D")

    future_records = []
    for wh in WAREHOUSES:
        for p in PRODUCTS:
            preds = recursive_forecast(df, wh, p, future_dates, forecast_model)
            for d, val in zip(future_dates, preds):
                future_records.append({"Date": d, "Warehouse": wh, "Product": p,
                                        "Predicted_Demand": int(round(val))})

    predicted_demand_df = pd.DataFrame(future_records)
    print(f"Generated {len(predicted_demand_df)} predicted-demand rows "
          f"covering {SIMULATION_HORIZON_DAYS} future days for all Warehouse/Product combinations.")
    return predicted_demand_df, future_dates
