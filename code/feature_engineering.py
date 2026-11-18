"""
Feature engineering for demand forecasting.
Builds calendar, lag, rolling-average, and categorical features.
"""

import pandas as pd


def engineer_features(df):
    """
    Add calendar, lag, rolling-window, and encoded categorical features.
    
    Args:
        df: Cleaned DataFrame with Date, Warehouse, Product, Demand columns.
    
    Returns:
        tuple: (df_with_features, warehouse_categories, product_categories)
    """
    df = df.sort_values(["Warehouse", "Product", "Date"]).reset_index(drop=True)

    # Calendar features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["DayIndex"] = (df["Date"] - df["Date"].min()).dt.days  # trend proxy

    # Lag features
    for lag in [1, 7, 14]:
        df[f"Demand_Lag_{lag}"] = df.groupby(["Warehouse", "Product"])["Demand"].shift(lag)

    # Rolling-window features
    df["Demand_RollMean_7"] = df.groupby(["Warehouse", "Product"])["Demand"] \
                                 .transform(lambda s: s.shift(1).rolling(7).mean())
    df["Demand_RollMean_28"] = df.groupby(["Warehouse", "Product"])["Demand"] \
                                  .transform(lambda s: s.shift(1).rolling(28).mean())

    # Categorical encoding
    df["WarehouseCode"] = df["Warehouse"].astype("category").cat.codes
    df["ProductCode"] = df["Product"].astype("category").cat.codes
    warehouse_categories = dict(enumerate(df["Warehouse"].astype("category").cat.categories))
    product_categories = dict(enumerate(df["Product"].astype("category").cat.categories))

    print("Feature engineering complete. New columns added:")
    original_cols = ["Date", "Warehouse", "Product", "Demand", "Inventory",
                     "Selling_Price", "Purchase_Cost", "Holding_Cost", "Shipping_Cost",
                     "Supplier", "Lead_Time", "Order_Quantity", "Revenue"]
    print([c for c in df.columns if c not in original_cols])

    return df, warehouse_categories, product_categories
