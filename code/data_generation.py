"""
Dataset generation and cleaning for the Digital Twin framework.
Generates synthetic 2-year daily retail demand data with realistic patterns.
"""

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED, WAREHOUSES, PRODUCTS, HIST_START_DATE, N_HIST_DAYS,
    BASE_DEMAND, WAREHOUSE_MULTIPLIER, SELLING_PRICE, PURCHASE_COST,
    SUPPLIER_OF_PRODUCT, SUPPLIER_LEAD_TIME, HOLDING_COST_RATE, SHIPPING_COST_RATE,
)


def generate_historical_data():
    """
    Generate 2 years of synthetic daily retail data for all warehouse/product combos.
    
    Demand includes weekly seasonality, monthly seasonality, upward trend,
    random promotions, and noise — a realistic pattern for retail demand.
    
    Returns:
        pd.DataFrame: Historical dataset with columns Date, Warehouse, Product,
                      Demand, Inventory, costs, revenue, etc.
    """
    hist_dates = pd.date_range(HIST_START_DATE, periods=N_HIST_DAYS, freq="D")
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for wh in WAREHOUSES:
        for p in PRODUCTS:
            inventory_level = 500  # simple illustrative starting stock for the historical record
            for i, d in enumerate(hist_dates):
                dow, month = d.dayofweek, d.month
                weekly_seasonality = 1 + 0.25 * np.sin(2 * np.pi * dow / 7)
                monthly_seasonality = 1 + 0.15 * np.sin(2 * np.pi * month / 12)
                trend = 1 + 0.0003 * i
                promo = 1.4 if rng.random() < 0.03 else 1.0
                noise = rng.normal(1.0, 0.08)
                demand = max(0, BASE_DEMAND[p] * WAREHOUSE_MULTIPLIER[wh]
                             * weekly_seasonality * monthly_seasonality * trend * promo * noise)
                demand = int(round(demand))
                units_sold = min(demand, inventory_level)
                inventory_level -= units_sold
                order_qty = 0
                if inventory_level < 150:          # simple historical reorder rule
                    order_qty = 400
                    inventory_level += order_qty
                revenue = units_sold * SELLING_PRICE[p]
                rows.append({
                    "Date": d, "Warehouse": wh, "Product": p,
                    "Demand": demand, "Inventory": inventory_level,
                    "Selling_Price": SELLING_PRICE[p], "Purchase_Cost": PURCHASE_COST[p],
                    "Holding_Cost": round(inventory_level * HOLDING_COST_RATE, 2),
                    "Shipping_Cost": round(units_sold * SHIPPING_COST_RATE, 2),
                    "Supplier": SUPPLIER_OF_PRODUCT[p],
                    "Lead_Time": SUPPLIER_LEAD_TIME[SUPPLIER_OF_PRODUCT[p]],
                    "Order_Quantity": order_qty,
                    "Revenue": round(revenue, 2),
                })
    return pd.DataFrame(rows)


def clean_data(raw_df):
    """
    Clean the dataset by injecting and imputing missing values.
    
    Deliberately injects a small fraction of missing demand values to
    demonstrate a realistic cleaning step, then imputes them using the
    median demand for the same Warehouse/Product series.
    
    Args:
        raw_df: Raw DataFrame from generate_historical_data()
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values.
    """
    df = raw_df.copy()
    missing_idx = df.sample(frac=0.005, random_state=RANDOM_SEED).index
    df.loc[missing_idx, "Demand"] = np.nan
    print(f"Injected {len(missing_idx)} missing Demand values for demonstration.")
    print(f"Missing values before cleaning:\n{df.isna().sum()[df.isna().sum() > 0]}")

    df["Demand"] = df.groupby(["Warehouse", "Product"])["Demand"] \
                     .transform(lambda s: s.fillna(s.median()))
    df["Demand"] = df["Demand"].round().astype(int)
    assert df.isna().sum().sum() == 0, "Missing values remain after cleaning!"
    print("All missing values imputed. Dataset is clean.")
    return df
