"""
Configuration and constants for the Predictive Business Digital Twin Framework.
All project-wide constants, hyperparameters, and reproducibility settings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# REPRODUCIBILITY
# ================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.set_option("display.max_columns", 50)
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# ================================================================
# BUSINESS CONSTANTS
# ================================================================
WAREHOUSES = ["Warehouse_A", "Warehouse_B"]
PRODUCTS = ["Product_1", "Product_2", "Product_3"]

HIST_START_DATE = pd.Timestamp("2024-01-01")
N_HIST_DAYS = 730  # 2 years of daily history

BASE_DEMAND = {"Product_1": 60, "Product_2": 40, "Product_3": 25}
WAREHOUSE_MULTIPLIER = {"Warehouse_A": 1.20, "Warehouse_B": 0.85}

SELLING_PRICE = {"Product_1": 25.0, "Product_2": 45.0, "Product_3": 80.0}
PURCHASE_COST = {"Product_1": 14.0, "Product_2": 27.0, "Product_3": 50.0}

SUPPLIER_OF_PRODUCT = {"Product_1": "Supplier_X", "Product_2": "Supplier_Y", "Product_3": "Supplier_Z"}
SUPPLIER_LEAD_TIME = {"Supplier_X": 3, "Supplier_Y": 5, "Supplier_Z": 7}

HOLDING_COST_RATE = 0.02   # cost per unit held per day
SHIPPING_COST_RATE = 1.5   # cost per unit shipped

# ================================================================
# MODEL CONFIGURATION
# ================================================================
FEATURES = ["DayOfWeek", "Month", "WeekOfYear", "IsWeekend", "DayIndex",
            "Demand_Lag_1", "Demand_Lag_7", "Demand_Lag_14",
            "Demand_RollMean_7", "Demand_RollMean_28",
            "WarehouseCode", "ProductCode"]
TARGET = "Demand"

# ================================================================
# SIMULATION CONFIGURATION
# ================================================================
SIMULATION_HORIZON_DAYS = 60
LOOKAHEAD_DAYS = 7   # how many days ahead the transfer policy looks at demand

# ================================================================
# LIBRARY AVAILABILITY FLAGS
# ================================================================
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    print("Plotly not available in this environment — Plotly charts will be skipped, "
          "Matplotlib equivalents are provided for every figure.")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("xgboost not found — falling back to RandomForestRegressor as the "
          "demand forecasting model. Run '!pip install xgboost' in Colab to "
          "use XGBoost instead.")
