# Predictive Business Digital Twin Framework

A modular Python framework that builds a **Digital Twin** of a multi-warehouse retail business, uses **ML-based demand forecasting** (XGBoost / Random Forest) to predict future demand, and simulates **inventory management scenarios** to identify the optimal policy.

## Project Structure

```
code/
├── __init__.py              # Package initializer
├── config.py                # Constants, hyperparameters, reproducibility settings
├── data_generation.py       # Synthetic dataset generation & cleaning
├── feature_engineering.py   # Calendar, lag, rolling-window feature engineering
├── forecasting.py           # Demand forecasting model (XGBoost / Random Forest)
├── digital_twin.py          # OOP Digital Twin classes (Supplier, Warehouse, Inventory, etc.)
├── simulation.py            # Day-by-day simulation engine & scenario policies
├── kpi.py                   # KPI calculation & best policy selection
├── visualization.py         # EDA plots & scenario comparison dashboards
├── validation.py            # Simulation integrity checks & assertions
├── main.py                  # Main entry point — runs the complete pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Module Descriptions

| Module | Responsibility |
|---|---|
| `config.py` | All project-wide constants (warehouses, products, prices, costs, model features, seeds) |
| `data_generation.py` | Generates 2 years of synthetic daily retail data with seasonality, trend, promotions, and noise. Includes data cleaning with missing-value imputation |
| `feature_engineering.py` | Builds DayOfWeek, Month, WeekOfYear, IsWeekend, DayIndex, lag (1/7/14), rolling mean (7/28), and categorical encoding features |
| `forecasting.py` | Trains XGBoost (or Random Forest fallback) on 80/20 chronological split. Provides recursive multi-step forecasting for 60-day simulation horizon |
| `digital_twin.py` | Object-oriented business model — `Supplier`, `Transportation`, `Inventory`, `Warehouse`, and `BusinessDigitalTwin` classes with full KPI methods |
| `simulation.py` | Factory for fresh twin instances, demand lookup builder, day-by-day simulation loop, and scenario transfer policies |
| `kpi.py` | Computes Revenue, Profit, Costs, Stockouts, Service Level, Fill Rate, Utilization, Inventory Turnover. Auto-determines best policy |
| `visualization.py` | 5 EDA plots (demand trend, inventory, revenue, monthly seasonality, correlation heatmap) + scenario comparison charts and KPI dashboard |
| `validation.py` | 8 integrity checks per scenario (non-negative inventory, capacity limits, revenue formula, cost formulas, profit formula, etc.) |
| `main.py` | Orchestrates the full pipeline from data generation through validation and conclusion |

## Three Scenarios Simulated

1. **Scenario 1 — Current Inventory Policy**: Baseline reorder-point policy (reorder 700 units when inventory position drops below 400)
2. **Scenario 2 — Inventory Transfer Policy**: Proactive lateral transfers from Warehouse_A → Warehouse_B using 7-day demand lookahead
3. **Scenario 3 — Increased Reorder Quantity**: 1.5× reorder quantity to build larger safety stock buffers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd code
python main.py
```

## Dependencies

- **Required**: numpy, pandas, matplotlib, scikit-learn
- **Optional**: xgboost (preferred forecasting model), plotly (interactive charts)
