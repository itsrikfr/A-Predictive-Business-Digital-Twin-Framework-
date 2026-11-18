"""
Predictive Business Digital Twin Framework — Main Entry Point.

This script orchestrates the complete pipeline:
  1. Generate and clean historical data
  2. Engineer features
  3. Train the demand forecasting model
  4. Run EDA visualizations
  5. Generate future demand predictions
  6. Build and simulate three inventory policy scenarios
  7. Calculate KPIs and determine the best policy
  8. Visualize scenario comparisons
  9. Validate all simulation results
 10. Print the research conclusion

Usage:
    python main.py
"""

import sys
import os

# Ensure the code package is importable when running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    XGBOOST_AVAILABLE, SIMULATION_HORIZON_DAYS, PRODUCTS,
)
from data_generation import generate_historical_data, clean_data
from feature_engineering import engineer_features
from forecasting import train_forecast_model, generate_future_demand
from simulation import (
    build_digital_twin, build_demand_lookup, run_simulation,
    scenario2_transfer_policy,
)
from kpi import build_kpi_table, determine_best_policy
from visualization import run_all_eda, run_all_scenario_plots
from validation import validate_all_scenarios


def main():
    """Run the complete Predictive Business Digital Twin pipeline."""

    print("=" * 60)
    print("  PREDICTIVE BUSINESS DIGITAL TWIN FRAMEWORK")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # PART 1 — Generate and clean historical data
    # ------------------------------------------------------------------
    print("PART 1 — Generating historical dataset...")
    raw_df = generate_historical_data()
    print(f"Historical dataset shape: {raw_df.shape}")
    print(raw_df.head())
    print()

    print("PART 2 — Cleaning data...")
    df = clean_data(raw_df)
    print()

    # ------------------------------------------------------------------
    # PART 2 — Feature engineering
    # ------------------------------------------------------------------
    print("PART 3 — Engineering features...")
    df, warehouse_categories, product_categories = engineer_features(df)
    print()

    # ------------------------------------------------------------------
    # PART 3 — Train and evaluate the forecasting model
    # ------------------------------------------------------------------
    print("PART 4 — Training demand forecasting model...")
    forecast_model, test_df, mae, rmse, r2 = train_forecast_model(df)
    print()

    # ------------------------------------------------------------------
    # PART 4 — Exploratory Data Analysis
    # ------------------------------------------------------------------
    print("PART 5 — Running Exploratory Data Analysis...")
    run_all_eda(df, test_df)
    print()

    # ------------------------------------------------------------------
    # PART 5 — Generate future demand predictions
    # ------------------------------------------------------------------
    print("PART 6 — Generating future demand predictions...")
    predicted_demand_df, future_dates = generate_future_demand(df, forecast_model)
    print(predicted_demand_df.head(10))
    print()

    # ------------------------------------------------------------------
    # PART 6 — Build demand lookup and run scenarios
    # ------------------------------------------------------------------
    print("PART 7 — Building demand lookup...")
    demand_lookup = build_demand_lookup(predicted_demand_df)
    print()

    # Scenario 1 — Current Inventory Policy
    print("PART 8 — Running Scenario 1 (Current Inventory Policy)...")
    twin_scenario_1 = build_digital_twin()
    twin_scenario_1 = run_simulation(twin_scenario_1, demand_lookup, SIMULATION_HORIZON_DAYS,
                                      future_dates, reorder_multiplier=1.0, transfer_policy=None)
    print("Scenario 1 simulation complete.")
    twin_scenario_1.display_state()
    print()

    # Scenario 2 — Inventory Transfer Policy
    print("PART 9 — Running Scenario 2 (Inventory Transfer Policy)...")
    twin_scenario_2 = build_digital_twin()
    twin_scenario_2 = run_simulation(twin_scenario_2, demand_lookup, SIMULATION_HORIZON_DAYS,
                                      future_dates, reorder_multiplier=1.0,
                                      transfer_policy=scenario2_transfer_policy)
    print("Scenario 2 simulation complete.")
    print(f"Number of physical transfers executed: {len(twin_scenario_2.transfer_log)}")
    twin_scenario_2.display_state()
    print()

    # Scenario 3 — Increased Reorder Quantity Policy
    print("PART 10 — Running Scenario 3 (Increased Reorder Quantity)...")
    twin_scenario_3 = build_digital_twin()
    twin_scenario_3 = run_simulation(twin_scenario_3, demand_lookup, SIMULATION_HORIZON_DAYS,
                                      future_dates, reorder_multiplier=1.5, transfer_policy=None)
    print("Scenario 3 simulation complete.")
    twin_scenario_3.display_state()
    print()

    # ------------------------------------------------------------------
    # PART 7 — Calculate KPIs and determine best policy
    # ------------------------------------------------------------------
    scenario_twins = {
        "Scenario 1\n(Current Policy)": twin_scenario_1,
        "Scenario 2\n(Transfer Policy)": twin_scenario_2,
        "Scenario 3\n(Increased Reorder Qty)": twin_scenario_3,
    }

    print("PART 11 — Calculating KPIs...")
    kpi_table = build_kpi_table(scenario_twins)
    kpi_table_display = kpi_table.round(2)
    print(kpi_table_display)
    print()

    print("PART 12 — Determining best policy...")
    recommended, reason = determine_best_policy(kpi_table)
    print()

    # ------------------------------------------------------------------
    # PART 8 — Scenario comparison visualizations
    # ------------------------------------------------------------------
    print("PART 13 — Generating scenario comparison plots...")
    run_all_scenario_plots(scenario_twins, kpi_table)
    print()

    # ------------------------------------------------------------------
    # PART 9 — Validation
    # ------------------------------------------------------------------
    print("PART 14 — Running validation checks...")
    validate_all_scenarios(scenario_twins, kpi_table)
    print()

    # ------------------------------------------------------------------
    # PART 10 — Research Conclusion
    # ------------------------------------------------------------------
    print("PART 15 — Research Conclusion")
    print("=" * 60)
    print(f"The demand forecasting model "
          f"({'XGBoost' if XGBOOST_AVAILABLE else 'Random Forest (fallback)'}) achieved:")
    print(f"  MAE  = {mae:.2f} units")
    print(f"  RMSE = {rmse:.2f} units")
    print(f"  R^2  = {r2:.3f}")
    print()
    print("An R^2 above 0.85 indicates the model explains the large majority of demand "
          "variance, capturing weekly and monthly seasonality, trend, and product/warehouse "
          "effects well enough to be a credible input for downstream inventory simulation. "
          "The MAE, expressed in units, represents the average day-to-day forecasting error "
          "the Digital Twin must absorb through safety stock and reorder-point buffers.")
    print("=" * 60)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
