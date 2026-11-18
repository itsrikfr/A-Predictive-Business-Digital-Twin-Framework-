"""
KPI calculation and scenario comparison.
Computes business KPIs for each scenario and determines the optimal policy.
"""

import numpy as np
import pandas as pd

from config import SELLING_PRICE, PRODUCTS, SHIPPING_COST_RATE, HOLDING_COST_RATE


def compute_kpis(twin, label):
    """
    Compute all business KPIs for a simulated Digital Twin scenario.
    
    Args:
        twin: BusinessDigitalTwin with completed simulation history.
        label: Human-readable scenario label.
    
    Returns:
        dict: KPI values for the scenario.
    """
    h = pd.DataFrame(twin.history)
    ending_inventory = h[h["Day"] == h["Day"].max()]["Inventory"].sum()
    return {
        "Scenario": label,
        "Revenue": twin.calculate_revenue(),
        "Purchase_Cost": twin.calculate_purchase_cost(),
        "Holding_Cost": twin.calculate_holding_cost(),
        "Shipping_Cost": twin.calculate_shipping_cost(),
        "Transfer_Cost": twin.calculate_transfer_cost(),
        "Profit": twin.calculate_profit(),
        "Ending_Inventory": ending_inventory,
        "Avg_Inventory": twin.calculate_average_inventory(),
        "Warehouse_Utilization_%": twin.calculate_utilization() * 100,
        "Stockout_Units": twin.calculate_stockouts(),
        "Service_Level_%": twin.calculate_service_level() * 100,
        "Order_Fill_Rate_%": twin.calculate_order_fill_rate() * 100,
        "Inventory_Turnover": twin.calculate_inventory_turnover(),
    }


def build_kpi_table(scenario_twins):
    """
    Build a KPI comparison table from multiple scenario twins.
    
    Args:
        scenario_twins: dict mapping scenario labels to BusinessDigitalTwin instances.
    
    Returns:
        pd.DataFrame: KPI table indexed by scenario.
    """
    kpi_table = pd.DataFrame([
        compute_kpis(twin, label) for label, twin in scenario_twins.items()
    ]).set_index("Scenario")
    return kpi_table


def determine_best_policy(kpi_table):
    """
    Automatically determine the best policy based on profit and service level.
    
    Args:
        kpi_table: DataFrame from build_kpi_table().
    
    Returns:
        tuple: (recommended_scenario, reason)
    """
    best_profit_scenario = kpi_table["Profit"].idxmax()
    best_service_scenario = kpi_table["Service_Level_%"].idxmax()

    print("BEST POLICY DETERMINATION")
    print("=" * 60)
    print(kpi_table[["Profit", "Service_Level_%", "Stockout_Units", "Warehouse_Utilization_%"]])
    print("-" * 60)
    print(f"Highest profit:        {best_profit_scenario}  "
          f"(Profit = {kpi_table.loc[best_profit_scenario, 'Profit']:,.2f})")
    print(f"Highest service level: {best_service_scenario}  "
          f"(Service Level = {kpi_table.loc[best_service_scenario, 'Service_Level_%']:.2f}%)")

    if best_profit_scenario == best_service_scenario:
        recommended = best_profit_scenario
        reason = "it simultaneously maximizes profit and customer service level"
    else:
        # Tie-break: prefer the scenario with the higher profit, provided its service
        # level is not materially worse (within 2 percentage points) than the best one
        service_gap = (kpi_table.loc[best_service_scenario, "Service_Level_%"]
                        - kpi_table.loc[best_profit_scenario, "Service_Level_%"])
        if service_gap <= 2.0:
            recommended = best_profit_scenario
            reason = ("it achieves the highest profit while its service level is within "
                       "2 percentage points of the best-performing scenario")
        else:
            recommended = best_service_scenario
            reason = ("although it is not the top scenario on profit, its service level "
                       "is materially better, which protects long-term customer retention")

    print("=" * 60)
    print(f"RECOMMENDED POLICY: {recommended}")
    print(f"Reason: {reason}.")
    return recommended, reason
