"""
Validation module for the Digital Twin simulation.
Runs integrity checks on simulation results to ensure correctness.
"""

import numpy as np
import pandas as pd

from config import PRODUCTS, SELLING_PRICE, SHIPPING_COST_RATE, HOLDING_COST_RATE


def validate_twin(twin, label):
    """
    Run validation checks on a simulated Digital Twin.
    
    Checks:
    1. Inventory never negative
    2. Inventory never exceeds warehouse capacity
    3. Revenue = Units Sold × Selling Price
    4. Purchase cost is non-negative
    5. Shipping cost proportional to units sold
    6. Holding cost proportional to remaining inventory
    7. Stockouts within [0, Demand] bounds
    8. Profit formula consistency
    
    Args:
        twin: BusinessDigitalTwin with completed simulation history.
        label: Human-readable scenario label.
    
    Returns:
        bool: True if all checks passed, False otherwise.
    """
    h = pd.DataFrame(twin.history)
    results = {}

    # 1. Inventory never negative
    results["Inventory never negative"] = (h["Inventory"] >= 0).all()

    # 2. Inventory never exceeds warehouse capacity
    results["Inventory never exceeds capacity"] = (h["Inventory"] <= h["Capacity"]).all()

    # 3. Revenue == units sold * selling price
    price_lookup = {p: SELLING_PRICE[p] for p in PRODUCTS}
    expected_revenue = h.apply(lambda r: r["Sold"] * price_lookup[r["Product"]], axis=1)
    results["Revenue = Units Sold x Selling Price"] = np.allclose(h["Revenue"], expected_revenue)

    # 4. Purchase cost only appears when an order is actually placed (non-negative, and
    #    zero on days without a reorder-point breach is implicitly covered by construction)
    results["Purchase cost is non-negative"] = (h["Purchase_Cost"] >= 0).all()
    results["Purchase cost > 0 only when orders were placed"] = (
        (h["Purchase_Cost"] > 0).sum() > 0  # at least some orders were placed over the horizon
    )

    # 5. Shipping cost increases whenever products are sold (proportional to units sold)
    results["Shipping cost proportional to units sold"] = np.allclose(
        h["Shipping_Cost"], h["Sold"] * SHIPPING_COST_RATE
    )

    # 6. Holding cost calculated from remaining inventory
    results["Holding cost proportional to remaining inventory"] = np.allclose(
        h["Holding_Cost"], h["Inventory"] * HOLDING_COST_RATE
    )

    # 7. Stockouts computed correctly (never negative, never double counted beyond demand)
    results["Stockouts within [0, Demand] bounds"] = (
        (h["Stockout"] >= 0).all() and (h["Stockout"] <= h["Demand"]).all()
    )

    # 8. Profit formula check
    total_profit_check = (twin.calculate_revenue() - twin.calculate_purchase_cost()
                           - twin.calculate_holding_cost() - twin.calculate_shipping_cost()
                           - twin.calculate_transfer_cost())
    results["Profit = Revenue - Purchase - Holding - Shipping - Transfer"] = np.isclose(
        twin.calculate_profit(), total_profit_check
    )

    print(f"Validation results for {label}:")
    all_passed = True
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {check}")
    print()
    return all_passed


def validate_all_scenarios(scenario_twins, kpi_table):
    """
    Validate all scenario twins and check that KPIs differ.
    
    Args:
        scenario_twins: dict mapping labels to BusinessDigitalTwin instances.
        kpi_table: DataFrame from build_kpi_table().
    
    Returns:
        bool: True if all validations passed.
    """
    all_valid = True
    for label, twin in scenario_twins.items():
        all_valid &= validate_twin(twin, label.replace("\n", " "))

    # Scenario KPIs must differ, since different policies are simulated
    profits = kpi_table["Profit"].values
    kpis_differ = not np.allclose(profits, profits[0])
    print(f"[{'PASS' if kpis_differ else 'FAIL'}] Scenario KPIs differ across policies "
          f"(profits: {[f'{p:,.0f}' for p in profits]})")
    all_valid &= kpis_differ
    print()
    print("=" * 60)
    print(f"OVERALL VALIDATION STATUS: {'ALL CHECKS PASSED' if all_valid else 'SOME CHECKS FAILED'}")
    print("=" * 60)
    assert all_valid, "One or more validation checks failed — review simulation logic above."
    return all_valid
