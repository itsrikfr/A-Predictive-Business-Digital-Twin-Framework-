"""
Day-by-day simulation engine for the Business Digital Twin.
Handles twin construction, demand lookup, simulation execution, and scenario policies.
"""

import numpy as np
import pandas as pd

from config import (
    WAREHOUSES, PRODUCTS, SELLING_PRICE, PURCHASE_COST,
    SUPPLIER_LEAD_TIME, HOLDING_COST_RATE, SHIPPING_COST_RATE,
    LOOKAHEAD_DAYS,
)
from digital_twin import Supplier, Transportation, Warehouse, BusinessDigitalTwin


def build_digital_twin(reorder_point=400, reorder_qty=700, capacity=1500,
                        initial_inventory=600, transfer_cost_rate=0.75,
                        transfer_lead_time=1):
    """
    Factory that builds a FRESH BusinessDigitalTwin with baseline configuration.
    Every scenario calls this to get an independent twin instance.
    
    Args:
        reorder_point: Inventory level that triggers a reorder.
        reorder_qty: Quantity to order when reorder is triggered.
        capacity: Maximum warehouse capacity per product.
        initial_inventory: Starting inventory per product.
        transfer_cost_rate: Cost per unit for inter-warehouse transfers.
        transfer_lead_time: Days for a transfer to arrive.
    
    Returns:
        BusinessDigitalTwin: Fresh twin instance ready for simulation.
    """
    warehouses = []
    for wh_id in WAREHOUSES:
        product_configs = {
            p: {
                "price": SELLING_PRICE[p],
                "cost": PURCHASE_COST[p],
                "capacity": capacity,
                "reorder_point": reorder_point,
                "reorder_qty": reorder_qty,
                "initial_inventory": initial_inventory,
            }
            for p in PRODUCTS
        }
        supplier = Supplier(name=f"Supplier_of_{wh_id}",
                             lead_time=max(SUPPLIER_LEAD_TIME.values()) - (2 if wh_id == "Warehouse_A" else 0))
        wh = Warehouse(wh_id, supplier, HOLDING_COST_RATE, SHIPPING_COST_RATE, product_configs)
        warehouses.append(wh)
    transportation = Transportation(transfer_cost_rate, transfer_lead_time)
    return BusinessDigitalTwin(warehouses, transportation)


def build_demand_lookup(predicted_df):
    """
    Convert the predicted-demand dataframe into a fast (warehouse, product, day)->demand lookup,
    where day 0 is the first day of the simulation horizon.
    
    Args:
        predicted_df: DataFrame with Date, Warehouse, Product, Predicted_Demand columns.
    
    Returns:
        dict: Lookup dictionary mapping (warehouse, product, day) to demand.
    """
    lookup = {}
    horizon_start = predicted_df["Date"].min()
    for _, row in predicted_df.iterrows():
        day = (row["Date"] - horizon_start).days
        lookup[(row["Warehouse"], row["Product"], day)] = row["Predicted_Demand"]
    return lookup


def run_simulation(twin, demand_lookup, horizon_days, dates,
                    reorder_multiplier=1.0, transfer_policy=None):
    """
    Run the full day-by-day simulation for the given horizon.
    
    Args:
        twin: BusinessDigitalTwin instance.
        demand_lookup: Dict mapping (warehouse, product, day) to demand.
        horizon_days: Number of days to simulate.
        dates: DatetimeIndex of simulation dates.
        reorder_multiplier: Multiplier applied to base reorder quantity.
        transfer_policy: Optional callable(twin, day, demand_lookup) for proactive transfers.
    
    Returns:
        BusinessDigitalTwin: The twin with simulation history populated.
    """
    for day in range(horizon_days):
        date = dates[day]
        # Step 1 — receive deliveries scheduled for today (supplier + transfer arrivals)
        twin.update_inventory(day)

        # Optional proactive transfer policy, executed before demand is read/sold
        if transfer_policy is not None:
            transfer_policy(twin, day, demand_lookup)

        # Steps 2-11: for every warehouse/product, read demand, sell, cost, reorder
        for wh_id, wh in twin.warehouses.items():
            for product in PRODUCTS:
                demand = demand_lookup.get((wh_id, product, day), 0)

                # Step 3-4: sell inventory (capped, never negative)
                units_sold, stockout = wh.sell_inventory(product, demand)

                # Step 6: revenue
                revenue = wh.calculate_revenue(product, units_sold)

                # Step 8: shipping cost
                shipping_cost = wh.calculate_shipping_cost(units_sold)

                # Step 10: reorder-point check -> place supplier order (Step 7: purchase cost)
                # The reorder decision is based on INVENTORY POSITION (on-hand + already
                # on order) rather than on-hand stock alone, so the warehouse does not
                # keep placing duplicate orders while a previous shipment is still in
                # transit (a classic reorder-point bug that causes runaway over-ordering).
                purchase_cost = 0.0
                cfg = wh.product_configs[product]
                pending_qty = sum(o["qty"] for o in wh.pending_orders if o["product"] == product)
                inventory_position = wh.inventory[product].quantity + pending_qty
                if inventory_position < cfg["reorder_point"]:
                    order_qty = int(round(cfg["reorder_qty"] * reorder_multiplier))
                    purchase_cost = wh.place_supplier_order(product, order_qty, day)

                # Step 9: holding cost from remaining inventory
                holding_cost = wh.calculate_holding_cost(product)

                twin.history.append({
                    "Day": day, "Date": date, "Warehouse": wh_id, "Product": product,
                    "Demand": demand, "Sold": units_sold, "Stockout": stockout,
                    "Inventory": wh.inventory[product].quantity,
                    "Capacity": wh.inventory[product].capacity,
                    "Revenue": revenue, "Purchase_Cost": purchase_cost,
                    "Shipping_Cost": shipping_cost, "Holding_Cost": holding_cost,
                })
        # Step 13: loop advances to next day automatically
    return twin


# ================================================================
# SCENARIO TRANSFER POLICIES
# ================================================================

def transfer_lookahead_demand(warehouse_id, product, day, lookup, lookahead=LOOKAHEAD_DAYS):
    """Sum of predicted demand for a warehouse/product over the next `lookahead` days."""
    return sum(lookup.get((warehouse_id, product, d), 0) for d in range(day, day + lookahead))


def scenario2_transfer_policy(twin, day, lookup):
    """Proactively transfer stock from Warehouse_A to Warehouse_B ahead of a demand spike."""
    source_id, dest_id = "Warehouse_A", "Warehouse_B"
    source, dest = twin.warehouses[source_id], twin.warehouses[dest_id]
    for product in PRODUCTS:
        upcoming_demand_dest = transfer_lookahead_demand(dest_id, product, day, lookup)
        dest_reorder_point = dest.product_configs[product]["reorder_point"]
        source_reorder_point = source.product_configs[product]["reorder_point"]
        dest_at_risk = (dest.inventory[product].quantity + 0.0) < upcoming_demand_dest
        source_surplus = source.inventory[product].quantity - source_reorder_point
        if dest_at_risk and source_surplus > 0:
            transfer_qty = int(min(source_surplus * 0.4,
                                    max(0, upcoming_demand_dest - dest.inventory[product].quantity)))
            if transfer_qty > 0:
                twin.transfer_inventory(source_id, dest_id, product, transfer_qty, day)
