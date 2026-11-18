"""
Business Digital Twin — Object-Oriented Model.
Defines the core business entities: Supplier, Transportation, Inventory,
Warehouse, and BusinessDigitalTwin.
"""

import copy
import numpy as np
import pandas as pd

from config import PRODUCTS, SELLING_PRICE


class Supplier:
    """Represents a supplier that fulfils replenishment orders after a lead time."""

    def __init__(self, name, lead_time, reliability=1.0):
        self.name = name
        self.lead_time = lead_time          # days between order and arrival
        self.reliability = reliability       # fraction of ordered qty actually delivered

    def __repr__(self):
        return f"Supplier({self.name}, lead_time={self.lead_time}d)"


class Transportation:
    """Represents the inter-warehouse transportation link used for inventory transfers."""

    def __init__(self, transfer_cost_rate=0.75, transfer_lead_time=1):
        self.transfer_cost_rate = transfer_cost_rate  # cost per unit transferred
        self.transfer_lead_time = transfer_lead_time  # days for a transfer to arrive

    def cost_of(self, quantity):
        return quantity * self.transfer_cost_rate


class Inventory:
    """Lightweight helper wrapping a per-product on-hand quantity with capacity limits."""

    def __init__(self, initial_quantity, capacity):
        self.quantity = initial_quantity
        self.capacity = capacity

    def add(self, qty):
        """Add stock, never exceeding capacity. Returns quantity actually received."""
        space = max(0, self.capacity - self.quantity)
        received = min(qty, space)
        self.quantity += received
        return received

    def remove(self, qty):
        """Remove up to qty from stock, never going negative. Returns quantity removed."""
        removed = min(qty, self.quantity)
        self.quantity -= removed
        return removed

    def utilization(self):
        return self.quantity / self.capacity if self.capacity > 0 else 0.0


class Warehouse:
    """
    Represents a single warehouse holding one or more products.
    Maintains inventory, capacity, cost structure, demand fulfilment,
    and its assigned supplier / reorder policy per product.
    """

    def __init__(self, warehouse_id, supplier, holding_cost_rate, shipping_cost_rate,
                 product_configs):
        self.warehouse_id = warehouse_id
        self.supplier = supplier
        self.holding_cost_rate = holding_cost_rate
        self.shipping_cost_rate = shipping_cost_rate
        # product_configs: dict product -> dict(price, cost, capacity, reorder_point,
        #                                        reorder_qty, initial_inventory)
        self.product_configs = copy.deepcopy(product_configs)
        self.inventory = {
            p: Inventory(cfg["initial_inventory"], cfg["capacity"])
            for p, cfg in self.product_configs.items()
        }
        self.pending_orders = []   # list of dicts: {arrival_day, product, qty, source}

    # ---- core inventory operations -------------------------------------------------
    def receive_inventory(self, day):
        """Receive any supplier or transfer orders scheduled to arrive today,
        respecting warehouse capacity (excess is rejected/truncated)."""
        arrivals = [o for o in self.pending_orders if o["arrival_day"] == day]
        received_log = []
        for order in arrivals:
            received_qty = self.inventory[order["product"]].add(order["qty"])
            rejected_qty = order["qty"] - received_qty
            received_log.append({"product": order["product"], "ordered": order["qty"],
                                  "received": received_qty, "rejected": rejected_qty})
        self.pending_orders = [o for o in self.pending_orders if o["arrival_day"] != day]
        return received_log

    def sell_inventory(self, product, demand):
        """Fulfil demand from on-hand inventory. Sales can never exceed available stock.
        Returns (units_sold, stockout_units)."""
        available = self.inventory[product].quantity
        units_sold = self.inventory[product].remove(demand)
        stockout = max(0, demand - available)   # demand not met from current stock
        return units_sold, stockout

    def place_supplier_order(self, product, quantity, day):
        """Place a replenishment order with this warehouse's supplier.
        Purchase cost is recognised at the moment the order is placed."""
        arrival_day = day + self.supplier.lead_time
        self.pending_orders.append({"arrival_day": arrival_day, "product": product,
                                     "qty": quantity, "source": "supplier"})
        return self.calculate_purchase_cost(product, quantity)

    def receive_transfer(self, product, quantity, arrival_day):
        self.pending_orders.append({"arrival_day": arrival_day, "product": product,
                                     "qty": quantity, "source": "transfer"})

    # ---- cost / revenue calculations -----------------------------------------------
    def calculate_revenue(self, product, units_sold):
        return units_sold * self.product_configs[product]["price"]

    def calculate_purchase_cost(self, product, quantity):
        return quantity * self.product_configs[product]["cost"]

    def calculate_shipping_cost(self, units_sold):
        return units_sold * self.shipping_cost_rate

    def calculate_holding_cost(self, product):
        return self.inventory[product].quantity * self.holding_cost_rate

    def update_inventory(self, day):
        """Advance this warehouse's inventory state for the day: receive arrivals."""
        return self.receive_inventory(day)

    def display_state(self):
        print(f"--- {self.warehouse_id} ---  (Supplier: {self.supplier.name}, "
              f"Lead Time: {self.supplier.lead_time}d)")
        for p, inv in self.inventory.items():
            print(f"   {p}: {inv.quantity:.0f} / {inv.capacity} units "
                  f"({inv.utilization()*100:.1f}% utilized), "
                  f"pending orders: {sum(o['qty'] for o in self.pending_orders if o['product'] == p):.0f}")


class BusinessDigitalTwin:
    """
    The Business Digital Twin: represents the complete state of the multi-warehouse
    business and orchestrates day-by-day operations across all warehouses.
    It exposes the full set of business-level calculation methods required by the
    research brief, all derived from the twin's simulation history.
    """

    def __init__(self, warehouses, transportation):
        self.warehouses = {w.warehouse_id: w for w in warehouses}
        self.transportation = transportation
        self.history = []          # one row per (day, warehouse, product)
        self.transfer_log = []     # one row per executed transfer

    # ---- orchestration ---------------------------------------------------
    def update_inventory(self, day):
        """Receive all scheduled deliveries (supplier + transfer) for every warehouse."""
        for wh in self.warehouses.values():
            wh.update_inventory(day)

    def transfer_inventory(self, source_id, dest_id, product, quantity, day):
        """Physically move inventory from one warehouse to another, respecting the
        source's available stock and the destination's remaining capacity."""
        source, dest = self.warehouses[source_id], self.warehouses[dest_id]
        removed = source.inventory[product].remove(quantity)
        if removed <= 0:
            return 0
        arrival_day = day + self.transportation.transfer_lead_time
        dest.receive_transfer(product, removed, arrival_day)
        cost = self.transportation.cost_of(removed)
        self.transfer_log.append({"Day": day, "Product": product, "From": source_id,
                                   "To": dest_id, "Quantity": removed, "Cost": cost})
        return removed

    def sell_inventory(self, warehouse_id, product, demand):
        return self.warehouses[warehouse_id].sell_inventory(product, demand)

    def receive_inventory(self, day):
        for wh in self.warehouses.values():
            wh.receive_inventory(day)

    # ---- business-level KPI calculations (derived from self.history) -----
    def _history_df(self):
        return pd.DataFrame(self.history)

    def calculate_revenue(self):
        h = self._history_df()
        return h["Revenue"].sum() if len(h) else 0.0

    def calculate_purchase_cost(self):
        h = self._history_df()
        return h["Purchase_Cost"].sum() if len(h) else 0.0

    def calculate_shipping_cost(self):
        h = self._history_df()
        return h["Shipping_Cost"].sum() if len(h) else 0.0

    def calculate_holding_cost(self):
        h = self._history_df()
        return h["Holding_Cost"].sum() if len(h) else 0.0

    def calculate_transfer_cost(self):
        t = pd.DataFrame(self.transfer_log)
        return t["Cost"].sum() if len(t) else 0.0

    def calculate_profit(self):
        return (self.calculate_revenue() - self.calculate_purchase_cost()
                - self.calculate_holding_cost() - self.calculate_shipping_cost()
                - self.calculate_transfer_cost())

    def calculate_stockouts(self):
        h = self._history_df()
        return h["Stockout"].sum() if len(h) else 0.0

    def calculate_service_level(self):
        h = self._history_df()
        if len(h) == 0 or h["Demand"].sum() == 0:
            return 1.0
        return 1 - (h["Stockout"].sum() / h["Demand"].sum())

    def calculate_order_fill_rate(self):
        h = self._history_df()
        if len(h) == 0 or h["Demand"].sum() == 0:
            return 1.0
        return h["Sold"].sum() / h["Demand"].sum()

    def calculate_utilization(self):
        h = self._history_df()
        if len(h) == 0:
            return 0.0
        return (h["Inventory"] / h["Capacity"]).mean()

    def calculate_average_inventory(self):
        h = self._history_df()
        return h["Inventory"].mean() if len(h) else 0.0

    def calculate_inventory_turnover(self):
        h = self._history_df()
        avg_inv = self.calculate_average_inventory()
        if len(h) == 0 or avg_inv == 0:
            return 0.0
        return h["Sold"].sum() / avg_inv

    def display_state(self):
        print("=" * 60)
        print("BUSINESS DIGITAL TWIN — CURRENT STATE")
        print("=" * 60)
        for wh in self.warehouses.values():
            wh.display_state()
        print("-" * 60)
        if self.history:
            print(f"Cumulative Revenue:      {self.calculate_revenue():,.2f}")
            print(f"Cumulative Profit:       {self.calculate_profit():,.2f}")
            print(f"Cumulative Stockouts:    {self.calculate_stockouts():,.0f} units")
            print(f"Service Level:           {self.calculate_service_level()*100:.2f}%")
        print("=" * 60)
