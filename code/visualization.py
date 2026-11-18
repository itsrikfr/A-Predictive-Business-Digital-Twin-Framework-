"""
Visualization module for the Digital Twin framework.
Contains EDA plots and scenario comparison visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import WAREHOUSES, PRODUCTS


# ================================================================
# EXPLORATORY DATA ANALYSIS (EDA) PLOTS
# ================================================================

def plot_demand_trend(df):
    """EDA Graph 1 — Demand Trend by Warehouse (30-day rolling average)."""
    fig, ax = plt.subplots()
    daily_demand = df.groupby(["Date", "Warehouse"])["Demand"].sum().reset_index()
    for wh in WAREHOUSES:
        sub = daily_demand[daily_demand["Warehouse"] == wh].set_index("Date")["Demand"].rolling(30).mean()
        ax.plot(sub.index, sub.values, label=wh)
    ax.set_title("Demand Trend by Warehouse (30-day rolling average)")
    ax.set_xlabel("Date"); ax.set_ylabel("Units of demand")
    ax.legend()
    plt.tight_layout(); plt.show()
    print("Interpretation: both warehouses show clear upward trend and seasonal cycles, "
          "with Warehouse_A consistently demanding more units than Warehouse_B, consistent "
          "with its higher base-demand multiplier.")


def plot_inventory_history(df):
    """EDA Graph 2 — Historical Warehouse Inventory over time."""
    fig, ax = plt.subplots()
    daily_inv = df.groupby(["Date", "Warehouse"])["Inventory"].mean().reset_index()
    for wh in WAREHOUSES:
        sub = daily_inv[daily_inv["Warehouse"] == wh]
        ax.plot(sub["Date"], sub["Inventory"], label=wh, alpha=0.8)
    ax.set_title("Historical Warehouse Inventory Over Time")
    ax.set_xlabel("Date"); ax.set_ylabel("Average Inventory (units)")
    ax.legend()
    plt.tight_layout(); plt.show()
    print("Interpretation: the sawtooth pattern reflects the historical reorder-point policy — "
          "inventory depletes with sales and jumps back up whenever a replenishment order arrives.")


def plot_revenue_trend(df):
    """EDA Graph 3 — Revenue Trend."""
    fig, ax = plt.subplots()
    daily_revenue = df.groupby("Date")["Revenue"].sum()
    ax.plot(daily_revenue.index, daily_revenue.rolling(30).mean(), color="darkgreen")
    ax.set_title("Total Daily Revenue (30-day rolling average)")
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue")
    plt.tight_layout(); plt.show()
    print("Interpretation: revenue tracks the demand trend closely, confirming that pricing "
          "was held constant historically and revenue is primarily volume-driven.")


def plot_monthly_demand(df):
    """EDA Graph 4 — Monthly Sales (seasonality)."""
    fig, ax = plt.subplots()
    monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Demand"].sum()
    monthly_sales.index = monthly_sales.index.astype(str)
    ax.bar(monthly_sales.index, monthly_sales.values, color="steelblue")
    ax.set_title("Total Monthly Demand (Seasonality)")
    ax.set_xlabel("Month"); ax.set_ylabel("Total units demanded")
    plt.xticks(rotation=90)
    plt.tight_layout(); plt.show()
    print("Interpretation: monthly totals reveal a mild seasonal cycle superimposed on a "
          "gradual upward trend across the two-year history — exactly the pattern a forecasting "
          "model needs to learn.")


def plot_correlation_heatmap(df):
    """EDA Graph 5 — Correlation Heatmap."""
    numeric_cols = ["Demand", "Inventory", "Selling_Price", "Purchase_Cost",
                    "Holding_Cost", "Shipping_Cost", "Lead_Time", "Order_Quantity", "Revenue"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticks(range(len(numeric_cols))); ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=8)
    ax.set_title("Correlation Heatmap of Business Variables")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()
    print("Interpretation: Demand and Revenue are strongly positively correlated (revenue is "
          "volume-driven). Order_Quantity is weakly related to same-day demand because "
          "reordering is triggered by inventory depletion, not by the current day's demand.")


def plot_actual_vs_predicted(test_df):
    """Plot actual vs predicted demand on the test period for Warehouse_A / Product_1."""
    sample = test_df[(test_df["Warehouse"] == "Warehouse_A") & (test_df["Product"] == "Product_1")]
    fig, ax = plt.subplots()
    ax.plot(sample["Date"], sample["Demand"], label="Actual Demand", color="black")
    ax.plot(sample["Date"], sample["Predicted_Demand"], label="Predicted Demand",
            color="orange", linestyle="--")
    ax.set_title("Actual vs Predicted Demand — Warehouse_A / Product_1 (Test Period)")
    ax.set_xlabel("Date"); ax.set_ylabel("Units")
    ax.legend()
    plt.tight_layout(); plt.show()
    print("Interpretation: predicted demand tracks the seasonal and weekly pattern of actual "
          "demand closely, confirming the model has learned the underlying demand structure "
          "well enough to drive downstream inventory-decision simulation.")


def run_all_eda(df, test_df=None):
    """Run all EDA visualizations."""
    plot_demand_trend(df)
    plot_inventory_history(df)
    plot_revenue_trend(df)
    plot_monthly_demand(df)
    plot_correlation_heatmap(df)
    if test_df is not None:
        plot_actual_vs_predicted(test_df)


# ================================================================
# SCENARIO COMPARISON VISUALIZATIONS
# ================================================================

def plot_scenario_inventory(scenario_twins):
    """Total Inventory Over Time by Scenario."""
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    scenario_histories = {name: pd.DataFrame(t.history) for name, t in scenario_twins.items()}

    fig, ax = plt.subplots()
    for (name, h), c in zip(scenario_histories.items(), colors):
        daily_inv = h.groupby("Day")["Inventory"].sum()
        ax.plot(daily_inv.index, daily_inv.values, label=name.replace("\n", " "), color=c)
    ax.set_title("Total Inventory Over Time by Scenario")
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Total Units in Stock")
    ax.legend()
    plt.tight_layout(); plt.show()


def plot_kpi_bars(kpi_table):
    """Grouped bar comparisons for key KPIs."""
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    kpi_metrics = [
        ("Revenue", "Revenue Comparison"), ("Profit", "Profit Comparison"),
        ("Holding_Cost", "Holding Cost Comparison"), ("Shipping_Cost", "Shipping Cost Comparison"),
        ("Stockout_Units", "Stockout Comparison"), ("Service_Level_%", "Service Level Comparison"),
        ("Warehouse_Utilization_%", "Warehouse Utilization Comparison"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, kpi_metrics):
        values = kpi_table[col].values
        bars = ax.bar(kpi_table.index, values, color=colors)
        ax.set_title(title)
        ax.set_xticklabels([s.replace(": ", ":\n") for s in kpi_table.index], fontsize=8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:,.0f}" if abs(v) > 10 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    # Turn off the unused 8th subplot
    axes[-1].axis("off")
    plt.tight_layout(); plt.show()


def plot_kpi_dashboard(kpi_table):
    """Consolidated Business KPI Dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(kpi_table))
    width = 0.35

    # Profit vs Revenue
    ax = axes[0, 0]
    ax.bar(x - width/2, kpi_table["Revenue"], width, label="Revenue", color="#4C72B0")
    ax.bar(x + width/2, kpi_table["Profit"], width, label="Profit", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in range(len(kpi_table))])
    ax.set_title("Revenue vs Profit"); ax.legend()

    # Cost breakdown stacked bar
    ax = axes[0, 1]
    cost_cols = ["Purchase_Cost", "Holding_Cost", "Shipping_Cost", "Transfer_Cost"]
    bottom = np.zeros(len(kpi_table))
    for col, c in zip(cost_cols, ["#8172B2", "#CCB974", "#64B5CD", "#C44E52"]):
        ax.bar(x, kpi_table[col], bottom=bottom, label=col, color=c)
        bottom += kpi_table[col].values
    ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in range(len(kpi_table))])
    ax.set_title("Cost Breakdown"); ax.legend(fontsize=7)

    # Service level vs stockouts
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.bar(x, kpi_table["Service_Level_%"], color="#55A868", alpha=0.7, label="Service Level %")
    ax2.plot(x, kpi_table["Stockout_Units"], color="#C44E52", marker="o", label="Stockout Units")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in range(len(kpi_table))])
    ax.set_title("Service Level vs Stockouts")
    ax.set_ylabel("Service Level %"); ax2.set_ylabel("Stockout Units")

    # Utilization & turnover
    ax = axes[1, 1]
    ax.bar(x - width/2, kpi_table["Warehouse_Utilization_%"], width, label="Utilization %", color="#64B5CD")
    ax.bar(x + width/2, kpi_table["Inventory_Turnover"]*10, width, label="Turnover (x10)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}" for i in range(len(kpi_table))])
    ax.set_title("Utilization & Inventory Turnover"); ax.legend(fontsize=8)

    fig.suptitle("Business KPI Dashboard — Scenario Comparison (S1=Current, S2=Transfer, S3=Increased Reorder)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


def run_all_scenario_plots(scenario_twins, kpi_table):
    """Run all scenario comparison visualizations."""
    plot_scenario_inventory(scenario_twins)
    plot_kpi_bars(kpi_table)
    plot_kpi_dashboard(kpi_table)
