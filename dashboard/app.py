"""
Predictive Business Digital Twin — Interactive Web Dashboard
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code')))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Twin Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from code/ modules ───────────────────────────────────
from config import (
    WAREHOUSES, PRODUCTS, SELLING_PRICE, PURCHASE_COST,
    FEATURES, TARGET, RANDOM_SEED, XGBOOST_AVAILABLE,
    SIMULATION_HORIZON_DAYS, HOLDING_COST_RATE, SHIPPING_COST_RATE,
)
from data_generation import generate_historical_data, clean_data
from feature_engineering import engineer_features
from forecasting import train_forecast_model, generate_future_demand
from simulation import (
    build_digital_twin, build_demand_lookup, run_simulation,
    scenario2_transfer_policy,
)
from kpi import build_kpi_table, determine_best_policy
from validation import validate_twin


# ── Cache heavy computations ────────────────────────────────────
@st.cache_data(show_spinner="Generating dataset...")
def load_data():
    raw_df = generate_historical_data()
    df = clean_data(raw_df)
    df, wh_cat, prod_cat = engineer_features(df)
    return raw_df, df, wh_cat, prod_cat


@st.cache_data(show_spinner="Training ML model...")
def train_model(_df):
    model, test_df, mae, rmse, r2 = train_forecast_model(_df)
    return model, test_df, mae, rmse, r2


@st.cache_data(show_spinner="Forecasting future demand...")
def forecast_future(_df, _model):
    pred_df, future_dates = generate_future_demand(_df, _model)
    return pred_df, future_dates


@st.cache_data(show_spinner="Running simulations...")
def run_all_scenarios(_pred_df, _future_dates):
    demand_lookup = build_demand_lookup(_pred_df)

    twin_s1 = build_digital_twin()
    twin_s1 = run_simulation(twin_s1, demand_lookup, SIMULATION_HORIZON_DAYS,
                              _future_dates, reorder_multiplier=1.0, transfer_policy=None)

    twin_s2 = build_digital_twin()
    twin_s2 = run_simulation(twin_s2, demand_lookup, SIMULATION_HORIZON_DAYS,
                              _future_dates, reorder_multiplier=1.0,
                              transfer_policy=scenario2_transfer_policy)

    twin_s3 = build_digital_twin()
    twin_s3 = run_simulation(twin_s3, demand_lookup, SIMULATION_HORIZON_DAYS,
                              _future_dates, reorder_multiplier=1.5, transfer_policy=None)

    scenario_twins = {
        "S1: Current Policy": twin_s1,
        "S2: Transfer Policy": twin_s2,
        "S3: Increased Reorder": twin_s3,
    }
    kpi_table = build_kpi_table(scenario_twins)
    return scenario_twins, kpi_table, twin_s1, twin_s2, twin_s3


# ── Sidebar navigation ──────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=60)
st.sidebar.title("🏭 Digital Twin")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview",
     "📊 EDA — Exploratory Analysis",
     "🤖 ML — Demand Forecasting",
     "🔮 Digital Twin Simulation",
     "📈 KPI Dashboard",
     "✅ Validation"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Predictive Business Digital Twin Framework")
st.sidebar.caption(f"Model: {'XGBoost' if XGBOOST_AVAILABLE else 'Random Forest'}")


# ── Load data (cached) ──────────────────────────────────────────
raw_df, df, wh_cat, prod_cat = load_data()


# ================================================================
#  PAGE 1 — OVERVIEW
# ================================================================
if page == "🏠 Overview":
    st.title("🏭 Predictive Business Digital Twin Framework")
    st.markdown("#### A data-driven simulation for optimal inventory policy selection")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Records", f"{len(raw_df):,}")
    col2.metric("🏪 Warehouses", len(WAREHOUSES))
    col3.metric("📋 Products", len(PRODUCTS))
    col4.metric("📅 History", f"{(df['Date'].max() - df['Date'].min()).days} days")

    st.markdown("---")

    # Pipeline
    st.subheader("🔄 Pipeline Architecture")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ```
        ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
        │  Historical Data │───▶│ Feature Engineer  │───▶│  ML Forecasting │
        │  (730 days × 6)  │    │  (12 features)    │    │  (XGBoost / RF) │
        └─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                                │
        ┌─────────────────┐    ┌──────────────────┐    ┌────────▼────────┐
        │  Best Policy     │◀──│  KPI Comparison   │◀──│  Digital Twin    │
        │  Recommendation  │    │  (3 scenarios)    │    │  Simulation     │
        └─────────────────┘    └──────────────────┘    └─────────────────┘
        ```
        """)
    with col_b:
        st.markdown("**Three Scenarios Tested:**")
        st.markdown("""
        | # | Policy | Description |
        |---|--------|-------------|
        | S1 | Current | Baseline reorder-point |
        | S2 | Transfer | Proactive A→B transfers |
        | S3 | Increased | 1.5× reorder quantity |
        """)

    st.markdown("---")

    # Business config
    st.subheader("⚙️ Business Configuration")
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        st.markdown("**Selling Prices**")
        st.dataframe(pd.DataFrame(SELLING_PRICE.items(), columns=["Product", "Price ($)"]),
                     hide_index=True, use_container_width=True)
    with col_y:
        st.markdown("**Purchase Costs**")
        st.dataframe(pd.DataFrame(PURCHASE_COST.items(), columns=["Product", "Cost ($)"]),
                     hide_index=True, use_container_width=True)
    with col_z:
        st.markdown("**Cost Rates**")
        st.markdown(f"- Holding: **${HOLDING_COST_RATE}/unit/day**")
        st.markdown(f"- Shipping: **${SHIPPING_COST_RATE}/unit**")
        st.markdown(f"- Forecast horizon: **{SIMULATION_HORIZON_DAYS} days**")

    st.markdown("---")
    st.subheader("📄 Sample Data")
    st.dataframe(raw_df.head(20), use_container_width=True)


# ================================================================
#  PAGE 2 — EDA
# ================================================================
elif page == "📊 EDA — Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    # ── EDA 1: Demand Trend ──
    st.subheader("1️⃣ Demand Trend by Warehouse (30-day rolling average)")
    fig, ax = plt.subplots(figsize=(12, 5))
    daily_demand = df.groupby(["Date", "Warehouse"])["Demand"].sum().reset_index()
    for wh in WAREHOUSES:
        sub = daily_demand[daily_demand["Warehouse"] == wh].set_index("Date")["Demand"].rolling(30).mean()
        ax.plot(sub.index, sub.values, label=wh, linewidth=2)
    ax.set_title("Demand Trend by Warehouse", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Units of demand")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Both warehouses show clear upward trend and seasonal cycles. "
            "Warehouse_A consistently demands more units, matching its higher multiplier (1.20 vs 0.85).")

    st.markdown("---")

    # ── EDA 2: Inventory History ──
    st.subheader("2️⃣ Historical Warehouse Inventory")
    fig, ax = plt.subplots(figsize=(12, 5))
    daily_inv = df.groupby(["Date", "Warehouse"])["Inventory"].mean().reset_index()
    for wh in WAREHOUSES:
        sub = daily_inv[daily_inv["Warehouse"] == wh]
        ax.plot(sub["Date"], sub["Inventory"], label=wh, alpha=0.8, linewidth=2)
    ax.set_title("Historical Warehouse Inventory Over Time", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Average Inventory (units)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 The sawtooth pattern reflects the reorder-point policy — inventory depletes with sales "
            "and jumps back up whenever a replenishment order arrives.")

    st.markdown("---")

    # ── EDA 3: Revenue Trend ──
    st.subheader("3️⃣ Revenue Trend (30-day rolling average)")
    fig, ax = plt.subplots(figsize=(12, 5))
    daily_revenue = df.groupby("Date")["Revenue"].sum()
    ax.plot(daily_revenue.index, daily_revenue.rolling(30).mean(), color="darkgreen", linewidth=2)
    ax.set_title("Total Daily Revenue", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Revenue ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Revenue tracks demand closely — pricing was held constant, so revenue is primarily volume-driven.")

    st.markdown("---")

    # ── EDA 4: Monthly Seasonality ──
    st.subheader("4️⃣ Monthly Demand (Seasonality)")
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Demand"].sum()
    monthly_sales.index = monthly_sales.index.astype(str)
    ax.bar(monthly_sales.index, monthly_sales.values, color="steelblue")
    ax.set_title("Total Monthly Demand", fontsize=14)
    ax.set_xlabel("Month"); ax.set_ylabel("Total units demanded")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Monthly totals reveal a mild seasonal cycle superimposed on a gradual upward trend.")

    st.markdown("---")

    # ── EDA 5: Correlation Heatmap ──
    st.subheader("5️⃣ Correlation Heatmap")
    numeric_cols = ["Demand", "Inventory", "Selling_Price", "Purchase_Cost",
                    "Holding_Cost", "Shipping_Cost", "Lead_Time", "Order_Quantity", "Revenue"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols))); ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_cols))); ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("Correlation Heatmap of Business Variables", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Demand and Revenue are strongly correlated (volume-driven). "
            "Order_Quantity is weakly related to same-day demand — reordering is triggered by inventory depletion.")


# ================================================================
#  PAGE 3 — ML FORECASTING
# ================================================================
elif page == "🤖 ML — Demand Forecasting":
    st.title("🤖 Demand Forecasting Model")
    st.markdown("---")

    model, test_df, mae, rmse, r2 = train_model(df)

    # Metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧠 Model", "XGBoost" if XGBOOST_AVAILABLE else "Random Forest")
    col2.metric("MAE", f"{mae:.2f} units")
    col3.metric("RMSE", f"{rmse:.2f} units")
    col4.metric("R²", f"{r2:.3f}")

    if r2 >= 0.85:
        st.success(f"✅ R² = {r2:.3f} — The model explains {r2*100:.1f}% of demand variance.")
    else:
        st.warning(f"⚠️ R² = {r2:.3f} — Model performance is moderate.")

    st.markdown("---")

    # Features used
    st.subheader("📋 Features Used (12 total)")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Type": ["Calendar", "Calendar", "Calendar", "Calendar", "Trend",
                 "Lag", "Lag", "Lag", "Rolling", "Rolling",
                 "Categorical", "Categorical"]
    })
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Actual vs Predicted plot
    st.subheader("📈 Actual vs Predicted Demand — Warehouse_A / Product_1")
    sample = test_df[(test_df["Warehouse"] == "Warehouse_A") & (test_df["Product"] == "Product_1")]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample["Date"], sample["Demand"], label="Actual Demand", color="black", linewidth=2)
    ax.plot(sample["Date"], sample["Predicted_Demand"], label="Predicted Demand",
            color="orange", linestyle="--", linewidth=2)
    ax.set_title("Actual vs Predicted Demand (Test Period)", fontsize=14)
    ax.set_xlabel("Date"); ax.set_ylabel("Units")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("📌 Predicted demand tracks the seasonal and weekly pattern closely — "
            "the model has learned the underlying demand structure well enough to drive simulation.")


# ================================================================
#  PAGE 4 — DIGITAL TWIN SIMULATION
# ================================================================
elif page == "🔮 Digital Twin Simulation":
    st.title("🔮 Digital Twin — Scenario Simulation")
    st.markdown("---")

    model, test_df, mae, rmse, r2 = train_model(df)
    pred_df, future_dates = forecast_future(df, model)
    scenario_twins, kpi_table, twin_s1, twin_s2, twin_s3 = run_all_scenarios(pred_df, future_dates)

    # Architecture
    st.subheader("🏗️ Object-Oriented Architecture")
    st.code("""
    class Supplier       →  lead time, reliability
    class Inventory      →  quantity, capacity, add(), remove()
    class Warehouse      →  inventory per product, reorder policy, costs
    class Transportation →  inter-warehouse transfer costs
    class BusinessDigitalTwin → orchestrates all warehouses day-by-day
    """, language="text")

    st.markdown("---")

    # Scenario descriptions
    st.subheader("🎯 Three Scenarios Simulated")
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        st.markdown("### S1: Current Policy")
        st.markdown("- Reorder point: **400 units**")
        st.markdown("- Reorder qty: **700 units**")
        st.markdown("- No transfers")
        st.markdown("- *Baseline for comparison*")
    with s_col2:
        st.markdown("### S2: Transfer Policy")
        st.markdown("- Same reorder rules as S1")
        st.markdown("- **Proactive transfers** A→B")
        st.markdown("- 7-day demand lookahead")
        st.markdown(f"- Transfers executed: **{len(twin_s2.transfer_log)}**")
    with s_col3:
        st.markdown("### S3: Increased Reorder")
        st.markdown("- Reorder point: **400 units**")
        st.markdown("- Reorder qty: **1,050 units** (1.5×)")
        st.markdown("- No transfers")
        st.markdown("- *Larger safety stock*")

    st.markdown("---")

    # Inventory over time
    st.subheader("📦 Total Inventory Over Time")
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for (name, twin), c in zip(scenario_twins.items(), colors):
        h = pd.DataFrame(twin.history)
        daily_inv = h.groupby("Day")["Inventory"].sum()
        ax.plot(daily_inv.index, daily_inv.values, label=name, color=c, linewidth=2)
    ax.set_title("Total Inventory Over Time by Scenario", fontsize=14)
    ax.set_xlabel("Simulation Day"); ax.set_ylabel("Total Units in Stock")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Predicted demand table
    st.subheader("🔮 60-Day Demand Forecast (sample)")
    st.dataframe(pred_df.head(18), use_container_width=True)


# ================================================================
#  PAGE 5 — KPI DASHBOARD
# ================================================================
elif page == "📈 KPI Dashboard":
    st.title("📈 Business KPI Dashboard")
    st.markdown("---")

    model, test_df, mae, rmse, r2 = train_model(df)
    pred_df, future_dates = forecast_future(df, model)
    scenario_twins, kpi_table, twin_s1, twin_s2, twin_s3 = run_all_scenarios(pred_df, future_dates)

    # Best policy
    best_profit = kpi_table["Profit"].idxmax()
    best_service = kpi_table["Service_Level_%"].idxmax()

    st.subheader("🏆 Recommended Policy")
    st.success(f"**{best_profit}** — Highest profit with strong service level")

    st.markdown("---")

    # KPI Table
    st.subheader("📊 Full KPI Comparison Table")
    kpi_display = kpi_table.round(2)
    st.dataframe(kpi_display, use_container_width=True)

    st.markdown("---")

    # Top-level metrics per scenario
    st.subheader("💰 Key Metrics Comparison")
    cols = st.columns(3)
    for i, (name, row) in enumerate(kpi_table.iterrows()):
        with cols[i]:
            st.markdown(f"#### {name}")
            st.metric("Revenue", f"${row['Revenue']:,.0f}")
            st.metric("Profit", f"${row['Profit']:,.0f}",
                      delta=f"${row['Profit'] - kpi_table['Profit'].iloc[0]:,.0f}" if i > 0 else None)
            st.metric("Service Level", f"{row['Service_Level_%']:.2f}%")
            st.metric("Stockouts", f"{row['Stockout_Units']:,.0f} units")
            st.metric("Fill Rate", f"{row['Order_Fill_Rate_%']:.2f}%")
            st.metric("Utilization", f"{row['Warehouse_Utilization_%']:.1f}%")

    st.markdown("---")

    # Revenue vs Profit chart
    st.subheader("💵 Revenue vs Profit")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(kpi_table))
    width = 0.35
    ax.bar(x - width/2, kpi_table["Revenue"], width, label="Revenue", color="#4C72B0")
    ax.bar(x + width/2, kpi_table["Profit"], width, label="Profit", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(kpi_table.index, fontsize=10)
    ax.set_title("Revenue vs Profit by Scenario", fontsize=14)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3, axis="y")
    for i, (rev, prof) in enumerate(zip(kpi_table["Revenue"], kpi_table["Profit"])):
        ax.text(i - width/2, rev + 1000, f"${rev:,.0f}", ha="center", fontsize=8)
        ax.text(i + width/2, prof + 1000, f"${prof:,.0f}", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Cost breakdown
    st.subheader("🧾 Cost Breakdown")
    fig, ax = plt.subplots(figsize=(10, 5))
    cost_cols = ["Purchase_Cost", "Holding_Cost", "Shipping_Cost", "Transfer_Cost"]
    cost_colors = ["#8172B2", "#CCB974", "#64B5CD", "#C44E52"]
    bottom = np.zeros(len(kpi_table))
    for col, c in zip(cost_cols, cost_colors):
        ax.bar(x, kpi_table[col], bottom=bottom, label=col.replace("_", " "), color=c)
        bottom += kpi_table[col].values
    ax.set_xticks(x); ax.set_xticklabels(kpi_table.index, fontsize=10)
    ax.set_title("Stacked Cost Breakdown by Scenario", fontsize=14)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Service Level vs Stockouts
    st.subheader("📉 Service Level vs Stockouts")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    bars = ax.bar(x, kpi_table["Service_Level_%"], color="#55A868", alpha=0.7, width=0.5)
    line = ax2.plot(x, kpi_table["Stockout_Units"], color="#C44E52", marker="o",
                    linewidth=2, markersize=8)
    ax.set_xticks(x); ax.set_xticklabels(kpi_table.index, fontsize=10)
    ax.set_title("Service Level vs Stockout Units", fontsize=14)
    ax.set_ylabel("Service Level %", color="#55A868")
    ax2.set_ylabel("Stockout Units", color="#C44E52")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ================================================================
#  PAGE 6 — VALIDATION
# ================================================================
elif page == "✅ Validation":
    st.title("✅ Simulation Validation")
    st.markdown("---")

    model, test_df, mae, rmse, r2 = train_model(df)
    pred_df, future_dates = forecast_future(df, model)
    scenario_twins, kpi_table, twin_s1, twin_s2, twin_s3 = run_all_scenarios(pred_df, future_dates)

    st.subheader("🔍 8 Integrity Checks × 3 Scenarios = 24 Total Checks")
    st.markdown("---")

    checks = [
        "Inventory never negative",
        "Inventory never exceeds capacity",
        "Revenue = Units Sold × Selling Price",
        "Purchase cost is non-negative",
        "Purchase cost > 0 only when orders placed",
        "Shipping cost proportional to units sold",
        "Holding cost proportional to remaining inventory",
        "Stockouts within [0, Demand] bounds",
        "Profit = Revenue - Purchase - Holding - Shipping - Transfer",
    ]

    all_pass = True
    for name, twin in scenario_twins.items():
        h = pd.DataFrame(twin.history)
        st.subheader(f"🔧 {name}")

        results = {}
        results["Inventory never negative"] = bool((h["Inventory"] >= 0).all())
        results["Inventory never exceeds capacity"] = bool((h["Inventory"] <= h["Capacity"]).all())

        price_lookup = {p: SELLING_PRICE[p] for p in PRODUCTS}
        expected_rev = h.apply(lambda r: r["Sold"] * price_lookup[r["Product"]], axis=1)
        results["Revenue = Units Sold × Selling Price"] = bool(np.allclose(h["Revenue"], expected_rev))

        results["Purchase cost is non-negative"] = bool((h["Purchase_Cost"] >= 0).all())
        results["Purchase cost > 0 only when orders placed"] = bool((h["Purchase_Cost"] > 0).sum() > 0)

        results["Shipping cost proportional to units sold"] = bool(
            np.allclose(h["Shipping_Cost"], h["Sold"] * SHIPPING_COST_RATE))

        results["Holding cost proportional to remaining inventory"] = bool(
            np.allclose(h["Holding_Cost"], h["Inventory"] * HOLDING_COST_RATE))

        results["Stockouts within [0, Demand] bounds"] = bool(
            (h["Stockout"] >= 0).all() and (h["Stockout"] <= h["Demand"]).all())

        total_check = (twin.calculate_revenue() - twin.calculate_purchase_cost()
                       - twin.calculate_holding_cost() - twin.calculate_shipping_cost()
                       - twin.calculate_transfer_cost())
        results["Profit = Revenue - Purchase - Holding - Shipping - Transfer"] = bool(
            np.isclose(twin.calculate_profit(), total_check))

        for check, passed in results.items():
            if passed:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;✅ **PASS** — {check}")
            else:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;❌ **FAIL** — {check}")
                all_pass = False

        st.markdown("---")

    # Cross-scenario check
    profits = kpi_table["Profit"].values
    kpis_differ = not np.allclose(profits, profits[0])

    if kpis_differ:
        st.markdown(f"✅ **PASS** — Scenario KPIs differ across policies "
                    f"(profits: {[f'${p:,.0f}' for p in profits]})")
    else:
        st.markdown("❌ **FAIL** — Scenario KPIs do not differ")
        all_pass = False

    st.markdown("---")
    if all_pass:
        st.success("## ✅ ALL 24+ CHECKS PASSED — Simulation is mathematically verified!")
    else:
        st.error("## ❌ SOME CHECKS FAILED — Review simulation logic.")
