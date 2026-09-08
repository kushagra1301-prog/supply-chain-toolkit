"""
Supply Chain Optimization Suite
-----------------------------------
A single Streamlit app combining four modules, selectable from the sidebar:

  1. Network Optimization  — capacitated facility location (PuLP/CBC)
  2. Greenfield / COG       — Weiszfeld center-of-gravity analysis
  3. Inventory Policy       — discrete-event simulation ((s,Q) / (R,S))
  4. Business Case Builder  — combines baseline-vs-optimized results from
                               modules 1-3 into a multi-year NPV business case

Run with: streamlit run app.py
"""

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pulp
import simpy
import streamlit as st

st.set_page_config(page_title="Supply Chain Optimization Suite", layout="wide")

# ==========================================================================================
# Shared helpers
# ==========================================================================================

def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def sample_facilities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"facility_id": "F1", "name": "Chicago DC", "lat": 41.8781, "lon": -87.6298, "fixed_cost": 250000, "capacity": 50000},
            {"facility_id": "F2", "name": "Dallas DC", "lat": 32.7767, "lon": -96.7970, "fixed_cost": 220000, "capacity": 45000},
            {"facility_id": "F3", "name": "Atlanta DC", "lat": 33.7490, "lon": -84.3880, "fixed_cost": 210000, "capacity": 40000},
            {"facility_id": "F4", "name": "Los Angeles DC", "lat": 34.0522, "lon": -118.2437, "fixed_cost": 300000, "capacity": 60000},
            {"facility_id": "F5", "name": "Newark DC", "lat": 40.7357, "lon": -74.1724, "fixed_cost": 280000, "capacity": 55000},
            {"facility_id": "F6", "name": "Denver DC", "lat": 39.7392, "lon": -104.9903, "fixed_cost": 190000, "capacity": 35000},
        ]
    )


def sample_customers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "C1", "name": "Seattle", "lat": 47.6062, "lon": -122.3321, "demand": 8000},
            {"customer_id": "C2", "name": "San Francisco", "lat": 37.7749, "lon": -122.4194, "demand": 12000},
            {"customer_id": "C3", "name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "demand": 6000},
            {"customer_id": "C4", "name": "Houston", "lat": 29.7604, "lon": -95.3698, "demand": 10000},
            {"customer_id": "C5", "name": "Minneapolis", "lat": 44.9778, "lon": -93.2650, "demand": 7000},
            {"customer_id": "C6", "name": "Detroit", "lat": 42.3314, "lon": -83.0458, "demand": 9000},
            {"customer_id": "C7", "name": "Miami", "lat": 25.7617, "lon": -80.1918, "demand": 8500},
            {"customer_id": "C8", "name": "New York", "lat": 40.7128, "lon": -74.0060, "demand": 15000},
            {"customer_id": "C9", "name": "Boston", "lat": 42.3601, "lon": -71.0589, "demand": 6500},
            {"customer_id": "C10", "name": "Salt Lake City", "lat": 40.7608, "lon": -111.8910, "demand": 5000},
        ]
    )


# ==========================================================================================
# Engine 1: Facility Location (network optimization)
# ==========================================================================================

def build_distance_matrix(facilities, customers):
    dist = pd.DataFrame(index=customers["customer_id"], columns=facilities["facility_id"], dtype=float)
    for _, c in customers.iterrows():
        for _, f in facilities.iterrows():
            dist.loc[c["customer_id"], f["facility_id"]] = haversine_miles(c["lat"], c["lon"], f["lat"], f["lon"])
    return dist


def solve_facility_location(facilities, customers, dist_matrix, cost_per_unit_distance,
                             max_facilities=None, forced_open=None, forced_closed=None):
    F = list(facilities["facility_id"])
    C = list(customers["customer_id"])
    fixed_cost = facilities.set_index("facility_id")["fixed_cost"].to_dict()
    capacity = facilities.set_index("facility_id")["capacity"].to_dict()
    demand = customers.set_index("customer_id")["demand"].to_dict()
    ship_cost = {(c, f): dist_matrix.loc[c, f] * cost_per_unit_distance for c in C for f in F}

    prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("Open", F, cat="Binary")
    x = pulp.LpVariable.dicts("Assign", (C, F), cat="Binary")

    for c in C:
        prob += pulp.lpSum(x[c][f] for f in F) == 1
    for f in F:
        prob += pulp.lpSum(demand[c] * x[c][f] for c in C) <= capacity[f] * y[f]

    if forced_open:
        for f in forced_open:
            prob += y[f] == 1
    if forced_closed:
        for f in forced_closed:
            prob += y[f] == 0
    if max_facilities is not None and max_facilities > 0:
        prob += pulp.lpSum(y[f] for f in F) <= max_facilities

    prob += pulp.lpSum(capacity[f] * y[f] for f in F) >= sum(demand.values())

    transport_cost_expr = pulp.lpSum(demand[c] * ship_cost[(c, f)] * x[c][f] for c in C for f in F)
    fixed_cost_expr = pulp.lpSum(fixed_cost[f] * y[f] for f in F)
    prob += fixed_cost_expr + transport_cost_expr

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    opened = [f for f in F if pulp.value(y[f]) and pulp.value(y[f]) > 0.5]
    objective = pulp.value(prob.objective)

    rows = []
    for c in C:
        for f in F:
            val = pulp.value(x[c][f])
            if val is not None and val > 0.5:
                rows.append({
                    "customer_id": c, "facility_id": f, "quantity": demand[c],
                    "distance": dist_matrix.loc[c, f], "transport_cost": demand[c] * ship_cost[(c, f)],
                })
    assignment_df = pd.DataFrame(rows)

    return {"status": status, "objective": objective, "opened": opened, "assignment": assignment_df}


def make_network_map(facilities, customers, opened, assignment_df):
    fig = go.Figure()
    if assignment_df is not None and not assignment_df.empty:
        fac_lookup = facilities.set_index("facility_id")
        cust_lookup = customers.set_index("customer_id")
        for _, row in assignment_df.iterrows():
            f = fac_lookup.loc[row["facility_id"]]
            c = cust_lookup.loc[row["customer_id"]]
            fig.add_trace(go.Scattergeo(lon=[c["lon"], f["lon"]], lat=[c["lat"], f["lat"]], mode="lines",
                                         line=dict(width=1, color="rgba(100,100,255,0.35)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scattergeo(lon=customers["lon"], lat=customers["lat"],
                                 text=customers["name"] + "<br>demand: " + customers["demand"].astype(str),
                                 mode="markers", marker=dict(size=8, color="dodgerblue"), name="Customers"))
    fac_closed = facilities[~facilities["facility_id"].isin(opened)]
    fac_open = facilities[facilities["facility_id"].isin(opened)]
    fig.add_trace(go.Scattergeo(lon=fac_closed["lon"], lat=fac_closed["lat"], text=fac_closed["name"] + " (not opened)",
                                 mode="markers", marker=dict(size=10, color="lightgray", symbol="square", line=dict(width=1, color="gray")),
                                 name="Candidate (closed)"))
    fig.add_trace(go.Scattergeo(lon=fac_open["lon"], lat=fac_open["lat"], text=fac_open["name"] + " (OPEN)",
                                 mode="markers", marker=dict(size=16, color="crimson", symbol="star", line=dict(width=1, color="darkred")),
                                 name="Opened Facility"))
    fig.update_layout(geo=dict(scope="world", projection_type="natural earth", showland=True,
                                landcolor="rgb(240,240,240)", countrycolor="rgb(200,200,200)"),
                       margin=dict(l=0, r=0, t=0, b=0), height=520,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# ==========================================================================================
# Engine 2: Center of Gravity / Greenfield
# ==========================================================================================

def weiszfeld(points, max_iter=200, tol=1e-5):
    w = points["demand"].values
    lats = points["lat"].values
    lons = points["lon"].values
    lat = np.average(lats, weights=w)
    lon = np.average(lons, weights=w)
    for _ in range(max_iter):
        dists = np.array([haversine_miles(lat, lon, lats[i], lons[i]) for i in range(len(points))])
        dists = np.where(dists < 1e-6, 1e-6, dists)
        weights = w / dists
        new_lat = np.sum(weights * lats) / np.sum(weights)
        new_lon = np.sum(weights * lons) / np.sum(weights)
        if abs(new_lat - lat) < tol and abs(new_lon - lon) < tol:
            lat, lon = new_lat, new_lon
            break
        lat, lon = new_lat, new_lon
    total_cost = sum(w[i] * haversine_miles(lat, lon, lats[i], lons[i]) for i in range(len(points)))
    return lat, lon, total_cost


def weighted_distance_cost(points, lat, lon):
    return sum(row["demand"] * haversine_miles(lat, lon, row["lat"], row["lon"]) for _, row in points.iterrows())


def weighted_kmeans_greenfield(points, k, max_iter=50, seed=42):
    rng = np.random.default_rng(seed)
    n = len(points)
    k = min(k, n)
    probs = points["demand"].values / points["demand"].sum()
    init_idx = rng.choice(n, size=k, replace=False, p=probs)
    centers = points.iloc[init_idx][["lat", "lon"]].values.astype(float)
    assignments = np.zeros(n, dtype=int)
    for iteration in range(max_iter):
        new_assignments = np.zeros(n, dtype=int)
        for i in range(n):
            dists = [haversine_miles(points.iloc[i]["lat"], points.iloc[i]["lon"], c[0], c[1]) for c in centers]
            new_assignments[i] = int(np.argmin(dists))
        if np.array_equal(new_assignments, assignments) and iteration > 0:
            assignments = new_assignments
            break
        assignments = new_assignments
        new_centers = []
        for c_idx in range(k):
            cluster_pts = points[assignments == c_idx]
            if len(cluster_pts) == 0:
                new_centers.append(centers[c_idx])
                continue
            lat, lon, _ = weiszfeld(cluster_pts)
            new_centers.append((lat, lon))
        centers = np.array(new_centers)
    results = []
    for c_idx in range(k):
        cluster_pts = points[assignments == c_idx]
        if len(cluster_pts) == 0:
            continue
        lat, lon = centers[c_idx]
        cost = sum(row["demand"] * haversine_miles(lat, lon, row["lat"], row["lon"]) for _, row in cluster_pts.iterrows())
        results.append({"facility": f"New Facility {c_idx + 1}", "lat": lat, "lon": lon,
                         "num_customers": len(cluster_pts), "total_demand": cluster_pts["demand"].sum(),
                         "weighted_distance_cost": cost})
    return pd.DataFrame(results), assignments


def make_cog_map(points, centers_df, assignments=None):
    fig = go.Figure()
    colors = ["crimson", "darkorange", "seagreen", "royalblue", "purple", "brown", "teal", "magenta"]
    if assignments is not None and len(centers_df) > 1:
        for c_idx in range(len(centers_df)):
            mask = assignments == c_idx
            cluster_pts = points[mask]
            color = colors[c_idx % len(colors)]
            center = centers_df.iloc[c_idx]
            for _, row in cluster_pts.iterrows():
                fig.add_trace(go.Scattergeo(lon=[row["lon"], center["lon"]], lat=[row["lat"], center["lat"]],
                                             mode="lines", line=dict(width=1, color=color), opacity=0.35,
                                             showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scattergeo(lon=cluster_pts["lon"], lat=cluster_pts["lat"],
                                         text=cluster_pts["name"] + "<br>demand: " + cluster_pts["demand"].astype(str),
                                         mode="markers", marker=dict(size=8, color=color), name=f"Customers → {center['facility']}"))
    else:
        center = centers_df.iloc[0]
        for _, row in points.iterrows():
            fig.add_trace(go.Scattergeo(lon=[row["lon"], center["lon"]], lat=[row["lat"], center["lat"]], mode="lines",
                                         line=dict(width=1, color="rgba(100,100,255,0.35)"), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scattergeo(lon=points["lon"], lat=points["lat"],
                                     text=points["name"] + "<br>demand: " + points["demand"].astype(str),
                                     mode="markers", marker=dict(size=8, color="dodgerblue"), name="Customers"))
    fig.add_trace(go.Scattergeo(lon=centers_df["lon"], lat=centers_df["lat"], text=centers_df["facility"],
                                 mode="markers+text", textposition="top center",
                                 marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
                                 name="Optimal Location(s)"))
    fig.update_layout(geo=dict(scope="world", projection_type="natural earth", showland=True,
                                landcolor="rgb(240,240,240)", countrycolor="rgb(200,200,200)"),
                       margin=dict(l=0, r=0, t=0, b=0), height=520,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# ==========================================================================================
# Engine 3: Inventory DES
# ==========================================================================================

class InventorySimulation:
    def __init__(self, policy, sim_duration, initial_inventory, demand_rate, demand_size_mean,
                 demand_size_std, lead_time_mean, lead_time_std, holding_cost_per_unit_per_time,
                 stockout_cost_per_unit, ordering_cost_per_order, s=None, Q=None, R=None, S=None, seed=42):
        self.policy = policy
        self.sim_duration = sim_duration
        self.demand_rate = demand_rate
        self.demand_size_mean = demand_size_mean
        self.demand_size_std = demand_size_std
        self.lead_time_mean = lead_time_mean
        self.lead_time_std = lead_time_std
        self.holding_cost = holding_cost_per_unit_per_time
        self.stockout_cost = stockout_cost_per_unit
        self.ordering_cost = ordering_cost_per_order
        self.s, self.Q, self.R, self.S = s, Q, R, S
        self.rng = np.random.default_rng(seed)
        self.on_hand = initial_inventory
        self.inventory_position = initial_inventory
        self.inventory_log = [(0.0, self.on_hand)]
        self.order_log = []
        self.total_demand_units = 0
        self.fulfilled_units = 0
        self.num_orders_placed = 0
        self.env = simpy.Environment()

    def demand_process(self):
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / self.demand_rate))
            qty = max(1, int(round(self.rng.normal(self.demand_size_mean, self.demand_size_std))))
            self.total_demand_units += qty
            fulfilled = min(qty, max(self.on_hand, 0))
            self.on_hand -= qty
            self.inventory_position -= qty
            if self.on_hand < 0:
                self.on_hand = 0
            self.fulfilled_units += fulfilled
            self.inventory_log.append((self.env.now, self.on_hand))
            if self.policy == "sQ" and self.inventory_position <= self.s:
                self.place_order(self.Q)

    def periodic_review_process(self):
        while True:
            yield self.env.timeout(self.R)
            order_qty = max(0, self.S - self.inventory_position)
            if order_qty > 0:
                self.place_order(order_qty)

    def place_order(self, qty):
        self.inventory_position += qty
        self.num_orders_placed += 1
        lt = max(0.01, self.rng.normal(self.lead_time_mean, self.lead_time_std))
        self.order_log.append((self.env.now, qty, self.env.now + lt))
        self.env.process(self.receive_order(qty, lt))

    def receive_order(self, qty, lead_time):
        yield self.env.timeout(lead_time)
        self.on_hand += qty
        self.inventory_log.append((self.env.now, self.on_hand))

    def run(self):
        self.env.process(self.demand_process())
        if self.policy == "RS":
            self.env.process(self.periodic_review_process())
        self.env.run(until=self.sim_duration)
        return self.results()

    def results(self):
        inv_df = pd.DataFrame(self.inventory_log, columns=["time", "on_hand"]).sort_values("time")
        inv_df["next_time"] = inv_df["time"].shift(-1).fillna(self.sim_duration)
        inv_df["duration"] = inv_df["next_time"] - inv_df["time"]
        avg_on_hand = (inv_df["on_hand"] * inv_df["duration"]).sum() / self.sim_duration
        fill_rate = self.fulfilled_units / self.total_demand_units if self.total_demand_units > 0 else 1.0
        stockout_units = self.total_demand_units - self.fulfilled_units
        holding_cost_total = avg_on_hand * self.holding_cost * self.sim_duration
        stockout_cost_total = stockout_units * self.stockout_cost
        ordering_cost_total = self.num_orders_placed * self.ordering_cost
        total_cost = holding_cost_total + stockout_cost_total + ordering_cost_total
        return {
            "inventory_log": inv_df, "order_log": pd.DataFrame(self.order_log, columns=["time_placed", "quantity", "time_arrives"]),
            "avg_on_hand": avg_on_hand, "fill_rate": fill_rate,
            "stockout_units": stockout_units, "num_orders_placed": self.num_orders_placed,
            "holding_cost_total": holding_cost_total, "stockout_cost_total": stockout_cost_total,
            "ordering_cost_total": ordering_cost_total, "total_cost": total_cost,
        }


def make_inventory_chart(inv_df, reorder_level=None, target_level=None, title="On-Hand Inventory Over Time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inv_df["time"], y=inv_df["on_hand"], mode="lines",
                              line=dict(shape="hv", color="royalblue", width=2), name="On-hand inventory",
                              fill="tozeroy", fillcolor="rgba(65,105,225,0.15)"))
    if reorder_level is not None:
        fig.add_hline(y=reorder_level, line_dash="dash", line_color="orange", annotation_text="Reorder point (s)")
    if target_level is not None:
        fig.add_hline(y=target_level, line_dash="dot", line_color="green", annotation_text="Order-up-to (S)")
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Units on hand", height=430, margin=dict(l=40, r=20, t=50, b=40))
    return fig


# ==========================================================================================
# Financial rollup helpers
# ==========================================================================================

def npv(rate, cashflows):
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def payback_period(implementation_cost, annual_savings):
    if annual_savings <= 0:
        return None
    return implementation_cost / annual_savings


def build_cashflows(annual_savings, implementation_cost, years):
    return [-implementation_cost] + [annual_savings] * years


# ==========================================================================================
# Session state init
# ==========================================================================================

if "initiatives" not in st.session_state:
    st.session_state["initiatives"] = {}


# ==========================================================================================
# PAGE 1: Network Optimization
# ==========================================================================================

def page_network():
    st.title("🏭 Network Optimization")
    st.caption("Capacitated facility location — decide which warehouses/plants to open to minimize total network cost. Optionally compare against your current ('baseline') network.")

    with st.sidebar:
        st.header("Data")
        use_sample = st.checkbox("Use sample data", value=True, key="net_sample")
        fac_file = st.file_uploader("Candidate facilities", type=["csv", "xlsx", "xls"], disabled=use_sample, key="net_fac_up")
        cust_file = st.file_uploader("Customers / demand", type=["csv", "xlsx", "xls"], disabled=use_sample, key="net_cust_up")
        st.markdown("---")
        st.header("Parameters")
        cost_per_mile = st.number_input("Transport cost (per unit demand per mile)", min_value=0.0, value=0.05, step=0.01, key="net_cost")
        max_fac = st.number_input("Max facilities to open (0 = no limit)", min_value=0, value=0, step=1, key="net_max")
        compare_baseline = st.checkbox("Compare against my current network (for Business Case)", value=False, key="net_compare")

    if use_sample:
        facilities, customers = sample_facilities(), sample_customers()
    else:
        facilities = read_uploaded_table(fac_file) if fac_file else None
        customers = read_uploaded_table(cust_file) if cust_file else None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Candidate Facilities")
        st.dataframe(facilities, use_container_width=True, height=200) if facilities is not None else st.info("Upload or use sample data.")
    with col2:
        st.subheader("Customer Demand")
        st.dataframe(customers, use_container_width=True, height=200) if customers is not None else st.info("Upload or use sample data.")

    if facilities is None or customers is None:
        return

    current_open = None
    if compare_baseline:
        current_open = st.multiselect("Facilities currently open (baseline)", options=list(facilities["facility_id"]),
                                       default=list(facilities["facility_id"])[:2], key="net_current_open")

    if st.button("🚀 Optimize network", type="primary", key="net_run"):
        with st.spinner("Solving..."):
            dist = build_distance_matrix(facilities, customers)
            optimized = solve_facility_location(facilities, customers, dist, cost_per_mile, max_facilities=(max_fac or None))
            baseline = None
            if compare_baseline and current_open:
                baseline = solve_facility_location(facilities, customers, dist, cost_per_mile, forced_open=current_open,
                                                    forced_closed=[f for f in facilities["facility_id"] if f not in current_open])
        st.session_state["net_result"] = (optimized, baseline, facilities, customers)

    if "net_result" in st.session_state:
        optimized, baseline, facilities, customers = st.session_state["net_result"]
        if optimized["status"] != "Optimal":
            st.warning(f"Solver status: {optimized['status']}.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Cost", f"${optimized['objective']:,.0f}")
            m2.metric("Facilities Opened", f"{len(optimized['opened'])} / {len(facilities)}")
            if baseline and baseline["status"] == "Optimal":
                savings = baseline["objective"] - optimized["objective"]
                m3.metric("Baseline Cost", f"${baseline['objective']:,.0f}")
                m4.metric("Annual Savings", f"${savings:,.0f}")

            st.subheader("🗺️ Network Map")
            st.plotly_chart(make_network_map(facilities, customers, optimized["opened"], optimized["assignment"]), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Opened Facilities")
                opened_df = facilities[facilities["facility_id"].isin(optimized["opened"])].copy()
                if not optimized["assignment"].empty:
                    served = optimized["assignment"].groupby("facility_id")["quantity"].sum()
                    opened_df["demand_served"] = opened_df["facility_id"].map(served).fillna(0)
                    opened_df["utilization_%"] = (opened_df["demand_served"] / opened_df["capacity"] * 100).round(1)
                st.dataframe(opened_df, use_container_width=True)
            with col_b:
                st.subheader("Assignment Plan")
                st.dataframe(optimized["assignment"], use_container_width=True)

            csv_buf = io.StringIO()
            optimized["assignment"].to_csv(csv_buf, index=False)
            st.download_button("Download assignment plan (CSV)", data=csv_buf.getvalue(), file_name="assignment_plan.csv", mime="text/csv")

            if baseline and baseline["status"] == "Optimal":
                st.markdown("---")
                impl_cost = st.number_input("One-time implementation cost for this network change ($)", min_value=0.0, value=0.0, step=10000.0, key="net_impl")
                if st.button("➕ Add to Business Case", key="net_add"):
                    st.session_state["initiatives"]["Network Optimization"] = {
                        "annual_savings": max(0.0, baseline["objective"] - optimized["objective"]),
                        "implementation_cost": impl_cost,
                        "detail": f"Baseline open: {', '.join(baseline['opened'])} → Optimized open: {', '.join(optimized['opened'])}",
                    }
                    st.success("Added to Business Case Builder page.")


# ==========================================================================================
# PAGE 2: Greenfield / COG
# ==========================================================================================

def page_greenfield():
    st.title("🎯 Center of Gravity / Greenfield Analysis")
    st.caption("Find the optimal location(s) for new facilities that minimize total weighted transportation distance to your customers.")

    with st.sidebar:
        st.header("Data")
        use_sample = st.checkbox("Use sample data", value=True, key="cog_sample")
        cust_file = st.file_uploader("Customers / demand points", type=["csv", "xlsx", "xls"], disabled=use_sample, key="cog_cust_up")
        st.markdown("---")
        st.header("Parameters")
        method = st.radio("Method", ["Weiszfeld (distance-optimal)", "Simple weighted average"], key="cog_method")
        num_facilities = st.number_input("Number of new facilities (K)", min_value=1, max_value=8, value=1, step=1, key="cog_k")
        cost_per_dist_unit = st.number_input("Cost per unit demand per distance unit ($)", min_value=0.0, value=0.05, step=0.01, key="cog_cost")
        compare_baseline = st.checkbox("Compare against my current site (for Business Case)", value=False, key="cog_compare")

    if use_sample:
        customers = sample_customers()
    else:
        customers = read_uploaded_table(cust_file) if cust_file else None

    st.subheader("Customer Demand")
    st.dataframe(customers, use_container_width=True, height=200) if customers is not None else st.info("Upload or use sample data.")
    if customers is None:
        return

    current_lat = current_lon = None
    if compare_baseline:
        c1, c2 = st.columns(2)
        current_lat = c1.number_input("Current facility latitude", value=39.0997, format="%.4f", key="cog_lat")
        current_lon = c2.number_input("Current facility longitude", value=-94.5786, format="%.4f", key="cog_lon")

    if st.button("🚀 Run analysis", type="primary", key="cog_run"):
        with st.spinner("Solving..."):
            if num_facilities == 1:
                if method.startswith("Weiszfeld"):
                    lat, lon, cost = weiszfeld(customers)
                else:
                    w = customers["demand"].values
                    lat = np.average(customers["lat"].values, weights=w)
                    lon = np.average(customers["lon"].values, weights=w)
                    cost = weighted_distance_cost(customers, lat, lon)
                centers_df = pd.DataFrame([{"facility": "New Facility 1", "lat": lat, "lon": lon,
                                             "num_customers": len(customers), "total_demand": customers["demand"].sum(),
                                             "weighted_distance_cost": cost}])
                assignments = None
            else:
                centers_df, assignments = weighted_kmeans_greenfield(customers, int(num_facilities))

            baseline_cost = None
            if compare_baseline and current_lat is not None:
                baseline_cost = weighted_distance_cost(customers, current_lat, current_lon)

        st.session_state["cog_result"] = (centers_df, assignments, customers, baseline_cost, cost_per_dist_unit, current_lat, current_lon)

    if "cog_result" in st.session_state:
        centers_df, assignments, customers, baseline_cost, cost_per_dist_unit, current_lat, current_lon = st.session_state["cog_result"]
        total_cost = centers_df["weighted_distance_cost"].sum() * cost_per_dist_unit

        m1, m2, m3 = st.columns(3)
        m1.metric("Facilities", len(centers_df))
        m2.metric("Total Transport Cost", f"${total_cost:,.0f}")
        if baseline_cost is not None:
            base_cost_dollars = baseline_cost * cost_per_dist_unit
            savings = base_cost_dollars - total_cost
            m3.metric("Annual Savings vs. Current Site", f"${savings:,.0f}")

        st.subheader("🗺️ Optimal Location(s)")
        st.plotly_chart(make_cog_map(customers, centers_df, assignments), use_container_width=True)

        st.subheader("📍 Recommended Locations")
        display_df = centers_df.copy()
        display_df["lat"] = display_df["lat"].round(5)
        display_df["lon"] = display_df["lon"].round(5)
        display_df["weighted_distance_cost"] = display_df["weighted_distance_cost"].round(1)
        st.dataframe(display_df, use_container_width=True)

        csv_buf = io.StringIO()
        display_df.to_csv(csv_buf, index=False)
        st.download_button("Download recommended locations (CSV)", data=csv_buf.getvalue(), file_name="greenfield_locations.csv", mime="text/csv")

        if baseline_cost is not None:
            st.markdown("---")
            impl_cost = st.number_input("One-time cost to relocate/build new facility ($)", min_value=0.0, value=0.0, step=10000.0, key="cog_impl")
            if st.button("➕ Add to Business Case", key="cog_add"):
                savings = baseline_cost * cost_per_dist_unit - total_cost
                st.session_state["initiatives"]["Greenfield / COG Relocation"] = {
                    "annual_savings": max(0.0, savings),
                    "implementation_cost": impl_cost,
                    "detail": f"Current site ({current_lat:.4f},{current_lon:.4f}) → Optimal site(s) recommended",
                }
                st.success("Added to Business Case Builder page.")


# ==========================================================================================
# PAGE 3: Inventory Policy
# ==========================================================================================

def page_inventory():
    st.title("📊 Inventory Discrete-Event Simulation")
    st.caption("Simulate a single-item inventory system with (s, Q) or (R, S) policies using true discrete-event simulation (SimPy).")

    with st.sidebar:
        st.header("Policy")
        policy_label = st.radio("Replenishment policy", ["(s, Q) — Reorder point / quantity", "(R, S) — Periodic review, order-up-to"], key="inv_policy_label")
        policy = "sQ" if policy_label.startswith("(s, Q)") else "RS"
        compare_baseline = st.checkbox("Compare against current policy (for Business Case)", value=False, key="inv_compare")

        st.markdown("---")
        st.header("Demand")
        demand_rate = st.number_input("Demand rate (events/day)", min_value=0.01, value=5.0, step=0.5, key="inv_dr")
        demand_size_mean = st.number_input("Avg demand size", min_value=1.0, value=10.0, step=1.0, key="inv_dsm")
        demand_size_std = st.number_input("Std dev demand size", min_value=0.0, value=3.0, step=0.5, key="inv_dss")

        st.markdown("---")
        st.header("Lead Time")
        lead_time_mean = st.number_input("Avg lead time (days)", min_value=0.1, value=5.0, step=0.5, key="inv_ltm")
        lead_time_std = st.number_input("Std dev lead time", min_value=0.0, value=1.0, step=0.5, key="inv_lts")

        st.markdown("---")
        st.header("Policy Parameters")
        initial_inventory = st.number_input("Initial inventory", min_value=0, value=200, step=10, key="inv_init")

        if policy == "sQ":
            s = st.number_input("Reorder point (s)", min_value=0, value=100, step=10, key="inv_s")
            Q = st.number_input("Reorder quantity (Q)", min_value=1, value=300, step=10, key="inv_Q")
            R, S = None, None
        else:
            R = st.number_input("Review period (R, days)", min_value=0.1, value=7.0, step=1.0, key="inv_R")
            S = st.number_input("Order-up-to level (S)", min_value=1, value=400, step=10, key="inv_S")
            s, Q = None, None

        if compare_baseline:
            st.markdown("**Baseline (current) parameters**")
            if policy == "sQ":
                base_s = st.number_input("Baseline s", min_value=0, value=80, step=10, key="inv_base_s")
                base_Q = st.number_input("Baseline Q", min_value=1, value=250, step=10, key="inv_base_Q")
                base_R, base_S = None, None
            else:
                base_R = st.number_input("Baseline R (days)", min_value=0.1, value=14.0, step=1.0, key="inv_base_R")
                base_S = st.number_input("Baseline S", min_value=1, value=350, step=10, key="inv_base_S")
                base_s, base_Q = None, None

        st.markdown("---")
        st.header("Costs")
        holding_cost = st.number_input("Holding cost (per unit per day)", min_value=0.0, value=0.05, step=0.01, format="%.4f", key="inv_hc")
        stockout_cost = st.number_input("Stockout cost (per unit short)", min_value=0.0, value=5.0, step=0.5, key="inv_sc")
        ordering_cost = st.number_input("Ordering cost (per order)", min_value=0.0, value=50.0, step=5.0, key="inv_oc")

        st.markdown("---")
        st.header("Simulation")
        sim_duration = st.number_input("Simulation duration (days)", min_value=10, value=365, step=10, key="inv_dur")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="inv_seed")
        run_clicked = st.button("🚀 Run simulation", type="primary", use_container_width=True, key="inv_run")

    if run_clicked:
        with st.spinner("Running discrete-event simulation..."):
            common = dict(sim_duration=float(sim_duration), initial_inventory=int(initial_inventory),
                          demand_rate=demand_rate, demand_size_mean=demand_size_mean, demand_size_std=demand_size_std,
                          lead_time_mean=lead_time_mean, lead_time_std=lead_time_std,
                          holding_cost_per_unit_per_time=holding_cost, stockout_cost_per_unit=stockout_cost,
                          ordering_cost_per_order=ordering_cost, seed=int(seed))
            sim = InventorySimulation(policy=policy, s=s, Q=Q, R=R, S=S, **common)
            results = sim.run()

            baseline_results = None
            if compare_baseline:
                base_sim = InventorySimulation(policy=policy, s=base_s if policy == "sQ" else None,
                                                Q=base_Q if policy == "sQ" else None,
                                                R=base_R if policy == "RS" else None,
                                                S=base_S if policy == "RS" else None, **common)
                baseline_results = base_sim.run()

        st.session_state["inv_results"] = (results, baseline_results, policy, s, S, sim_duration)

    if "inv_results" in st.session_state:
        results, baseline_results, policy, s_val, S_val, sim_duration = st.session_state["inv_results"]

        st.success(f"Simulation complete over {sim_duration:.0f} days.")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Fill Rate", f"{results['fill_rate']*100:.1f}%")
        m2.metric("Avg On-Hand", f"{results['avg_on_hand']:.1f}")
        m3.metric("Orders Placed", f"{results['num_orders_placed']}")
        m4.metric("Stockout Units", f"{results['stockout_units']:,.0f}")
        m5.metric("Total Cost", f"${results['total_cost']:,.0f}")

        annual_savings = None
        if baseline_results is not None:
            annualization = 365.0 / sim_duration
            base_annual_cost = baseline_results["total_cost"] * annualization
            prop_annual_cost = results["total_cost"] * annualization
            annual_savings = base_annual_cost - prop_annual_cost
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Annual Cost", f"${base_annual_cost:,.0f}", delta=f"Fill rate {baseline_results['fill_rate']*100:.1f}%")
            c2.metric("Proposed Annual Cost", f"${prop_annual_cost:,.0f}", delta=f"Fill rate {results['fill_rate']*100:.1f}%")
            c3.metric("Annual Savings", f"${annual_savings:,.0f}")

        st.subheader("📈 Inventory Over Time")
        reorder_level = s_val if policy == "sQ" else None
        target_level = S_val if policy == "RS" else None
        fig = make_inventory_chart(results["inventory_log"], reorder_level, target_level, "Proposed Policy — On-Hand Inventory")
        if baseline_results is not None:
            fig.add_trace(go.Scatter(x=baseline_results["inventory_log"]["time"], y=baseline_results["inventory_log"]["on_hand"],
                                      mode="lines", line=dict(shape="hv", color="gray", dash="dot"), name="Baseline"))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Order History")
            st.dataframe(results["order_log"], use_container_width=True, height=280)
        with col_b:
            st.subheader("Cost Breakdown")
            st.dataframe(pd.DataFrame([
                {"Cost Type": "Holding", "Amount": results["holding_cost_total"]},
                {"Cost Type": "Stockout", "Amount": results["stockout_cost_total"]},
                {"Cost Type": "Ordering", "Amount": results["ordering_cost_total"]},
            ]), use_container_width=True, height=140)

        buf = io.StringIO()
        results["inventory_log"][["time", "on_hand"]].to_csv(buf, index=False)
        st.download_button("Download inventory log (CSV)", data=buf.getvalue(), file_name="inventory_log.csv", mime="text/csv")

        if baseline_results is not None:
            st.markdown("---")
            impl_cost = st.number_input("One-time cost to implement new policy ($)", min_value=0.0, value=0.0, step=1000.0, key="inv_impl")
            if st.button("➕ Add to Business Case", key="inv_add"):
                st.session_state["initiatives"]["Inventory Policy Change"] = {
                    "annual_savings": max(0.0, annual_savings),
                    "implementation_cost": impl_cost,
                    "detail": f"Baseline fill rate {baseline_results['fill_rate']*100:.1f}% → Proposed fill rate {results['fill_rate']*100:.1f}%",
                }
                st.success("Added to Business Case Builder page.")


# ==========================================================================================
# PAGE 4: Business Case Builder
# ==========================================================================================

def page_business_case():
    st.title("💼 Business Case Builder")
    st.caption("Combines baseline-vs-optimized results from the other three modules into a multi-year NPV business case.")

    initiatives = st.session_state["initiatives"]
    if not initiatives:
        st.info("No initiatives yet. Go to the **Network Optimization**, **Greenfield / COG**, or **Inventory Policy** pages, check 'Compare against my current...', run the analysis, and click **➕ Add to Business Case**.")
        return

    st.markdown("**Included initiatives:**")
    summary_rows = [{"Initiative": name, "Annual Savings": d["annual_savings"], "Implementation Cost": d["implementation_cost"], "Detail": d["detail"]}
                     for name, d in initiatives.items()]
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    to_remove = st.multiselect("Remove initiatives", options=list(initiatives.keys()), key="bc_remove")
    if st.button("Remove selected"):
        for name in to_remove:
            del st.session_state["initiatives"][name]
        st.rerun()

    st.markdown("---")
    st.markdown("**Financial assumptions**")
    f1, f2 = st.columns(2)
    discount_rate = f1.number_input("Discount rate (annual, %)", min_value=0.0, value=10.0, step=0.5, key="bc_rate") / 100.0
    projection_years = f2.number_input("Projection horizon (years)", min_value=1, value=5, step=1, key="bc_years")

    total_annual_savings = summary_df["Annual Savings"].sum()
    total_impl_cost = summary_df["Implementation Cost"].sum()
    cashflows = build_cashflows(total_annual_savings, total_impl_cost, int(projection_years))
    total_npv = npv(discount_rate, cashflows)
    pb = payback_period(total_impl_cost, total_annual_savings)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Annual Savings", f"${total_annual_savings:,.0f}")
    m2.metric("Total Implementation Cost", f"${total_impl_cost:,.0f}")
    m3.metric(f"{projection_years}-Year NPV", f"${total_npv:,.0f}")
    m4.metric("Payback Period", f"{pb:.1f} yrs" if pb else "Immediate / N/A")

    cumulative = np.cumsum(cashflows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(cashflows))), y=cashflows, name="Annual Cash Flow", marker_color="lightblue"))
    fig.add_trace(go.Scatter(x=list(range(len(cashflows))), y=cumulative, mode="lines+markers", name="Cumulative Cash Flow", line=dict(color="crimson", width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Projected Cash Flow", xaxis_title="Year (0 = implementation)", yaxis_title="$", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📝 Business Case Summary")
    n = len(initiatives)
    st.markdown(f"""
This business case combines **{n} initiative{'s' if n != 1 else ''}** — {', '.join(initiatives.keys())}.

- **Combined annual savings**: ${total_annual_savings:,.0f} per year
- **Total one-time implementation investment**: ${total_impl_cost:,.0f}
- **Net present value over {projection_years} years** (at a {discount_rate*100:.1f}% discount rate): **${total_npv:,.0f}**
- **Estimated payback period**: {f"{pb:.1f} years" if pb else "N/A"}

{"This is a **positive-NPV** business case — the projected savings outweigh the cost of capital over the projection horizon." if total_npv > 0 else "This case shows a **negative NPV** at the current discount rate and horizon — consider a longer horizon, lower implementation cost, or re-scoping the initiative."}
    """)


# ==========================================================================================
# Navigation
# ==========================================================================================

PAGES = {
    "🏭 Network Optimization": page_network,
    "🎯 Greenfield / COG": page_greenfield,
    "📊 Inventory Policy": page_inventory,
    "💼 Business Case Builder": page_business_case,
}

st.sidebar.title("📦 Supply Chain Suite")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), key="nav")
st.sidebar.markdown("---")

PAGES[selection]()
