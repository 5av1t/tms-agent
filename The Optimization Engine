import os
import io
import pandas as pd
import pulp
from typing import Dict, Optional

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculates the distance between two points on Earth."""
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return 2 * R * asin(sqrt(a))

def _find_col(df: pd.DataFrame, candidates: list[str], contains: bool = False) -> Optional[str]:
    """Finds the first matching column name from a list of candidates."""
    cols = list(df.columns)
    norm_cols = {str(c).strip().lower(): str(c).strip() for c in cols}
    if contains:
        for c_norm, c_orig in norm_cols.items():
            if any(token in c_norm for token in candidates):
                return c_orig
    else:
        want = {c.lower() for c in candidates}
        for c_norm, c_orig in norm_cols.items():
            if c_norm in want:
                return c_orig
    return None

class SupplyChainModel:
    """
    Encapsulates the supply chain network data and the PuLP optimization model.
    """
    def __init__(self, data_path_or_file):
        self.case = self._read_case(data_path_or_file)

    def _read_case(self, data_path_or_file) -> dict:
        """Loads and processes the supply chain data from an Excel file."""
        try:
            xls = pd.ExcelFile(data_path_or_file)
        except Exception as e:
            raise FileNotFoundError(f"Could not read the Excel file. Ensure 'base_case.xlsx' is available. Error: {e}")

        # --- Read all sheets ---
        customers_df = pd.read_excel(xls, "Customers")
        demand_df = pd.read_excel(xls, "Customer Product Data")
        loc_df = pd.read_excel(xls, "Location")
        lanes_tpl = pd.read_excel(xls, "Transport Cost")
        groups_df = pd.read_excel(xls, "Location Groups")
        supp_prod = pd.read_excel(xls, "Supplier Product")
        wh_df = pd.read_excel(xls, "Warehouse")

        for df in (customers_df, demand_df, loc_df, lanes_tpl, groups_df, supp_prod, wh_df):
            df.columns = [str(c).strip() for c in df.columns]

        # --- Process data into structured format ---
        cust_id_col = _find_col(customers_df, ["customer", "name", "id"])
        customers = customers_df[cust_id_col].dropna().astype(str).tolist()

        dem_cust_col = _find_col(demand_df, ["customer", "name", "id"])
        dem_prod_col = _find_col(demand_df, ["product"])
        dem_qty_col = _find_col(demand_df, ["demand", "qty", "quantity", "volume"])
        dem = demand_df[[dem_cust_col, dem_prod_col, dem_qty_col]].copy()
        dem.columns = ["customer", "product", "demand"]
        dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)
        demand = dem.groupby(["customer", "product"])["demand"].sum().reset_index()

        loc_id_col = _find_col(loc_df, ["location", "name", "id"])
        lat_col = _find_col(loc_df, ["lat"], contains=True)
        lon_col = _find_col(loc_df, ["lon", "lng", "long"], contains=True)
        loc = loc_df[[loc_id_col, lat_col, lon_col]].dropna(subset=[loc_id_col]).copy()
        loc.columns = ["id", "lat", "lon"]
        coord = {r["id"]: {"lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in loc.iterrows()}

        FG = set(groups_df[groups_df["Location"] == "FG"]["SubLocation"].dropna().astype(str))
        DC = set(groups_df[groups_df["Location"] == "DC"]["SubLocation"].dropna().astype(str))

        sp = supp_prod.rename(columns={"Location": "supplier"}).copy()
        cap_col = _find_col(sp, ["Maximum Capacity", "Capacity", "Cap"])
        if cap_col is None: sp["Maximum Capacity"] = 0.0; cap_col = "Maximum Capacity"
        sp["supply"] = pd.to_numeric(sp[cap_col], errors="coerce").fillna(0.0)
        sp = sp[["supplier", "Product", "supply"]].rename(columns={"Product": "product"})
        supply = sp.groupby(["supplier", "product"])["supply"].sum().reset_index()

        products = sorted(demand["product"].dropna().astype(str).unique().tolist())
        
        nodes = []
        warehouses = set(wh_df.get("Location", pd.Series(dtype=str)).dropna().astype(str)) | DC
        for s in FG: nodes.append({"id": s, "lat": coord.get(s, {}).get('lat'), "lon": coord.get(s, {}).get('lon'), "kind": "supplier"})
        for d in warehouses: nodes.append({"id": d, "lat": coord.get(d, {}).get('lat'), "lon": coord.get(d, {}).get('lon'), "kind": "dc"})
        for cst in customers: nodes.append({"id": cst, "lat": coord.get(cst, {}).get('lat'), "lon": coord.get(cst, {}).get('lon'), "kind": "customer"})
        nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"])

        # --- Build Lanes ---
        tpl_fg_dc = {"cpd": 1.0, "cpu": 5.0}
        tpl_dc_ct = {"cpd": 2.0, "cpu": 10.0}
        lanes = []
        for s in FG:
            for d in warehouses:
                dist = haversine_km(coord[s]["lat"], coord[s]["lon"], coord[d]["lat"], coord[d]["lon"]) if s in coord and d in coord else 100
                for p in products:
                    sup_p = float(supply[(supply["supplier"] == s) & (supply["product"] == p)]["supply"].sum())
                    cost = tpl_fg_dc["cpu"] + tpl_fg_dc["cpd"] * dist
                    lanes.append({"src": s, "dst": d, "product": p, "capacity": max(0.0, sup_p), "cost_per_unit": cost})
        
        for d in warehouses:
            for cst in customers:
                dist = haversine_km(coord[d]["lat"], coord[d]["lon"], coord[cst]["lat"], coord[cst]["lon"]) if d in coord and cst in coord else 100
                for p in products:
                    dem_p = float(demand[(demand["customer"] == cst) & (demand["product"] == p)]["demand"].sum())
                    cost = tpl_dc_ct["cpu"] + tpl_dc_ct["cpd"] * dist
                    lanes.append({"src": d, "dst": cst, "product": p, "capacity": (dem_p * 2.0) + 1e6, "cost_per_unit": cost})

        return {"nodes": nodes_df, "demand": demand, "supply": supply, "lanes": pd.DataFrame(lanes), "products": products}

    def solve(self, shock: dict | None = None) -> dict:
        """Solves the min-cost flow problem with an optional disruption."""
        demand_df = self.case["demand"].copy()
        supply_df = self.case["supply"].copy()
        lanes_df = self.case["lanes"].copy()

        # --- Apply incident shock ---
        if shock:
            t = shock.get("type")
            if t == "lane_cap":
                cond = (lanes_df["src"] == shock["src"]) & (lanes_df["dst"] == shock["dst"])
                lanes_df.loc[cond, "capacity"] = float(shock["new_capacity"])
            elif t == "demand_spike":
                cond = (demand_df["customer"] == shock["customer"])
                demand_df.loc[cond, "demand"] *= (1.0 + float(shock["pct"]))
            # Add other shock types here if needed

        prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)
        
        # --- Variables ---
        lane_keys = [tuple(x) for x in lanes_df[['src', 'dst', 'product']].to_numpy()]
        flow_vars = pulp.LpVariable.dicts("flow", lane_keys, lowBound=0)
        
        supp_prod_keys = [tuple(x) for x in supply_df[['supplier', 'product']].to_numpy()]
        disposal_vars = pulp.LpVariable.dicts("disposal", supp_prod_keys, lowBound=0)
        
        cust_prod_keys = [tuple(x) for x in demand_df[['customer', 'product']].to_numpy()]
        shortage_vars = pulp.LpVariable.dicts("shortage", cust_prod_keys, lowBound=0)
        
        BIGM = 1e6

        # --- Objective Function ---
        cost_expr = pulp.lpSum(lanes_df.loc[i, 'cost_per_unit'] * flow_vars[key] for i, key in enumerate(lane_keys))
        prob += cost_expr + (BIGM * pulp.lpSum(disposal_vars)) + (BIGM * pulp.lpSum(shortage_vars))
        
        # --- Constraints ---
        nodes_by_id = self.case['nodes']['id'].tolist()
        for s, p in supp_prod_keys:
            outflow = pulp.lpSum(flow_vars.get((s, d, p), 0) for d in nodes_by_id)
            supply_val = float(supply_df[(supply_df["supplier"] == s) & (supply_df["product"] == p)]["supply"].sum())
            prob += outflow + disposal_vars[(s, p)] == supply_val, f"supply_{s}_{p}"

        for cst, p in cust_prod_keys:
            inflow = pulp.lpSum(flow_vars.get((s, cst, p), 0) for s in nodes_by_id)
            demand_val = float(demand_df[(demand_df["customer"] == cst) & (demand_df["product"] == p)]["demand"].sum())
            prob += inflow + shortage_vars[(cst, p)] == demand_val, f"demand_{cst}_{p}"

        for i, r in lanes_df.iterrows():
            key = (r['src'], r['dst'], r['product'])
            prob += flow_vars[key] <= r['capacity'], f"cap_{r['src']}_{r['dst']}_{r['product']}_{i}"
            
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # --- Extract Results ---
        is_optimal = pulp.LpStatus[prob.status] == 'Optimal'
        flows = []
        if is_optimal:
            for key, var in flow_vars.items():
                if var.value() is not None and var.value() > 1e-6:
                    flows.append({"src": key[0], "dst": key[1], "product": key[2], "flow": var.value()})
        
        return {
            "ok": is_optimal,
            "objective_cost": pulp.value(prob.objective) if is_optimal else None,
            "total_shortage": sum(v.value() for v in shortage_vars.values() if v.value() is not None),
            "total_disposal": sum(v.value() for v in disposal_vars.values() if v.value() is not None),
            "flows": flows,
            "nodes": self.case["nodes"].to_dict(orient="records"),
            "products": self.case["products"]
        }
