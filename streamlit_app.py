import os
import io
import json
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from scipy.optimize import linprog
import folium
from streamlit_folium import st_folium

# Optional LLM (Gemini) for narrative; app works without it
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit MUST start with set_page_config
# ──────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "TMS Agent — Control Tower (Gemini)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# App constants & small helpers
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "base_case.xlsx")
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

def num(x, default=0.0) -> float:
    """Coerce possibly None/NaN to float, with default for math."""
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float, np.floating)): return float(x)
        if isinstance(x, str) and x.strip()=="":
            return float(default)
        v = float(x)
        if np.isnan(v): return float(default)
        return v
    except Exception:
        return float(default)

def fmt(x, none="—", places=2):
    """Format number or return placeholder."""
    if x is None: return none
    try:
        v = float(x)
        if np.isnan(v): return none
        return f"{v:,.{places}f}"
    except Exception:
        return none

def safe_delta(after, before, none="—", places=2):
    if after is None or before is None: return none
    try:
        a, b = float(after), float(before)
        if any(map(np.isnan, [a,b])): return none
        return f"{(a-b):+,.{places}f}"
    except Exception:
        return none

def ensure_state():
    ss = st.session_state
    ss.setdefault("case_loaded_from", None)   # "repo" | "upload"
    ss.setdefault("case", None)               # parsed inputs
    ss.setdefault("base", None)               # baseline solve payload
    ss.setdefault("maps", {})                 # map cache
    ss.setdefault("payloads", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("explanations", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("phase", "idle")            # idle|baseline|incident1|incident2|incident3
    ss.setdefault("gem_model", "gemini-1.5-pro")
    ss.setdefault("gem_temperature", 0.1)
    ss.setdefault("gem_max_tokens", 300)

ensure_state()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s): return str(s).strip().lower()

def _find_col(df: pd.DataFrame, candidates: list[str], contains: bool=False) -> Optional[str]:
    cols = list(df.columns)
    if contains:
        for c in cols:
            if any(token in _norm(c) for token in candidates):
                return c
    else:
        for c in cols:
            if _norm(c) in [_norm(t) for t in candidates]:
                return c
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R=6371.0
    dlat=radians(lat2-lat1)
    dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

# ──────────────────────────────────────────────────────────────────────────────
# Load Excel & parse your schema (AIMMS-like demo workbook)
# ──────────────────────────────────────────────────────────────────────────────
def load_excel() -> pd.ExcelFile:
    # Try repo file first
    if os.path.exists(DATA_PATH):
        return pd.ExcelFile(DATA_PATH)

    # Otherwise allow upload
    up = st.file_uploader("Upload base_case.xlsx", type=["xlsx"])
    if up is None:
        st.error("Data file missing. Add `data/base_case.xlsx` to the repo or upload it here.")
        st.stop()
    return pd.ExcelFile(io.BytesIO(up.read()))

def read_case() -> dict:
    xls = load_excel()
    st.session_state["case_loaded_from"] = "repo" if os.path.exists(DATA_PATH) else "upload"

    # Load required sheets
    customers_df = pd.read_excel(xls, "Customers")
    demand_df    = pd.read_excel(xls, "Customer Product Data")
    loc_df       = pd.read_excel(xls, "Location")
    lanes_tpl    = pd.read_excel(xls, "Transport Cost")
    groups_df    = pd.read_excel(xls, "Location Groups")
    supp_prod    = pd.read_excel(xls, "Supplier Product")
    wh_df        = pd.read_excel(xls, "Warehouse")  # optional; used to classify DCs

    # Normalize columns
    for df in (customers_df, demand_df, loc_df, lanes_tpl, groups_df, supp_prod, wh_df):
        df.columns = [str(c).strip() for c in df.columns]

    # Customers
    cust_id_col = _find_col(customers_df, ["customer","name","id"])
    if not cust_id_col:
        st.error("Customers sheet must contain a Customer/Name/ID column.")
        st.stop()
    customers = customers_df[cust_id_col].dropna().astype(str).tolist()

    # Demand
    dem_cust_col = _find_col(demand_df, ["customer","name","id"])
    dem_prod_col = _find_col(demand_df, ["product"])
    dem_qty_col  = _find_col(demand_df, ["demand","qty","quantity","volume"])
    if not all([dem_cust_col, dem_prod_col, dem_qty_col]):
        st.error("Customer Product Data must contain Customer, Product, Demand.")
        st.stop()
    dem = demand_df[[dem_cust_col, dem_prod_col, dem_qty_col]].copy()
    dem.columns = ["customer","product","demand"]
    dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)
    demand = dem.groupby(["customer","product"])["demand"].sum().reset_index()

    # Locations (lat/lon)
    loc_id_col = _find_col(loc_df, ["location","name","id"])
    lat_col = _find_col(loc_df, ["lat"], contains=True)
    lon_col = _find_col(loc_df, ["lon","lng","long"], contains=True)
    if not all([loc_id_col, lat_col, lon_col]):
        st.error("Location sheet must contain Location, Latitude, Longitude.")
        st.stop()
    loc = loc_df[[loc_id_col, lat_col, lon_col]].dropna(subset=[loc_id_col]).copy()
    loc.columns = ["id","lat","lon"]
    coord = {r["id"]: {"lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in loc.iterrows()}

    # Location groups (FG/DC)
    if "Location" not in groups_df.columns or "SubLocation" not in groups_df.columns:
        st.error("Location Groups sheet must contain columns Location and SubLocation.")
        st.stop()
    FG = set(groups_df[groups_df["Location"]=="FG"]["SubLocation"].dropna().astype(str))
    DC = set(groups_df[groups_df["Location"]=="DC"]["SubLocation"].dropna().astype(str))

    # Suppliers & supply by product at FG locations
    sp = supp_prod.copy()
    sp = sp.rename(columns={"Location":"supplier"})
    if not {"supplier","Product"}.issubset(set(sp.columns)):
        st.error("Supplier Product sheet must include Location (or supplier) and Product.")
        st.stop()
    cap_col = _find_col(sp, ["Maximum Capacity","Capacity","Cap"])
    if cap_col is None:
        sp["Maximum Capacity"] = 0.0
        cap_col = "Maximum Capacity"
    sp["supply"] = pd.to_numeric(sp[cap_col], errors="coerce").fillna(0.0)
    if "Available" in sp.columns:
        sp["avail"] = pd.to_numeric(sp["Available"], errors="coerce").fillna(1.0)
        sp = sp[sp["avail"] > 0]
    sp = sp[["supplier","Product","supply"]].rename(columns={"Product":"product"})
    sp["supplier"] = sp["supplier"].astype(str)
    sp = sp[sp["supplier"].isin(FG)]
    supply = sp.groupby(["supplier","product"])["supply"].sum().reset_index()

    # Products set
    products = sorted(demand["product"].dropna().astype(str).unique().tolist())

    # Template transport cost: use two rows (FG→DC and DC→City)
    def _tpl_row(fr, to):
        if "From Location" in lanes_tpl.columns and "To Location" in lanes_tpl.columns:
            hit = lanes_tpl[(lanes_tpl.get("From Location")==fr) & (lanes_tpl.get("To Location")==to)]
        else:
            hit = pd.DataFrame()
        if len(hit) > 0:
            r = hit.iloc[0].to_dict()
            return {
                "cpd": float(r.get("Cost per Distance", 1.0) or 1.0),
                "cpu": float(r.get("Cost Per UOM", 0.0) or 0.0)
            }
        # defaults if template not present
        return {"cpd": 1.0 if fr=="FG" else 2.0, "cpu": 5.0 if fr=="FG" else 10.0}

    tpl_fg_dc = _tpl_row("FG","DC")
    tpl_dc_ct = _tpl_row("DC","City")

    # Build nodes
    nodes = []
    warehouses = set(wh_df.get("Location", pd.Series(dtype=str)).dropna().astype(str)) | DC
    for s in FG:
        coords = coord.get(s)
        nodes.append({"id": s, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "supplier"})
    for d in warehouses:
        coords = coord.get(d)
        nodes.append({"id": d, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "dc"})
    for c in customers:
        coords = coord.get(c)
        nodes.append({"id": c, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "customer"})
    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"])

    # Build lanes programmatically
    lanes = []
    # FG -> DC
    for s in FG:
        if s not in coord: continue
        for d in warehouses:
            if d not in coord: continue
            dist = haversine_km(coord[s]["lat"], coord[s]["lon"], coord[d]["lat"], coord[d]["lon"])
            for p in products:
                sup_p = float(supply[(supply["supplier"]==s) & (supply["product"]==p)]["supply"].sum())
                cap = sup_p if sup_p > 0 else 0.0
                cost = tpl_fg_dc["cpu"] + tpl_fg_dc["cpd"] * dist
                lanes.append({"src": s, "dst": d, "product": p, "capacity": cap, "cost_per_unit": cost})

    # DC -> City
    for d in warehouses:
        if d not in coord: continue
        for c in customers:
            if c not in coord: continue
            dist = haversine_km(coord[d]["lat"], coord[d]["lon"], coord[c]["lat"], coord[c]["lon"])
            for p in products:
                dem_p = float(demand[(demand["customer"]==c) & (demand["product"]==p)]["demand"].sum())
                cap = (dem_p * 2.0) + 1e6  # generous cap
                cost = tpl_dc_ct["cpu"] + tpl_dc_ct["cpd"] * dist
                lanes.append({"src": d, "dst": c, "product": p, "capacity": cap, "cost_per_unit": cost})

    return {
        "nodes": nodes_df,
        "demand": demand,             # customer, product, demand
        "supply": supply,             # supplier, product, supply
        "lanes": pd.DataFrame(lanes), # src, dst, product, capacity, cost_per_unit
        "products": products
    }

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (LP with shortage/disposal slacks)
# ──────────────────────────────────────────────────────────────────────────────
def solve_min_cost_flow(case: dict, shock: dict | None = None) -> dict:
    nodes = case["nodes"]
    demand_df = case["demand"].copy()
    supply_df = case["supply"].copy()
    lanes_df  = case["lanes"].copy()

    # Apply incident shock
    if shock:
        t = shock.get("type")
        if t == "lane_cap":
            cond = (lanes_df["src"] == shock["src"]) & (lanes_df["dst"] == shock["dst"])
            if shock.get("product") is not None:
                cond &= (lanes_df["product"] == shock["product"])
            lanes_df.loc[cond, "capacity"] = float(shock["new_capacity"])
        elif t == "demand_spike":
            cond = (demand_df["customer"] == shock["customer"])
            demand_df.loc[cond, "demand"] = demand_df.loc[cond, "demand"] * (1.0 + float(shock["pct"]))
        elif t == "express_lane":
            newrow = {
                "src": shock["src"], "dst": shock["dst"], "product": shock["product"],
                "capacity": float(shock["capacity"]), "cost_per_unit": float(shock["cost_per_unit"])
            }
            lanes_df = pd.concat([lanes_df, pd.DataFrame([newrow])], ignore_index=True)

    # Decision vars x[src,dst,product]
    lanes = lanes_df[["src","dst","product","capacity","cost_per_unit"]].to_dict(orient="records")
    var_keys = [(l["src"], l["dst"], l["product"]) for l in lanes]
    vidx = {k:i for i,k in enumerate(var_keys)}
    n_x = len(var_keys)

    # Slack vars
    cust_prod = sorted({(r["customer"], r["product"]) for _, r in demand_df.iterrows()})
    supp_prod = sorted({(r["supplier"], r["product"]) for _, r in supply_df.iterrows()})
    sp_index = {k: n_x + i for i, k in enumerate(supp_prod)}                 # disposal
    sh_index = {k: n_x + len(sp_index) + i for i, k in enumerate(cust_prod)} # shortage
    n_vars = n_x + len(sp_index) + len(sh_index)

    # Objective
    BIGM = 1e6
    c = np.zeros(n_vars)
    for (src,dst,pr), i in vidx.items():
        row = next(r for r in lanes if r["src"]==src and r["dst"]==dst and r["product"]==pr)
        c[i] = float(row["cost_per_unit"])
    for _, i in sp_index.items():
        c[i] = BIGM
    for _, i in sh_index.items():
        c[i] = BIGM

    # Constraints
    A_eq = []; b_eq = []

    # Supply balance: sum outflow + disposal = supply
    for s,p in supp_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if src == s and pr == p:
                row[i] = 1.0
        row[sp_index[(s,p)]] = 1.0
        sup_val = float(supply_df[(supply_df["supplier"]==s) & (supply_df["product"]==p)]["supply"].sum())
        A_eq.append(row); b_eq.append(sup_val)

    # Demand balance: sum inflow + shortage = demand
    for cst,p in cust_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if dst == cst and pr == p:
                row[i] = 1.0
        row[sh_index[(cst,p)]] = 1.0
        dem_val = float(demand_df[(demand_df["customer"]==cst) & (demand_df["product"]==p)]["demand"].sum())
        A_eq.append(row); b_eq.append(dem_val)

    # Capacity: x <= capacity
    A_ub = []; b_ub = []
    for (src,dst,pr), i in vidx.items():
        row = np.zeros(n_vars); row[i] = 1.0
        cap = float(next(r for r in lanes if r["src"]==src and r["dst"]==dst and r["product"]==pr)["capacity"])
        A_ub.append(row); b_ub.append(cap)

    bounds = [(0, None)] * n_vars

    res = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if A_ub else None,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )

    total_shortage = 0.0
    total_disposal = 0.0
    flows = []
    objective = math.inf

    if res.success:
        x = res.x
        for (src,dst,pr), i in vidx.items():
            q = float(x[i])
            if q > 1e-6:
                flows.append({"src": src, "dst": dst, "product": pr, "flow": q})
        for (s,p), i in sp_index.items():
            total_disposal += float(x[i])
        for (cst,p), i in sh_index.items():
            total_shortage += float(x[i])
        objective = float(res.fun)

    # Nodes for map
    nrecs = []
    for _, r in nodes.iterrows():
        nrecs.append({
            "id": str(r.get("id")),
            "kind": str(r.get("kind")),
            "lat": float(r["lat"]) if pd.notna(r["lat"]) else None,
            "lon": float(r["lon"]) if pd.notna(r["lon"]) else None
        })

    products_list = case.get("products", sorted(set([f["product"] for f in flows])))

    return {
        "ok": bool(res.success),
        "objective_cost": objective if res.success else None,
        "used_slacks": (total_disposal > 1e-6) or (total_shortage > 1e-6),
        "total_shortage": total_shortage if res.success else None,
        "total_disposal": total_disposal if res.success else None,
        "nodes": nrecs,
        "products": products_list,
        "flows": flows if res.success else [],
        "currency": "EUR",
        "flow_unit": "units"
    }

# ──────────────────────────────────────────────────────────────────────────────
# Incidents
# ──────────────────────────────────────────────────────────────────────────────
def build_baseline() -> tuple[dict, dict]:
    case = read_case()
    base = solve_min_cost_flow(case, None)
    base["title"] = "Baseline"
    return base, case

def incident_1(case: dict, prev: dict) -> dict:
    flows_prev = prev.get("flows_after") or prev.get("flows") or []
    if not flows_prev:
        return {"title":"Incident 1","objective_before":prev.get("objective_after"),"objective_after":prev.get("objective_after"),"flows_after":flows_prev}
    biggest = sorted(flows_prev, key=lambda x: x["flow"], reverse=True)[0]
    shock = {"type": "lane_cap", "src": biggest["src"], "dst": biggest["dst"], "product": biggest["product"], "new_capacity": max(0.5*biggest["flow"], 1.0)}
    before = prev.get("objective_after")
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 1 — Corridor capacity restriction",
        "lane": {"src": shock["src"], "dst": shock["dst"], "product": biggest["product"]},
        "objective_before": before,
        "objective_after": aft.get("objective_cost"),
        "used_slacks": aft.get("used_slacks"),
        "total_shortage": aft.get("total_shortage"),
        "total_disposal": aft.get("total_disposal"),
        "flows_after": aft.get("flows", []),
        "currency": aft.get("currency"), "flow_unit": aft.get("flow_unit")
    }

def incident_2(case: dict, prev: dict) -> dict:
    flows_prev = prev.get("flows_after") or prev.get("flows") or []
    if not flows_prev:
        return {"title":"Incident 2","objective_before":prev.get("objective_after"),"objective_after":prev.get("objective_after"),"flows_after":flows_prev}
    dest_tot = {}
    for f in flows_prev:
        dest_tot[f["dst"]] = dest_tot.get(f["dst"], 0.0) + f["flow"]
    customer = sorted(dest_tot.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    shock = {"type": "demand_spike", "customer": customer, "pct": 0.25}
    before = prev.get("objective_after")
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 2 — Demand surge (+25%)",
        "customer": customer,
        "objective_before": before,
        "objective_after": aft.get("objective_cost"),
        "used_slacks": aft.get("used_slacks"),
        "total_shortage": aft.get("total_shortage"),
        "total_disposal": aft.get("total_disposal"),
        "flows_after": aft.get("flows", []),
        "currency": aft.get("currency"), "flow_unit": aft.get("flow_unit")
    }

def incident_3(case: dict, prev: dict) -> dict:
    flows_prev = prev.get("flows_after") or prev.get("flows") or []
    if not flows_prev:
        return {"title":"Incident 3","objective_before":prev.get("objective_after"),"objective_after":prev.get("objective_after"),"flows_after":flows_prev}
    top = sorted(flows_prev, key=lambda x: x["flow"], reverse=True)[0]
    src, dst, pr = top["src"], top["dst"], top["product"]
    shock = {"type": "express_lane", "src": src, "dst": dst, "product": pr,
             "capacity": max(top["flow"]*0.5, 1.0), "cost_per_unit": 0.5}
    before = prev.get("objective_after")
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 3 — Strategic express lane",
        "lane": {"src": src, "dst": dst, "cost_per_unit": shock["cost_per_unit"], "capacity": shock["capacity"]},
        "objective_before": before,
        "objective_after": aft.get("objective_cost"),
        "used_slacks": aft.get("used_slacks"),
        "total_shortage": aft.get("total_shortage"),
        "total_disposal": aft.get("total_disposal"),
        "flows_after": aft.get("flows", []),
        "currency": aft.get("currency"), "flow_unit": aft.get("flow_unit")
    }

# ──────────────────────────────────────────────────────────────────────────────
# Maps & Tables
# ──────────────────────────────────────────────────────────────────────────────
def _center_latlon(nodes: List[Dict]) -> Tuple[float,float]:
    lats = [n.get("lat") for n in nodes if n.get("lat") is not None]
    lons = [n.get("lon") for n in nodes if n.get("lon") is not None]
    if not lats or not lons: return (20.5937, 78.9629)  # India fallback
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def _product_color_map(products: List[str]) -> Dict[str,str]:
    return {p: PALETTE[i % len(PALETTE)] for i,p in enumerate(sorted(products))}

def _node_by_id(nodes: List[Dict]) -> Dict[str,Dict]:
    return {n["id"]: n for n in nodes}

def _flow_weight_scaler(flows: List[Dict], min_w: float = 0.6, max_w: float = 2.2):
    vals = sorted(float(f.get("flow", 0.0)) for f in flows if f.get("flow", 0.0) > 0)
    if not vals: return lambda x: min_w
    p90 = vals[min(int(0.9*(len(vals)-1)), len(vals)-1)]
    p90 = p90 if p90 > 0 else (max(vals) if vals else 1.0)
    def to_w(q):
        if q <= 0: return 0
        w = min_w + (max_w - min_w) * (q/(p90*1.0))
        return max(min_w, min(max_w, w))
    return to_w

def build_map(nodes: List[Dict], flows: List[Dict], products: List[str]) -> folium.Map:
    node_by_id = _node_by_id(nodes)
    lat0, lon0 = _center_latlon(nodes)
    m = folium.Map(location=[lat0,lon0], zoom_start=5, tiles="cartodbpositron")

    # nodes
    for n in nodes:
        if n.get("lat") is None or n.get("lon") is None:
            continue
        kind = n.get("kind","")
        color = "#198754" if kind not in ("supplier","dc") else ("#6f42c1" if kind=="supplier" else "#0d6efd")
        radius = 6 if kind in ("supplier","dc") else 5
        folium.CircleMarker(
            (n["lat"], n["lon"]), radius=radius, color="#1d1d1d", weight=1,
            fill=True, fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(html=f"{n['id']} ({kind})", max_width=250)
        ).add_to(m)

    # flows
    to_w = _flow_weight_scaler(flows)
    color_by_product = _product_color_map(products)
    for f in flows:
        q = float(f.get("flow",0))
        if q <= 0: continue
        u, v = f["src"], f["dst"]
        urec, vrec = node_by_id.get(u), node_by_id.get(v)
        if not urec or not vrec: continue
        if None in (urec.get("lat"), urec.get("lon"), vrec.get("lat"), vrec.get("lon")): continue
        folium.PolyLine(
            [(urec["lat"], urec["lon"]), (vrec["lat"], vrec["lon"])],
            weight=to_w(q), color=color_by_product.get(f.get("product",""), "#555"),
            opacity=0.55, tooltip=f"{u} → {v} | {f.get('product','')}: {q:,.2f}"
        ).add_to(m)
    return m

def show_flows_table(flows: list[dict], caption="Solved flows"):
    if not flows:
        st.info("No flows to display.")
        return
    df = pd.DataFrame(flows).sort_values(["src","dst","product"])
    st.dataframe(df, use_container_width=True, height=300)
    st.caption(caption)

# ──────────────────────────────────────────────────────────────────────────────
# LLM Narrative (Gemini) with deterministic fallback & light guardrails
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a supply-chain control-tower analyst. Domain: freight TMS (not passengers).\n"
    "Use ONLY the JSON fields provided. If a field is missing, write 'insufficient data'.\n"
    "Echo numbers and units exactly. Do not invent currencies or people-related terms.\n"
    "Output format:\n"
    "VERIFIED\n"
    "- objective_before: <value>\n"
    "- objective_after: <value>\n"
    "- delta: <value>\n"
    "- lane: <src> -> <dst>\n"
    "- cost_per_unit: <value>\n"
    "- capacity: <value>\n"
    "- used_slacks: <true/false>\n"
    "- units: currency=<...>, flow_unit=<...>\n\n"
    "EXPLANATION\n"
    "- Why the objective changed or not, using only these facts.\n"
    "- Options considered (only if supported by data).\n"
    "- Final decision and reasoning.\n"
    "- Risks and next steps.\n"
)

def deterministic_explanation(idx: int, payload: dict) -> str:
    before = payload.get("objective_before")
    after  = payload.get("objective_after")
    bits = [payload.get("title", f"Incident {idx}"), "."]
    if before is not None and after is not None:
        bits.append(f" Objective moved {fmt(before)} → {fmt(after)} (Δ {safe_delta(after,before)}).")
    lane = payload.get("lane")
    if lane:
        parts=[]
        if "src" in lane and "dst" in lane: parts.append(f"{lane['src']}→{lane['dst']}")
        if "cost_per_unit" in lane: parts.append(f"cpu={fmt(lane['cost_per_unit'])}")
        if "capacity" in lane: parts.append(f"cap={fmt(lane['capacity'])}")
        if parts: bits.append(" Lane: " + ", ".join(parts) + ".")
    if payload.get("customer"): bits.append(f" Customer: {payload['customer']}.")
    if payload.get("used_slacks"): bits.append(" Feasibility fallback applied.")
    return "".join(bits)

def gemini_explain(idx: int, payload: dict) -> str:
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or genai is None:
        return deterministic_explanation(idx, payload)
    try:
        genai.configure(api_key=api_key)
        model_name = st.session_state.get("gem_model", "gemini-1.5-pro")
        temperature = float(st.session_state.get("gem_temperature", 0.1))
        max_tokens = int(st.session_state.get("gem_max_tokens", 300))

        model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
        ctx = {
            "incident_index": idx,
            "title": payload.get("title"),
            "objective_before": payload.get("objective_before"),
            "objective_after": payload.get("objective_after"),
            "used_slacks": payload.get("used_slacks"),
            "total_shortage": payload.get("total_shortage"),
            "total_disposal": payload.get("total_disposal"),
            "lane": payload.get("lane"),
            "customer": payload.get("customer"),
            "units": {"currency": payload.get("currency","EUR"), "flow_unit": payload.get("flow_unit","units")}
        }
        prompt = f"USER JSON:\n{json.dumps(ctx, ensure_ascii=False)}"
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        text = (resp.text or "").strip()
        if any(w in text.lower() for w in ["passenger", "ridership", "ticket"]):
            return deterministic_explanation(idx, payload)
        return text if text else deterministic_explanation(idx, payload)
    except Exception:
        return deterministic_explanation(idx, payload)

def ensure_explanation(idx: int, payload: dict):
    key = str(idx)
    if st.session_state["explanations"].get(key):  # cached
        return
    st.session_state["explanations"][key] = gemini_explain(idx, payload)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title(PAGE_TITLE)

with st.sidebar:
    st.subheader("Gemini (for explanations)")
    st.caption("Add GOOGLE_API_KEY in Streamlit Secrets. If absent, deterministic narrative is used.")
    st.session_state["gem_model"] = st.text_input("Model", value=st.session_state.get("gem_model","gemini-1.5-pro"))
    st.session_state["gem_temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("gem_temperature",0.1)), 0.05)
    st.session_state["gem_max_tokens"] = st.number_input("Max tokens", 64, 2048, int(st.session_state.get("gem_max_tokens",300)), 32)

st.divider()
agent = st.container()
with agent:
    st.subheader("Agent Narrative")

st.divider()
c1, c2 = st.columns([1,1])
with c1:
    if st.button("🚀 Bootstrap baseline (solve from Excel)"):
        base, case = build_baseline()
        st.session_state["case"] = case
        st.session_state["base"] = base

        base_obj = base.get("objective_cost")
        st.session_state["payloads"]["0"] = {
            "title": "Baseline",
            "objective_before": base_obj,
            "objective_after": base_obj,
            "flows_after": base.get("flows", []),
            "currency": base.get("currency"), "flow_unit": base.get("flow_unit")
        }
        st.session_state["phase"] = "baseline"
        st.session_state["maps"]["0"] = build_map(base.get("nodes", []), base.get("flows", []), base.get("products", []))
        if base.get("ok"):
            st.success(f"Baseline solved (source: {st.session_state['case_loaded_from']}). Objective {fmt(base_obj)}.")
        else:
            st.warning("Baseline infeasible. Slacks required or data incomplete. You can still run incidents to test the narrative.")

with c2:
    if st.button("🚨 Run Incident 1 now"):
        if not st.session_state.get("base"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            prev = st.session_state["payloads"]["0"] or {}
            resp1 = incident_1(case, prev)
            st.session_state["payloads"]["1"] = resp1
            st.session_state["phase"] = "incident1"
            flows1 = resp1.get("flows_after", [])
            st.session_state["maps"]["1"] = build_map(st.session_state["base"]["nodes"], flows1, st.session_state["base"]["products"])
            ensure_explanation(1, resp1)
            st.success("Incident 1 executed.")

with st.sidebar:
    st.divider()
    st.caption("Run additional incidents")
    if st.button("▶️ Incident 2 (demand surge)"):
        if not st.session_state.get("base"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            prev = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"] or {}
            resp2 = incident_2(case, prev)
            st.session_state["payloads"]["2"] = resp2
            st.session_state["phase"] = "incident2"
            flows2 = resp2.get("flows_after", [])
            st.session_state["maps"]["2"] = build_map(st.session_state["base"]["nodes"], flows2, st.session_state["base"]["products"])
            ensure_explanation(2, resp2)
            st.success("Incident 2 executed.")
    if st.button("▶️ Incident 3 (strategic express lane)"):
        if not st.session_state.get("base"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            prev = st.session_state["payloads"]["2"] or st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"] or {}
            resp3 = incident_3(case, prev)
            st.session_state["payloads"]["3"] = resp3
            st.session_state["phase"] = "incident3"
            flows3 = resp3.get("flows_after", [])
            st.session_state["maps"]["3"] = build_map(st.session_state["base"]["nodes"], flows3, st.session_state["base"]["products"])
            ensure_explanation(3, resp3)
            st.success("Incident 3 executed.")

# Agent narrative (safe formatting everywhere)
with agent:
    ph = st.session_state["phase"]
    if ph == "baseline" and st.session_state["payloads"]["0"]:
        p = st.session_state["payloads"]["0"]
        st.write(f"**Agent**: Baseline active. Objective **{fmt(p.get('objective_after'))}**. Monitoring lanes & capacity headroom.")
    elif ph == "incident1" and st.session_state["payloads"]["1"]:
        p = st.session_state["payloads"]["1"]
        st.write(f"**Agent**: {p.get('title','Incident 1')} | Objective delta **{safe_delta(p.get('objective_after'), p.get('objective_before'))}**.")
        st.markdown(st.session_state["explanations"].get("1") or "")
    elif ph == "incident2" and st.session_state["payloads"]["2"]:
        p = st.session_state["payloads"]["2"]
        st.write(f"**Agent**: {p.get('title','Incident 2')} | Objective delta **{safe_delta(p.get('objective_after'), p.get('objective_before'))}**.")
        st.markdown(st.session_state["explanations"].get("2") or "")
    elif ph == "incident3" and st.session_state["payloads"]["3"]:
        p = st.session_state["payloads"]["3"]
        st.write(f"**Agent**: {p.get('title','Incident 3')} | Objective delta **{safe_delta(p.get('objective_after'), p.get('objective_before'))}**.")
        st.markdown(st.session_state["explanations"].get("3") or "")
    else:
        st.caption("Ready. Solve baseline, then run incidents.")

# Baseline section
if st.session_state.get("base"):
    base = st.session_state["base"]
    st.subheader("Baseline")
    st.caption(f"Products: {', '.join(base.get('products', []))} | Nodes: {len(base.get('nodes', []))} | Flows: {len(base.get('flows', []))}")
    with st.container(border=True):
        st.markdown("**Network Map (Baseline)**")
        st_folium(st.session_state["maps"]["0"], width=None, height=520, key="map_base")
    show_flows_table(base.get("flows", []), caption="Baseline solved flows")

# Incident 1
if st.session_state["payloads"]["1"]:
    p1 = st.session_state["payloads"]["1"]
    st.divider()
    st.subheader("Incident 1 — Corridor capacity restriction")
    before, after = p1.get("objective_before"), p1.get("objective_after")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", fmt(before))
    c2.metric("Objective (after)", fmt(after), delta=safe_delta(after, before))
    c3.metric("Δ (after-before)", safe_delta(after, before))
    if p1.get("lane"):
        st.caption(f"Lane impacted: {p1['lane']['src']} → {p1['lane']['dst']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 1)**")
        st_folium(st.session_state["maps"]["1"], width=None, height=520, key="map_i1")
    show_flows_table(p1.get("flows_after", []), caption="Flows after Incident 1")

# Incident 2
if st.session_state["payloads"]["2"]:
    p2 = st.session_state["payloads"]["2"]
    st.divider()
    st.subheader("Incident 2 — Demand surge (+25%)")
    before, after = p2.get("objective_before"), p2.get("objective_after")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", fmt(before))
    c2.metric("Objective (after)", fmt(after), delta=safe_delta(after, before))
    c3.metric("Δ (after-before)", safe_delta(after, before))
    if p2.get("customer"): st.caption(f"Customer impacted: {p2['customer']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 2)**")
        st_folium(st.session_state["maps"]["2"], width=None, height=520, key="map_i2")
    show_flows_table(p2.get("flows_after", []), caption="Flows after Incident 2")

# Incident 3
if st.session_state["payloads"]["3"]:
    p3 = st.session_state["payloads"]["3"]
    st.divider()
    st.subheader("Incident 3 — Strategic express lane")
    before, after = p3.get("objective_before"), p3.get("objective_after")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", fmt(before))
    c2.metric("Objective (after)", fmt(after), delta=safe_delta(after, before))
    c3.metric("Δ (after-before)", safe_delta(after, before))
    if p3.get("lane"):
        l = p3["lane"]
        if "cost_per_unit" in l:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']} @ {fmt(l['cost_per_unit'])}")
        else:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 3)**")
        st_folium(st.session_state["maps"]["3"], width=None, height=520, key="map_i3")
    show_flows_table(p3.get("flows_after", []), caption="Flows after Incident 3")
