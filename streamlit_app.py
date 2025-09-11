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

# Gemini (optional; deterministic fallback if not configured)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ──────────────────────────────────────────────────────────────────────────────
# MUST be the first Streamlit command
# ──────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "TMS Agent — Control Tower (Gemini)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# App constants
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "base_case.xlsx")  # your Excel path in repo
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def ensure_state():
    ss = st.session_state
    ss.setdefault("case_loaded_from", None)  # "repo" or "upload"
    ss.setdefault("boot", None)
    ss.setdefault("phase", "idle")  # idle, baseline, incident1, incident2, incident3
    ss.setdefault("payloads", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("maps", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("explanations", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("gem_model", "gemini-1.5-pro")
    ss.setdefault("gem_temperature", 0.1)
    ss.setdefault("gem_max_tokens", 300)

ensure_state()

# ──────────────────────────────────────────────────────────────────────────────
# Excel parsing tailored to YOUR workbook
# Relevant sheets:
#   - Customers                (IDs; may or may not have lat/lon)
#   - Customer Product Data    (demand)
#   - Location                 (all site coordinates)
#   - Transport Cost           (lanes)
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s): return str(s).strip().lower()

def load_excel_from_repo_or_upload() -> dict:
    """Load /data/base_case.xlsx or allow user upload."""
    if os.path.exists(DATA_PATH):
        try:
            xls = pd.ExcelFile(DATA_PATH)
            return {"source": "repo", "xls": xls}
        except Exception as e:
            st.error(f"Failed to open {DATA_PATH}: {e}")

    up = st.file_uploader("Upload your base_case.xlsx", type=["xlsx"])
    if up is not None:
        try:
            xls = pd.ExcelFile(io.BytesIO(up.read()))
            return {"source": "upload", "xls": xls}
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    st.error(
        "Data file missing. Add `data/base_case.xlsx` to the repo "
        "or upload the Excel above."
    )
    st.stop()

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

def read_case() -> dict:
    bundle = load_excel_from_repo_or_upload()
    xls = bundle["xls"]
    st.session_state["case_loaded_from"] = bundle["source"]

    # Helper: robust sheet getter by fuzzy name
    def get_sheet(*candidates) -> Optional[pd.DataFrame]:
        name_map = { _norm(n): n for n in xls.sheet_names }
        for cand in candidates:
            key = _norm(cand)
            if key in name_map:
                df = pd.read_excel(xls, name_map[key])
                df.columns = [str(c).strip() for c in df.columns]
                return df
        # substring fallback
        for nm in xls.sheet_names:
            if any(_norm(cand) in _norm(nm) for cand in candidates):
                df = pd.read_excel(xls, nm)
                df.columns = [str(c).strip() for c in df.columns]
                return df
        return None

    customers_df = get_sheet("Customers")
    demand_df    = get_sheet("Customer Product Data", "CustomerProduct", "Cust Prod")
    loc_df       = get_sheet("Location", "Locations")
    lanes_df     = get_sheet("Transport Cost", "Transportation", "Transport")

    if customers_df is None or demand_df is None or loc_df is None or lanes_df is None:
        st.error("Could not find required sheets (Customers, Customer Product Data, Location, Transport Cost).")
        st.stop()

    # --- Identify key columns (robust, with fallbacks)
    # Customers: ID (required), lat/lon (optional)
    cust_id_col = _find_col(customers_df, ["customer","customerid","id","name"])
    # optional reference to Location table
    cust_loc_ref_col = _find_col(customers_df, ["location","site","node"])

    cust_lat_col = _find_col(customers_df, ["lat"], contains=True)  # accepts 'Latitude', 'lat', etc.
    cust_lon_col = _find_col(customers_df, ["lon","lng","long"], contains=True)

    if not cust_id_col:
        st.error("Customers sheet must contain an ID/Name column (e.g., 'Customer', 'CustomerID', 'Name').")
        st.stop()

    # Demand: Customer, Product, Demand
    dem_cust_col = _find_col(demand_df, ["customer","customerid","id","name"])
    dem_prod_col = _find_col(demand_df, ["product"])
    dem_qty_col  = _find_col(demand_df, ["demand","qty","quantity","volume"])
    if not all([dem_cust_col, dem_prod_col, dem_qty_col]):
        st.error("Customer Product Data sheet must contain Customer, Product, Demand columns.")
        st.stop()

    # Location: Location ID, lat, lon
    loc_id_col = _find_col(loc_df, ["location","id","name","site"])
    loc_lat_col = _find_col(loc_df, ["lat"], contains=True)
    loc_lon_col = _find_col(loc_df, ["lon","lng","long"], contains=True)
    if not all([loc_id_col, loc_lat_col, loc_lon_col]):
        st.error("Location sheet must contain Location ID/Name and Latitude/Longitude.")
        st.stop()

    # Lanes: Source, Destination, Product, Capacity, Cost
    lanes_src_col = _find_col(lanes_df, ["source","src","from"])
    lanes_dst_col = _find_col(lanes_df, ["destination","dest","dst","to"])
    lanes_pr_col  = _find_col(lanes_df, ["product"])
    lanes_cap_col = _find_col(lanes_df, ["capacity","cap"])
    lanes_cost_col= _find_col(lanes_df, ["cost","unit cost","cost_per_unit"], contains=True)
    if not all([lanes_src_col, lanes_dst_col, lanes_pr_col, lanes_cap_col, lanes_cost_col]):
        st.error("Transport Cost sheet must contain Source, Destination, Product, Capacity, and Cost columns.")
        st.stop()

    # --- Build nodes
    customers_df = customers_df.dropna(subset=[cust_id_col]).copy()
    loc_df = loc_df.dropna(subset=[loc_id_col]).copy()

    # Start with customers minimal frame
    cust_nodes = customers_df[[cust_id_col]].copy()
    cust_nodes.columns = ["id"]
    cust_nodes["kind"] = "customer"

    # Merge coordinates into customers:
    # 1) If Customers already has lat/lon, use them
    if cust_lat_col and cust_lon_col:
        cust_nodes = cust_nodes.merge(
            customers_df[[cust_id_col, cust_lat_col, cust_lon_col]].rename(
                columns={cust_id_col: "id", cust_lat_col: "lat", cust_lon_col: "lon"}
            ),
            on="id", how="left"
        )
    else:
        # 2) Try exact ID match against Location
        coords_from_loc = loc_df[[loc_id_col, loc_lat_col, loc_lon_col]].rename(
            columns={loc_id_col:"id", loc_lat_col:"lat", loc_lon_col:"lon"}
        )
        cust_nodes = cust_nodes.merge(coords_from_loc, on="id", how="left")

        # 3) If still missing, try Customers' Location/Site reference → Location table
        if cust_loc_ref_col:
            ref_map = customers_df[[cust_id_col, cust_loc_ref_col]].rename(
                columns={cust_id_col:"id", cust_loc_ref_col:"loc_ref"}
            )
            ref_coords = coords_from_loc.rename(columns={"id":"loc_ref"})
            cust_nodes = cust_nodes.merge(ref_map, on="id", how="left")
            cust_nodes = cust_nodes.merge(ref_coords, on="loc_ref", how="left", suffixes=("","_from_ref"))
            # fill missing lat/lon from ref
            cust_nodes["lat"] = cust_nodes["lat"].fillna(cust_nodes.pop("lat_from_ref"))
            cust_nodes["lon"] = cust_nodes["lon"].fillna(cust_nodes.pop("lon_from_ref"))
            cust_nodes.drop(columns=["loc_ref"], inplace=True)

    # Non-customer nodes (suppliers/DCs) directly from Location
    loc_nodes = loc_df[[loc_id_col, loc_lat_col, loc_lon_col]].copy()
    loc_nodes.columns = ["id","lat","lon"]

    def infer_kind(node_id: str) -> str:
        nid = str(node_id).lower()
        if "sup" in nid or "supplier" in nid:
            return "supplier"
        if "wh" in nid or "dc" in nid or "ware" in nid:
            return "dc"
        return "dc"  # safe default
    loc_nodes["kind"] = loc_nodes["id"].apply(infer_kind)

    # Combine & dedupe
    nodes_df = pd.concat([cust_nodes, loc_nodes], ignore_index=True).drop_duplicates(subset=["id"])

    # --- Demand
    demand = demand_df[[dem_cust_col, dem_prod_col, dem_qty_col]].copy()
    demand.columns = ["customer","product","demand"]
    demand["demand"] = pd.to_numeric(demand["demand"], errors="coerce").fillna(0.0)

    # --- Products
    products = sorted(set(demand["product"].unique()) | set(lanes_df[lanes_pr_col].unique()))

    # --- Lanes
    lanes = lanes_df[[lanes_src_col, lanes_dst_col, lanes_pr_col, lanes_cap_col, lanes_cost_col]].copy()
    lanes.columns = ["src","dst","product","capacity","cost_per_unit"]
    lanes["capacity"] = pd.to_numeric(lanes["capacity"], errors="coerce").fillna(0.0)
    lanes["cost_per_unit"] = pd.to_numeric(lanes["cost_per_unit"], errors="coerce").fillna(0.0)

    # --- Synthetic supply, balanced by total product demand across available sources
    total_demand_per_p = demand.groupby("product")["demand"].sum().to_dict()

    # identify customer ids to avoid treating them as sources
    customer_ids = set(customers_df[cust_id_col].astype(str))

    sources_by_p = {}
    for p in products:
        mask = (lanes["product"] == p)
        srcs = sorted(set(lanes.loc[mask, "src"].astype(str)))
        srcs = [s for s in srcs if s not in customer_ids]
        if not srcs:
            # fallback: any non-customer location IDs
            srcs = list(loc_nodes["id"].astype(str))
        sources_by_p[p] = srcs

    supply_rows = []
    for p in products:
        tot = float(total_demand_per_p.get(p, 0.0))
        srcs = sources_by_p[p]
        share = (tot / max(1, len(srcs))) if tot > 0 else 0.0
        for s in srcs:
            supply_rows.append({"supplier": s, "product": p, "supply": share})
    supply = pd.DataFrame(supply_rows)

    # Friendly note if some customers still lack coords (solver ok; map will skip)
    missing_coords = nodes_df.query("kind=='customer' and (lat.isna() or lon.isna())")["id"].tolist()
    if missing_coords:
        st.warning(f"{len(missing_coords)} customer(s) have no coordinates in Customers/Location. They’ll be solved but hidden on the map.")

    return {
        "nodes": nodes_df,
        "demand": demand,
        "supply": supply,
        "lanes": lanes,
        "products": products
    }

# ──────────────────────────────────────────────────────────────────────────────
# Optimizer (LP with slacks)
# ──────────────────────────────────────────────────────────────────────────────
def solve_min_cost_flow(case: dict, shock: dict | None = None) -> dict:
    """
    Balanced min-cost multi-commodity flow with shortage/disposal slacks (BIGM).
    shock:
      - {"type":"lane_cap","src":...,"dst":...,"product":<opt>,"new_capacity":...}
      - {"type":"demand_spike","customer":...,"pct":0.25}
      - {"type":"express_lane","src":...,"dst":...,"product":...,"capacity":...,"cost_per_unit":...}
    """
    nodes = case["nodes"].copy()
    demand_df = case["demand"].copy()
    supply_df = case["supply"].copy()
    lanes_df  = case["lanes"].copy()

    products = sorted(set(demand_df["product"].unique()) | set(supply_df["product"].unique()))

    # Apply shock
    if shock:
        t = shock.get("type")
        if t == "lane_cap":
            cond = (lanes_df["src"] == shock["src"]) & (lanes_df["dst"] == shock["dst"])
            if "product" in shock and pd.notna(shock["product"]):
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

    # Variables x[(src,dst,product)]
    lanes = lanes_df[["src","dst","product","capacity","cost_per_unit"]].to_dict(orient="records")
    var_keys = [(l["src"], l["dst"], l["product"]) for l in lanes]
    vidx = {k:i for i,k in enumerate(var_keys)}
    n_x = len(var_keys)

    # Slack vars: disposal per supplier/product, shortage per customer/product
    cust_prod = sorted({(r["customer"], r["product"]) for _, r in demand_df.iterrows()})
    supp_prod = sorted({(r["supplier"], r["product"]) for _, r in supply_df.iterrows()})
    sp_index = {k: n_x + i for i, k in enumerate(supp_prod)}  # disposal
    sh_index = {k: n_x + len(supp_prod) + i for i, k in enumerate(cust_prod)}  # shortage
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

    # Supply balance: outflow + disposal = supply
    for s,p in supp_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if src == s and pr == p:
                row[i] = 1.0
        row[sp_index[(s,p)]] = 1.0
        sup_val = float(supply_df[(supply_df["supplier"]==s) & (supply_df["product"]==p)]["supply"].sum())
        A_eq.append(row); b_eq.append(sup_val)

    # Demand balance: inflow + shortage = demand
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

    used_slacks = False
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
        used_slacks = (total_disposal > 1e-6) or (total_shortage > 1e-6)
        objective = float(res.fun)
    else:
        used_slacks = True  # show as fallback used

    # Node list for map
    nrecs = []
    for _, r in case["nodes"].iterrows():
        lt = r.get("lat")
        ln = r.get("lon")
        nrecs.append({
            "id": str(r.get("id")),
            "kind": str(r.get("kind")),
            "lat": float(lt) if pd.notna(lt) else None,
            "lon": float(ln) if pd.notna(ln) else None
        })

    products_list = case.get("products", sorted(set([f["product"] for f in flows])))

    return {
        "ok": bool(res.success),
        "objective_cost": objective,
        "used_slacks": used_slacks,
        "total_shortage": total_shortage,
        "total_disposal": total_disposal,
        "nodes": nrecs,
        "products": products_list,
        "flows": flows,
        "currency": "INR",
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
    if not prev.get("flows"):
        return {"title":"Incident 1","objective_before":prev.get("objective_cost"),"objective_after":prev.get("objective_cost"),"flows_after":prev.get("flows",[])}
    biggest = sorted(prev["flows"], key=lambda x: x["flow"], reverse=True)[0]
    shock = {"type": "lane_cap", "src": biggest["src"], "dst": biggest["dst"], "product": biggest["product"], "new_capacity": max(0.5*biggest["flow"], 1.0)}
    before = prev["objective_cost"]
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 1 — Corridor capacity restriction",
        "lane": {"src": shock["src"], "dst": shock["dst"], "product": biggest["product"]},
        "objective_before": before,
        "objective_after": aft["objective_cost"],
        "used_slacks": aft["used_slacks"],
        "total_shortage": aft["total_shortage"],
        "total_disposal": aft["total_disposal"],
        "flows_after": aft["flows"],
        "currency": aft["currency"], "flow_unit": aft["flow_unit"]
    }

def incident_2(case: dict, prev: dict) -> dict:
    if not prev.get("flows"):
        return {"title":"Incident 2","objective_before":prev.get("objective_cost"),"objective_after":prev.get("objective_cost"),"flows_after":prev.get("flows",[])}
    dest_tot = {}
    for f in prev["flows"]:
        dest_tot[f["dst"]] = dest_tot.get(f["dst"], 0.0) + f["flow"]
    customer = sorted(dest_tot.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    shock = {"type": "demand_spike", "customer": customer, "pct": 0.25}
    before = prev["objective_cost"]
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 2 — Demand surge (+25%)",
        "customer": customer,
        "objective_before": before,
        "objective_after": aft["objective_cost"],
        "used_slacks": aft["used_slacks"],
        "total_shortage": aft["total_shortage"],
        "total_disposal": aft["total_disposal"],
        "flows_after": aft["flows"],
        "currency": aft["currency"], "flow_unit": aft["flow_unit"]
    }

def incident_3(case: dict, prev: dict) -> dict:
    if not prev.get("flows"):
        return {"title":"Incident 3","objective_before":prev.get("objective_cost"),"objective_after":prev.get("objective_cost"),"flows_after":prev.get("flows",[])}
    top = sorted(prev["flows"], key=lambda x: x["flow"], reverse=True)[0]
    src, dst, pr = top["src"], top["dst"], top["product"]
    shock = {"type": "express_lane", "src": src, "dst": dst, "product": pr,
             "capacity": max(top["flow"]*0.5, 1.0), "cost_per_unit": 0.75}
    before = prev["objective_cost"]
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 3 — Strategic express lane",
        "lane": {"src": src, "dst": dst, "cost_per_unit": shock["cost_per_unit"], "capacity": shock["capacity"]},
        "objective_before": before,
        "objective_after": aft["objective_cost"],
        "used_slacks": aft["used_slacks"],
        "total_shortage": aft["total_shortage"],
        "total_disposal": aft["total_disposal"],
        "flows_after": aft["flows"],
        "currency": aft["currency"], "flow_unit": aft["flow_unit"]
    }

# ──────────────────────────────────────────────────────────────────────────────
# Maps & Tables
# ──────────────────────────────────────────────────────────────────────────────
def _center_latlon(nodes: List[Dict]) -> Tuple[float,float]:
    lats = [n.get("lat") for n in nodes if n.get("lat") is not None]
    lons = [n.get("lon") for n in nodes if n.get("lon") is not None]
    if not lats or not lons: return (20.5937, 78.9629)
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def _product_color_map(products: List[str]) -> Dict[str,str]:
    return {p: PALETTE[i % len(PALETTE)] for i,p in enumerate(sorted(products))}

def _node_by_id(nodes: List[Dict]) -> Dict[str,Dict]:
    return {n["id"]: n for n in nodes}

def _flow_weight_scaler(flows: List[Dict], min_w: float = 0.6, max_w: float = 3.0):
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
        if n.get("lat") is None or n.get("lon") is None:  # skip nodes without coords
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
# Gemini explanation (guardrailed) with deterministic fallback
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
        bits.append(f" Objective moved {before:,.2f} → {after:,.2f} (Δ {after-before:+,.2f}).")
    lane = payload.get("lane")
    if lane:
        parts=[]
        if "src" in lane and "dst" in lane: parts.append(f"{lane['src']}→{lane['dst']}")
        if "cost_per_unit" in lane: parts.append(f"cpu={lane['cost_per_unit']}")
        if "capacity" in lane: parts.append(f"cap={lane['capacity']}")
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
            "units": {"currency": payload.get("currency","INR"), "flow_unit": payload.get("flow_unit","units")}
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
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🚀 Bootstrap baseline (solve from Excel)"):
        base, case = build_baseline()
        st.session_state["case"] = case
        st.session_state["boot"] = base
        st.session_state["payloads"]["0"] = {
            "title": "Baseline",
            "objective_before": base["objective_cost"],
            "objective_after": base["objective_cost"],
            "flows_after": base["flows"],
            "currency": base["currency"], "flow_unit": base["flow_unit"]
        }
        st.session_state["phase"] = "baseline"
        st.session_state["maps"]["0"] = build_map(base["nodes"], base["flows"], base["products"])
        st.success(f"Baseline solved (data source: {st.session_state['case_loaded_from']}).")

with col2:
    if st.button("🚨 Run Incident 1 now"):
        if not st.session_state.get("boot"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            resp1 = incident_1(case, st.session_state["payloads"]["0"])
            st.session_state["payloads"]["1"] = resp1
            st.session_state["phase"] = "incident1"
            st.session_state["maps"]["1"] = build_map(st.session_state["boot"]["nodes"], resp1.get("flows_after",[]), st.session_state["boot"]["products"])
            ensure_explanation(1, resp1)
            st.success("Incident 1 executed.")

# Sidebar incident triggers
with st.sidebar:
    st.divider()
    st.caption("Run additional incidents")
    if st.button("▶️ Incident 2 (demand surge)"):
        if not st.session_state.get("boot"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            base_or_last = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
            resp2 = incident_2(case, base_or_last)
            st.session_state["payloads"]["2"] = resp2
            st.session_state["phase"] = "incident2"
            st.session_state["maps"]["2"] = build_map(st.session_state["boot"]["nodes"], resp2.get("flows_after",[]), st.session_state["boot"]["products"])
            ensure_explanation(2, resp2)
            st.success("Incident 2 executed.")
    if st.button("▶️ Incident 3 (strategic express lane)"):
        if not st.session_state.get("boot"):
            st.warning("Run baseline first.")
        else:
            case = st.session_state["case"]
            base_or_last = st.session_state["payloads"]["2"] or st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
            resp3 = incident_3(case, base_or_last if base_or_last else st.session_state["payloads"]["0"])
            st.session_state["payloads"]["3"] = resp3
            st.session_state["phase"] = "incident3"
            st.session_state["maps"]["3"] = build_map(st.session_state["boot"]["nodes"], resp3.get("flows_after",[]), st.session_state["boot"]["products"])
            ensure_explanation(3, resp3)
            st.success("Incident 3 executed.")

# Agent narrative
with agent:
    ph = st.session_state["phase"]
    if ph == "baseline" and st.session_state["payloads"]["0"]:
        p = st.session_state["payloads"]["0"]
        st.write(f"**Agent**: Baseline active. Objective **{p['objective_after']:,.2f}**. Monitoring lanes & capacity headroom.")
    elif ph == "incident1" and st.session_state["payloads"]["1"]:
        p = st.session_state["payloads"]["1"]
        st.write(f"**Agent**: {p['title']} | Objective delta **{p['objective_after']-p['objective_before']:+,.2f}**.")
        st.markdown(st.session_state["explanations"]["1"] or "")
    elif ph == "incident2" and st.session_state["payloads"]["2"]:
        p = st.session_state["payloads"]["2"]
        st.write(f"**Agent**: {p['title']} | Objective delta **{p['objective_after']-p['objective_before']:+,.2f}**.")
        st.markdown(st.session_state["explanations"]["2"] or "")
    elif ph == "incident3" and st.session_state["payloads"]["3"]:
        p = st.session_state["payloads"]["3"]
        st.write(f"**Agent**: {p['title']} | Objective delta **{p['objective_after']-p['objective_before']:+,.2f}**.")
        st.markdown(st.session_state["explanations"]["3"] or "")
    else:
        st.caption("Ready. Solve baseline, then run incidents.")

# Baseline section
if st.session_state.get("boot"):
    base = st.session_state["boot"]
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
    before, after = float(p1.get("objective_before", 0.0)), float(p1.get("objective_after", 0.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", f"{before:,.2f}")
    c2.metric("Objective (after)", f"{after:,.2f}", delta=f"{after-before:+,.2f}")
    c3.metric("Δ (after-before)", f"{after-before:+,.2f}")
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
    before, after = float(p2.get("objective_before", 0.0)), float(p2.get("objective_after", 0.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", f"{before:,.2f}")
    c2.metric("Objective (after)", f"{after:,.2f}", delta=f"{after-before:+,.2f}")
    c3.metric("Δ (after-before)", f"{after-before:+,.2f}")
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
    before, after = float(p3.get("objective_before", 0.0)), float(p3.get("objective_after", 0.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", f"{before:,.2f}")
    c2.metric("Objective (after)", f"{after:,.2f}", delta=f"{after-before:+,.2f}")
    c3.metric("Δ (after-before)", f"{after-before:+,.2f}")
    if p3.get("lane"):
        l = p3["lane"]
        if "cost_per_unit" in l:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']} @ {l['cost_per_unit']}")
        else:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 3)**")
        st_folium(st.session_state["maps"]["3"], width=None, height=520, key="map_i3")
    show_flows_table(p3.get("flows_after", []), caption="Flows after Incident 3")
