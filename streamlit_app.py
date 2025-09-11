import os
import io
import json
import math
import time
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
# App constants & helpers
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "base_case.xlsx")
TICK_INTERVAL = 3.0          # seconds between narration ticks
THINK_TICKS = 3              # number of narration steps before acting
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

def pct(x) -> Optional[float]:
    """Safely convert a fraction (e.g., 0.975) to percentage (97.5)."""
    try:
        if x is None: return None
        v = float(x)
        if np.isnan(v): return None
        return 100.0 * v
    except Exception:
        return None

def safe_delta(after, before, none="—", places=2):
    if after is None or before is None: return none
    try:
        a, b = float(after), float(before)
        if any(map(np.isnan, [a,b])): return none
        return f"{(a-b):+,.{places}f}"
    except Exception:
        return none

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
# Session state
# ──────────────────────────────────────────────────────────────────────────────
def ensure_state():
    ss = st.session_state
    # Data & solves
    ss.setdefault("case_loaded_from", None)
    ss.setdefault("case", None)                  # parsed inputs
    ss.setdefault("total_demand", None)          # sum of all customer demands
    ss.setdefault("base", None)                  # baseline solve payload
    ss.setdefault("maps", {})                    # cached folium maps
    # Incidents & agent state machine
    ss.setdefault("phase", "idle")               # idle|baseline_ready|thinking|acting|evaluating|complete
    ss.setdefault("is_running", False)
    ss.setdefault("current_incident", None)      # {"id": 1|2|3, "label": "..."}
    ss.setdefault("tick_count", 0)
    ss.setdefault("next_tick_at", None)          # epoch seconds
    ss.setdefault("logs", {"1": [], "2": [], "3": []})
    # Snapshots for compare
    ss.setdefault("payloads", {"0": None, "1": None, "2": None, "3": None})
    # LLM config
    ss.setdefault("gem_model", "gemini-1.5-pro")
    ss.setdefault("gem_temperature", 0.1)
    ss.setdefault("gem_max_tokens", 300)
    # Explanations cache
    ss.setdefault("explanations", {"0": None, "1": None, "2": None, "3": None})
ensure_state()

def log(iid: int, msg: str):
    st.session_state["logs"].setdefault(str(iid), []).append({"t": time.strftime("%H:%M:%S"), "msg": msg})

def schedule_next_tick():
    st.session_state["next_tick_at"] = time.time() + TICK_INTERVAL

# ──────────────────────────────────────────────────────────────────────────────
# Load & parse data (AIMMS-like workbook schema)
# ──────────────────────────────────────────────────────────────────────────────
def load_excel() -> pd.ExcelFile:
    if os.path.exists(DATA_PATH):
        return pd.ExcelFile(DATA_PATH)
    up = st.file_uploader("Upload data file", type=["xlsx"])
    if up is None:
        st.error("Data file missing. Add `data/base_case.xlsx` to the app or upload it here.")
        st.stop()
    return pd.ExcelFile(io.BytesIO(up.read()))

def read_case() -> dict:
    xls = load_excel()
    st.session_state["case_loaded_from"] = "repo" if os.path.exists(DATA_PATH) else "upload"

    customers_df = pd.read_excel(xls, "Customers")
    demand_df    = pd.read_excel(xls, "Customer Product Data")
    loc_df       = pd.read_excel(xls, "Location")
    lanes_tpl    = pd.read_excel(xls, "Transport Cost")
    groups_df    = pd.read_excel(xls, "Location Groups")
    supp_prod    = pd.read_excel(xls, "Supplier Product")
    wh_df        = pd.read_excel(xls, "Warehouse")

    for df in (customers_df, demand_df, loc_df, lanes_tpl, groups_df, supp_prod, wh_df):
        df.columns = [str(c).strip() for c in df.columns]

    # Customers
    cust_id_col = _find_col(customers_df, ["customer","name","id"])
    if not cust_id_col:
        st.error("Customers sheet must contain a Customer/Name/ID column."); st.stop()
    customers = customers_df[cust_id_col].dropna().astype(str).tolist()

    # Demand
    dem_cust_col = _find_col(demand_df, ["customer","name","id"])
    dem_prod_col = _find_col(demand_df, ["product"])
    dem_qty_col  = _find_col(demand_df, ["demand","qty","quantity","volume"])
    if not all([dem_cust_col, dem_prod_col, dem_qty_col]):
        st.error("Customer Product Data must contain Customer, Product, Demand."); st.stop()
    dem = demand_df[[dem_cust_col, dem_prod_col, dem_qty_col]].copy()
    dem.columns = ["customer","product","demand"]
    dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)
    demand = dem.groupby(["customer","product"])["demand"].sum().reset_index()
    st.session_state["total_demand"] = float(demand["demand"].sum())

    # Locations
    loc_id_col = _find_col(loc_df, ["location","name","id"])
    lat_col = _find_col(loc_df, ["lat"], contains=True)
    lon_col = _find_col(loc_df, ["lon","lng","long"], contains=True)
    if not all([loc_id_col, lat_col, lon_col]):
        st.error("Location sheet must contain Location, Latitude, Longitude."); st.stop()
    loc = loc_df[[loc_id_col, lat_col, lon_col]].dropna(subset=[loc_id_col]).copy()
    loc.columns = ["id","lat","lon"]
    coord = {r["id"]: {"lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in loc.iterrows()}

    # Location groups
    if "Location" not in groups_df.columns or "SubLocation" not in groups_df.columns:
        st.error("Location Groups sheet must contain columns Location and SubLocation."); st.stop()
    FG = set(groups_df[groups_df["Location"]=="FG"]["SubLocation"].dropna().astype(str))
    DC = set(groups_df[groups_df["Location"]=="DC"]["SubLocation"].dropna().astype(str))

    # Supply
    sp = supp_prod.rename(columns={"Location":"supplier"}).copy()
    if not {"supplier","Product"}.issubset(sp.columns):
        st.error("Supplier Product must include Location (supplier) and Product."); st.stop()
    cap_col = _find_col(sp, ["Maximum Capacity","Capacity","Cap"])
    if cap_col is None:
        sp["Maximum Capacity"] = 0.0; cap_col = "Maximum Capacity"
    sp["supply"] = pd.to_numeric(sp[cap_col], errors="coerce").fillna(0.0)
    if "Available" in sp.columns:
        sp["avail"] = pd.to_numeric(sp["Available"], errors="coerce").fillna(1.0)
        sp = sp[sp["avail"] > 0]
    sp = sp[["supplier","Product","supply"]].rename(columns={"Product":"product"})
    sp["supplier"] = sp["supplier"].astype(str)
    sp = sp[sp["supplier"].isin(FG)]
    supply = sp.groupby(["supplier","product"])["supply"].sum().reset_index()

    # Products
    products = sorted(demand["product"].dropna().astype(str).unique().tolist()

    # Transport cost templates
    def _tpl_row(fr, to):
        if "From Location" in lanes_tpl.columns and "To Location" in lanes_tpl.columns:
            hit = lanes_tpl[(lanes_tpl.get("From Location")==fr) & (lanes_tpl.get("To Location")==to)]
        else:
            hit = pd.DataFrame()
        if len(hit) > 0:
            r = hit.iloc[0].to_dict()
            return {"cpd": float(r.get("Cost per Distance", 1.0) or 1.0),
                    "cpu": float(r.get("Cost Per UOM", 0.0) or 0.0)}
        return {"cpd": 1.0 if fr=="FG" else 2.0, "cpu": 5.0 if fr=="FG" else 10.0}

    tpl_fg_dc = _tpl_row("FG","DC")
    tpl_dc_ct = _tpl_row("DC","City")

    # Nodes
    nodes = []
    warehouses = set(wh_df.get("Location", pd.Series(dtype=str)).dropna().astype(str)) | DC
    for s in FG:
        coords = coord.get(s); nodes.append({"id": s, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "supplier"})
    for d in warehouses:
        coords = coord.get(d); nodes.append({"id": d, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "dc"})
    for c in customers:
        coords = coord.get(c); nodes.append({"id": c, "lat": coords["lat"] if coords else None, "lon": coords["lon"] if coords else None, "kind": "customer"})
    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["id"])

    # Lanes
    def d_latlon(a, b):
        if a not in coord or b not in coord: return None
        return haversine_km(coord[a]["lat"], coord[a]["lon"], coord[b]["lat"], coord[b]["lon"])

    lanes = []
    # FG -> DC
    for s in FG:
        if s not in coord: continue
        for d in warehouses:
            if d not in coord: continue
            dist = d_latlon(s, d)
            if dist is None: continue
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
            dist = d_latlon(d, c)
            if dist is None: continue
            for p in products:
                dem_p = float(demand[(demand["customer"]==c) & (demand["product"]==p)]["demand"].sum())
                cap = (dem_p * 2.0) + 1e6  # generous cap
                cost = tpl_dc_ct["cpu"] + tpl_dc_ct["cpd"] * dist
                lanes.append({"src": d, "dst": c, "product": p, "capacity": cap, "cost_per_unit": cost})

    return {
        "nodes": nodes_df,
        "demand": demand,
        "supply": supply,
        "lanes": pd.DataFrame(lanes),
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
            if src == s and pr == p: row[i] = 1.0
        row[sp_index[(s,p)]] = 1.0
        sup_val = float(supply_df[(supply_df["supplier"]==s) & (supply_df["product"]==p)]["supply"].sum())
        A_eq.append(row); b_eq.append(sup_val)

    # Demand balance: sum inflow + shortage = demand
    for cst,p in cust_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if dst == cst and pr == p: row[i] = 1.0
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
# KPI computation & narrative helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_kpis(case: dict, payload: dict) -> dict:
    """Compute CXO-friendly KPIs from payload + case."""
    cost = payload.get("objective_after")
    flows = payload.get("flows_after") or payload.get("flows") or []
    total_flow = float(sum(num(f.get("flow")) for f in flows))
    lanes_used = len({(f["src"], f["dst"], f["product"]) for f in flows})
    avg_cpu = (cost / total_flow) if (cost is not None and total_flow > 0) else None
    shortage = payload.get("total_shortage")
    disposal = payload.get("total_disposal")
    total_demand = st.session_state.get("total_demand")
    fill_rate = None
    if total_demand is not None and shortage is not None:
        denom = max(1e-9, float(total_demand))
        fill_rate = max(0.0, min(1.0, 1.0 - float(shortage)/denom))
    return {
        "total_cost": cost,
        "total_flow_units": total_flow,
        "lanes_used": lanes_used,
        "avg_cost_per_unit": avg_cpu,
        "shortage_units": shortage,
        "fill_rate": fill_rate,
        "disposal_units": disposal,
        "currency": payload.get("currency", "EUR"),
        "flow_unit": payload.get("flow_unit", "units")
    }

def kpi_explanation_md(k: dict, label: str) -> str:
    """Plain-English explanations for CXO audience."""
    lines = [
        f"**{label} — KPI Summary**",
        f"- **Total transport cost**: {fmt(k.get('total_cost'))} {k.get('currency','')}.",
        f"- **Total shipped volume**: {fmt(k.get('total_flow_units'))} {k.get('flow_unit','')}.",
        f"- **Lanes used**: {fmt(k.get('lanes_used'), places=0)} lanes with non-zero flow.",
        f"- **Avg. cost per unit**: {fmt(k.get('avg_cost_per_unit'))} {k.get('currency','')}/{k.get('flow_unit','')}.",
        f"- **Shortage volume**: {fmt(k.get('shortage_units'))} {k.get('flow_unit','')} not fulfilled.",
        f"- **Service fill rate**: {fmt(pct(k.get('fill_rate')), places=1)}%.",
        f"- **Disposals (waste)**: {fmt(k.get('disposal_units'))} {k.get('flow_unit','')} scrapped."
    ]
    lines += [
        "",
        "**KPI definitions**",
        "- *Total transport cost*: Optimizer’s objective (sum of lane costs).",
        "- *Total shipped volume*: Sum of all product flows across the network.",
        "- *Lanes used*: Count of distinct corridors that carried any volume.",
        "- *Avg. cost per unit*: Total cost divided by shipped volume.",
        "- *Shortage volume*: Demand left unserved (model allows this at heavy penalty).",
        "- *Service fill rate*: Served demand / Total demand.",
        "- *Disposals*: Unused supply disposed (also heavily penalized).",
    ]
    return "\n".join(lines)

def headline_one_liner(title: str, k_now: dict, k_prev: Optional[dict]) -> str:
    """
    A single executive-friendly sentence: What happened & so what?
    Uses cost and fill-rate changes; mentions likely driver from title.
    """
    cost = k_now.get("total_cost")
    fill = k_now.get("fill_rate")
    cost_txt = f"total transport cost **{fmt(cost)}**" if cost is not None else "transport cost (n/a)"
    fill_txt = f"service fill rate **{fmt(pct(fill), places=1)}%**" if fill is not None else "service (n/a)"

    delta_cost = None
    delta_fill = None
    if k_prev:
        if k_prev.get("total_cost") is not None and cost is not None:
            delta_cost = cost - k_prev["total_cost"]
        if k_prev.get("fill_rate") is not None and fill is not None:
            delta_fill = (fill - k_prev["fill_rate"]) * 100.0

    driver = title.split("—")[-1].strip() if "—" in title else title
    parts = [f"**{title}** complete:"]
    if delta_cost is not None:
        parts.append(f"{'↑' if delta_cost>0 else '↓'} cost {fmt(abs(delta_cost))} vs reference;")
    parts.append(cost_txt + ",")
    if delta_fill is not None:
        parts.append(f"{'↑' if delta_fill>0 else '↓'} fill {fmt(abs(delta_fill), places=1)} pp;")
    parts.append(fill_txt + ".")
    parts.append(f" Driver: *{driver}*.")
    return " ".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Incidents (deterministic action planners)
# ──────────────────────────────────────────────────────────────────────────────
def pick_top_lane(payload: dict) -> Optional[dict]:
    flows = payload.get("flows_after") or payload.get("flows") or []
    if not flows: return None
    return sorted(flows, key=lambda x: x["flow"], reverse=True)[0]

def plan_incident_action(iid: int, case: dict, prev: dict) -> dict:
    """
    Returns a dict 'action' to feed into the solver as 'shock', plus a short reasoning string.
    If GOOGLE_API_KEY is provided, we ask Gemini to pick; else deterministic choice.
    """
    top = pick_top_lane(prev) or {}
    candidates = []
    if top:
        candidates.append({
            "type": "lane_cap",
            "label": "Temporarily restrict a congested corridor",
            "shock": {"type": "lane_cap", "src": top["src"], "dst": top["dst"], "product": top["product"],
                      "new_capacity": max(0.5 * float(top["flow"]), 1.0)}
        })
        candidates.append({
            "type": "express_lane",
            "label": "Add a cheaper express lane on the top corridor",
            "shock": {"type": "express_lane", "src": top["src"], "dst": top["dst"], "product": top["product"],
                      "capacity": max(0.5 * float(top["flow"]), 1.0), "cost_per_unit": 0.5}
        })
    dest_tot = {}
    base_flows = prev.get("flows_after") or prev.get("flows") or []
    for f in base_flows:
        dest_tot[f["dst"]] = dest_tot.get(f["dst"], 0.0) + f["flow"]
    if dest_tot:
        hot_cust = sorted(dest_tot.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        candidates.append({
            "type": "demand_spike",
            "label": "Simulate a +25% demand surge at the hottest customer",
            "shock": {"type": "demand_spike", "customer": hot_cust, "pct": 0.25}
        })

    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                st.session_state.get("gem_model","gemini-1.5-pro"),
                system_instruction=(
                    "Choose ONE action from the list based on the incident id.\n"
                    "Incident 1 prefers 'lane_cap'; Incident 2 prefers 'demand_spike'; Incident 3 prefers 'express_lane'.\n"
                    "Output JSON with keys: type, index (0-based in the candidate list).\n"
                )
            )
            pref = {1:"lane_cap", 2:"demand_spike", 3:"express_lane"}.get(iid, "lane_cap")
            prompt = {"incident_id": iid, "preferred": pref, "candidates": candidates}
            resp = model.generate_content(json.dumps(prompt), generation_config={"temperature": float(st.session_state.get("gem_temperature",0.1)), "max_output_tokens": 128})
            txt = (resp.text or "").strip()
            j = None
            try:
                j = json.loads(txt)
            except Exception:
                j = None
            if j and isinstance(j.get("index"), int) and 0 <= j["index"] < len(candidates):
                choice = candidates[j["index"]]
            else:
                order = {"lane_cap":0, "demand_spike":1, "express_lane":2}
                ranked = sorted(candidates, key=lambda c: abs(order.get(c["type"], 10) - order.get(pref,0)))
                choice = ranked[0] if ranked else (candidates[0] if candidates else {})
            reason = f"Agent selected '{choice.get('label','action')}' for Incident {iid}."
            return {"action": choice.get("shock", {}), "reason": reason}
        except Exception:
            pass

    if iid == 1 and any(c["type"]=="lane_cap" for c in candidates):
        choice = next(c for c in candidates if c["type"]=="lane_cap")
    elif iid == 2 and any(c["type"]=="demand_spike" for c in candidates):
        choice = next(c for c in candidates if c["type"]=="demand_spike")
    elif iid == 3 and any(c["type"]=="express_lane" for c in candidates):
        choice = next(c for c in candidates if c["type"]=="express_lane")
    else:
        choice = candidates[0] if candidates else {"shock": {},"label":"No-op"}

    return {"action": choice.get("shock", {}), "reason": f"Selected deterministic: {choice.get('label','action')}"}

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

def _flow_weight_scaler(flows: List[Dict], min_w: float = 0.6, max_w: float = 2.0):
    vals = sorted(float(f.get("flow", 0.0)) for f in flows if f.get("flow", 0.0) > 0)
    if not vals: return lambda x: min_w
    p90 = vals[min(int(0.9*(len(vals)-1)), len(vals)-1)]
    p90 = p90 if p90 > 0 else (max(vals) if vals else 1.0)
    def to_w(q):
        if q <= 0: return 0
        w = min_w + (max_w - min_w) * (q/(p90*1.0))
        return max(min_w, min(max_w, w))
    return to_w

def build_map(nodes: List[Dict], flows: List[Dict], products: List[str], key_suffix: str) -> folium.Map:
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
# LLM Narratives (Gemini) & deterministic fallback
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
    if st.session_state["explanations"].get(key):
        return
    st.session_state["explanations"][key] = gemini_explain(idx, payload)

# ──────────────────────────────────────────────────────────────────────────────
# Baseline & incident solvers that attach UI-ready payloads
# ──────────────────────────────────────────────────────────────────────────────
def build_baseline() -> tuple[dict, dict]:
    case = read_case()
    base = solve_min_cost_flow(case, None)
    base["title"] = "Baseline"
    return base, case

def wrap_payload_as_snapshot(payload_after: dict, title: str, before_cost: Optional[float]) -> dict:
    return {
        "title": title,
        "objective_before": before_cost,
        "objective_after": payload_after.get("objective_cost"),
        "used_slacks": payload_after.get("used_slacks"),
        "total_shortage": payload_after.get("total_shortage"),
        "total_disposal": payload_after.get("total_disposal"),
        "flows_after": payload_after.get("flows", []),
        "currency": payload_after.get("currency"),
        "flow_unit": payload_after.get("flow_unit"),
        "lane": None,
        "customer": None
    }

# ──────────────────────────────────────────────────────────────────────────────
# AGENT STATE MACHINE
# ──────────────────────────────────────────────────────────────────────────────
def start_incident(iid: int):
    if st.session_state["is_running"]:
        st.warning("Another scenario is running. Please wait for it to complete.")
        return
    if not st.session_state.get("base"):
        st.warning("Initialize baseline first."); return

    labels = {1: "Incident 1 — Corridor capacity restriction",
              2: "Incident 2 — Demand surge (+25%)",
              3: "Incident 3 — Strategic express lane"}
    st.session_state["current_incident"] = {"id": iid, "label": labels.get(iid, f"Incident {iid}")}
    st.session_state["is_running"] = True
    st.session_state["phase"] = "thinking"
    st.session_state["tick_count"] = 0
    schedule_next_tick()
    log(iid, "Started. Agent is analyzing network KPIs and bottlenecks.")

def advance_agent():
    if not st.session_state["is_running"]:
        return
    now = time.time()
    if st.session_state["next_tick_at"] is None or now < st.session_state["next_tick_at"]:
        return

    iid = st.session_state["current_incident"]["id"]
    phase = st.session_state["phase"]

    if phase == "thinking":
        st.session_state["tick_count"] += 1
        tick = st.session_state["tick_count"]
        msg = {
            1: "Scanning top corridors and DC headroom…",
            2: "Estimating re-route costs and service risk…",
            3: "Evaluating candidate interventions…",
        }.get(tick, "Analyzing…")
        log(iid, msg)
        if tick >= THINK_TICKS:
            st.session_state["phase"] = "acting"
            log(iid, "Decision taken. Executing intervention…")
        schedule_next_tick()
        return

    if phase == "acting":
        case = st.session_state["case"]
        prev = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if iid == 2:
            prev = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if iid == 3:
            prev = st.session_state["payloads"]["2"] or st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if not prev:
            prev = wrap_payload_as_snapshot(st.session_state["base"], "Baseline", st.session_state["base"].get("objective_cost"))

        plan = plan_incident_action(iid, case, prev)
        shock = plan.get("action", {})
        reason = plan.get("reason", "Selected action.")
        log(iid, reason)

        after = solve_min_cost_flow(case, shock)

        payload = {
            "title": st.session_state["current_incident"]["label"],
            "objective_before": prev.get("objective_after"),
            "objective_after": after.get("objective_cost"),
            "used_slacks": after.get("used_slacks"),
            "total_shortage": after.get("total_shortage"),
            "total_disposal": after.get("total_disposal"),
            "flows_after": after.get("flows", []),
            "currency": after.get("currency"),
            "flow_unit": after.get("flow_unit"),
            "lane": {k: shock[k] for k in ("src","dst","cost_per_unit","capacity") if k in shock} if shock.get("type")!="demand_spike" else None,
            "customer": shock.get("customer") if shock.get("type")=="demand_spike" else None
        }

        st.session_state["payloads"][str(iid)] = payload
        base_nodes = st.session_state["base"]["nodes"]
        base_products = st.session_state["base"]["products"]
        st.session_state["maps"][str(iid)] = build_map(base_nodes, payload.get("flows_after", []), base_products, key_suffix=f"i{iid}")

        ensure_explanation(iid, payload)
        st.session_state["phase"] = "evaluating"
        log(iid, "Intervention executed. Summarizing results…")
        schedule_next_tick()
        return

    if phase == "evaluating":
        st.session_state["is_running"] = False
        st.session_state["phase"] = "complete"
        log(iid, "Done. KPIs updated. You can now start the next scenario.")

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title(PAGE_TITLE)

# Sidebar — Gemini settings and logs
with st.sidebar:
    st.subheader("Agent explanations")
    st.caption("Add GOOGLE_API_KEY in Streamlit Secrets for Gemini. Otherwise a deterministic summary is used.")
    st.session_state["gem_model"] = st.text_input("Model", value=st.session_state.get("gem_model","gemini-1.5-pro"))
    st.session_state["gem_temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("gem_temperature",0.1)), 0.05)
    st.session_state["gem_max_tokens"] = st.number_input("Max tokens", 64, 2048, int(st.session_state.get("gem_max_tokens",300)), 32)

    st.divider()
    st.subheader("Agent logs")
    if st.session_state["current_incident"]:
        cur = st.session_state["current_incident"]["id"]
        st.caption(f"Current: Incident {cur}")
    tabs = st.tabs(["Incident 1", "Incident 2", "Incident 3"])
    for idx, tab in enumerate(tabs, start=1):
        with tab:
            logs = st.session_state["logs"].get(str(idx), [])
            if not logs:
                st.caption("No logs yet.")
            else:
                for entry in logs[-50:]:
                    st.write(f"**{entry['t']}** — {entry['msg']}")

st.divider()
agent = st.container()
with agent:
    st.subheader("Agent narrative")

# Controls
st.divider()
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    if st.button("🚀 Initialize baseline", disabled=st.session_state["is_running"]):
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
        st.session_state["phase"] = "baseline_ready"
        st.session_state["maps"]["0"] = build_map(base.get("nodes", []), base.get("flows", []), base.get("products", []), key_suffix="base")
        if base.get("ok"):
            st.success(f"Baseline ready. Total transport cost {fmt(base_obj)}.")
        else:
            st.warning("Baseline infeasible. You can still run scenarios to test the agent flow.")

with c2:
    st.button("▶️ Start Incident 1", key="btn_i1",
              disabled=st.session_state["is_running"] or (st.session_state.get("base") is None),
              on_click=lambda: start_incident(1))

with c3:
    st.button("▶️ Start Incident 2", key="btn_i2",
              disabled=st.session_state["is_running"] or (st.session_state.get("base") is None),
              on_click=lambda: start_incident(2))

with c4:
    st.button("▶️ Start Incident 3", key="btn_i3",
              disabled=st.session_state["is_running"] or (st.session_state.get("base") is None),
              on_click=lambda: start_incident(3))

# Comparison control
ref_options = []
if st.session_state["payloads"]["0"]: ref_options.append("Baseline")
if st.session_state["payloads"]["1"]: ref_options.append("Incident 1")
if st.session_state["payloads"]["2"]: ref_options.append("Incident 2")
if st.session_state["payloads"]["3"]: ref_options.append("Incident 3")
ref_label = st.selectbox("Compare KPIs against:", options=ref_options or ["Baseline"], index=0, key="compare_ref")

def get_payload_by_label(lbl: str) -> Optional[dict]:
    return {
        "Baseline": st.session_state["payloads"].get("0"),
        "Incident 1": st.session_state["payloads"].get("1"),
        "Incident 2": st.session_state["payloads"].get("2"),
        "Incident 3": st.session_state["payloads"].get("3"),
    }.get(lbl)

# Agent narrative rendering — now with safe percentage handling
with agent:
    ph = st.session_state["phase"]
    if ph == "baseline_ready" and st.session_state["payloads"]["0"]:
        p = st.session_state["payloads"]["0"]
        k = compute_kpis(st.session_state["case"], p)
        st.write(
            f"**Agent**: Baseline ready: total transport cost **{fmt(k['total_cost'])}**, "
            f"service fill rate **{fmt(pct(k.get('fill_rate')), places=1)}%**. "
            "This is our starting point to monitor cost and service."
        )
        st.markdown(kpi_explanation_md(k, "Baseline"))
    elif st.session_state["current_incident"] and st.session_state["is_running"]:
        iid = st.session_state["current_incident"]["id"]
        label = st.session_state["current_incident"]["label"]
        tick = st.session_state["tick_count"]
        steps = {
            0: f"{label}: Starting analysis…",
            1: "Analyzing corridor utilization and DC buffers…",
            2: "Comparing re-route options and cost deltas…",
            3: "Selecting the best intervention…",
        }
        st.write(f"**Agent**: {steps.get(tick, 'Working…')}")
        if st.session_state["next_tick_at"]:
            remain = max(0, int(st.session_state["next_tick_at"] - time.time()))
            st.caption(f"Next update in ~{remain}s")
    elif ph == "complete" and st.session_state["current_incident"]:
        iid = st.session_state["current_incident"]["id"]
        p = st.session_state["payloads"].get(str(iid)) or {}
        k_now = compute_kpis(st.session_state["case"], p)
        ref_payload = get_payload_by_label(ref_label) or st.session_state["payloads"]["0"]
        k_prev = compute_kpis(st.session_state["case"], ref_payload) if ref_payload else None
        st.write("**Agent**: " + headline_one_liner(p.get("title", f"Incident {iid}"), k_now, k_prev))
        st.markdown(kpi_explanation_md(k_now, p.get("title", f"Incident {iid}")))
        st.markdown("> " + (st.session_state["explanations"].get(str(iid)) or ""))
    else:
        st.caption("Ready. Initialize baseline, then start a scenario.")

# Baseline section
if st.session_state.get("base"):
    base = st.session_state["base"]
    st.subheader("Baseline")
    st.caption(f"Products: {', '.join(base.get('products', []))} | Nodes: {len(base.get('nodes', []))} | Flows: {len(base.get('flows', []))}")
    with st.container(border=True):
        st.markdown("**Network Map (Baseline)**")
        st_folium(st.session_state["maps"]["0"], width=None, height=520, key="map_base")
    show_flows_table(base.get("flows", []), caption="Baseline solved flows")

# Incident sections (render when available)
for iid in (1,2,3):
    payload = st.session_state["payloads"].get(str(iid))
    if not payload: continue
    st.divider()
    st.subheader(payload.get("title", f"Incident {iid}"))
    before, after = payload.get("objective_before"), payload.get("objective_after")

    ref_payload = get_payload_by_label(ref_label) or st.session_state["payloads"]["0"]
    ref_cost = ref_payload.get("objective_after") if ref_payload else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost (before)", fmt(before))
    c2.metric("Total cost (after)", fmt(after), delta=safe_delta(after, before))
    c3.metric(f"Δ vs {ref_label}", safe_delta(after, ref_cost))
    c4.metric("Used slacks?", "Yes" if payload.get("used_slacks") else "No")
    if payload.get("lane"):
        l = payload["lane"]
        tag = f"{l.get('src','?')} → {l.get('dst','?')}"
        if "cost_per_unit" in l: tag += f" @ {fmt(l['cost_per_unit'])}"
        st.caption(f"Lane changed: {tag}")
    if payload.get("customer"): st.caption(f"Customer impacted: {payload['customer']}")

    with st.container(border=True):
        st.markdown(f"**Network Map ({payload.get('title','Incident')})**")
        st_folium(st.session_state["maps"].get(str(iid)), width=None, height=520, key=f"map_i{iid}")
    show_flows_table(payload.get("flows_after", []), caption=f"Flows after {payload.get('title', f'Incident {iid}')}")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL: tick driver
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["is_running"]:
    if st.session_state["next_tick_at"] and time.time() >= st.session_state["next_tick_at"]:
        advance_agent()
        if st.session_state["is_running"]:
            schedule_next_tick()
        st.experimental_rerun()
