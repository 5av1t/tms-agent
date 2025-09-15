import os, io, json, math, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import pulp
import folium
from streamlit_folium import st_folium

# Optional Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
PAGE_TITLE = "TMS Agent — Control Tower (PuLP)"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

DATA_PATH = os.path.join("data", "base_case.xlsx")
TICK_INTERVAL = 3.0        # seconds between beats
THINK_TICKS = 3            # analyze beats before act
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

# ─────────────────────────────────────────────────────────────
# Small utils
# ─────────────────────────────────────────────────────────────
def num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def fmt(x, none="—", places=2):
    if x is None:
        return none
    try:
        v = float(x)
        if np.isnan(v):
            return none
        return f"{v:,.{places}f}"
    except Exception:
        return none

def pct(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return 100.0 * v
    except Exception:
        return None

def safe_delta(after, before, none="—", places=2):
    try:
        if after is None or before is None:
            return none
        a = float(after); b = float(before)
        if np.isnan(a) or np.isnan(b):
            return none
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
        want = {_norm(c) for c in candidates}
        for c in cols:
            if _norm(c) in want:
                return c
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R=6371.0
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
def ensure_state():
    ss = st.session_state
    ss.setdefault("case", None)
    ss.setdefault("base", None)
    ss.setdefault("maps", {})
    ss.setdefault("total_demand", None)

    ss.setdefault("phase", "idle")      # idle|baseline_ready|thinking|acting|evaluating|complete
    ss.setdefault("is_running", False)
    ss.setdefault("current_incident", None)
    ss.setdefault("tick_count", 0)
    ss.setdefault("next_tick_at", None)
    ss.setdefault("logs", {"1": [], "2": [], "3": []})

    ss.setdefault("payloads", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("progress_total", 0)
    ss.setdefault("progress_done", 0)
    ss.setdefault("acting_stage", None)   # None|"plan"|"apply"
    ss.setdefault("plan_cache", None)

    ss.setdefault("gem_model", "gemini-1.5-pro")
    ss.setdefault("gem_temperature", 0.1)
    ss.setdefault("gem_max_tokens", 300)
    ss.setdefault("explanations", {"0": None, "1": None, "2": None, "3": None})
ensure_state()

def log(iid: int, msg: str):
    st.session_state["logs"].setdefault(str(iid), []).append({"t": time.strftime("%H:%M:%S"), "msg": msg})

def schedule_next_tick():
    st.session_state["next_tick_at"] = time.time() + TICK_INTERVAL

# ─────────────────────────────────────────────────────────────
# Data loading (AIMMS-like worksheet set)
# ─────────────────────────────────────────────────────────────
def load_excel() -> pd.ExcelFile:
    if os.path.exists(DATA_PATH):
        return pd.ExcelFile(DATA_PATH)
    up = st.file_uploader("Upload base_case.xlsx", type=["xlsx"])
    if up is None:
        st.error("Data file missing. Add `data/base_case.xlsx` or upload here.")
        st.stop()
    return pd.ExcelFile(io.BytesIO(up.read()))

def read_case() -> dict:
    xls = load_excel()

    customers_df = pd.read_excel(xls, "Customers")
    demand_df    = pd.read_excel(xls, "Customer Product Data")
    loc_df       = pd.read_excel(xls, "Location")
    lanes_tpl    = pd.read_excel(xls, "Transport Cost")
    groups_df    = pd.read_excel(xls, "Location Groups")
    supp_prod    = pd.read_excel(xls, "Supplier Product")
    wh_df        = pd.read_excel(xls, "Warehouse")

    for df in (customers_df,demand_df,loc_df,lanes_tpl,groups_df,supp_prod,wh_df):
        df.columns = [str(c).strip() for c in df.columns]

    # Customers
    cust_id_col = _find_col(customers_df, ["customer","name","id"])
    if not cust_id_col:
        st.error("Customers sheet must contain Customer/Name/ID"); st.stop()
    customers = customers_df[cust_id_col].dropna().astype(str).tolist()

    # Demand
    dem_cust_col = _find_col(demand_df, ["customer","name","id"])
    dem_prod_col = _find_col(demand_df, ["product"])
    dem_qty_col  = _find_col(demand_df, ["demand","qty","quantity","volume"])
    if not all([dem_cust_col,dem_prod_col,dem_qty_col]):
        st.error("Customer Product Data must contain Customer, Product, Demand"); st.stop()
    dem = demand_df[[dem_cust_col,dem_prod_col,dem_qty_col]].copy()
    dem.columns = ["customer","product","demand"]
    dem["demand"] = pd.to_numeric(dem["demand"], errors="coerce").fillna(0.0)
    demand = dem.groupby(["customer","product"])["demand"].sum().reset_index()
    st.session_state["total_demand"] = float(demand["demand"].sum())

    # Locations
    loc_id_col = _find_col(loc_df, ["location","name","id"])
    lat_col = _find_col(loc_df, ["lat"], contains=True)
    lon_col = _find_col(loc_df, ["lon","lng","long"], contains=True)
    if not all([loc_id_col,lat_col,lon_col]):
        st.error("Location sheet must contain Location, Latitude, Longitude"); st.stop()
    loc = loc_df[[loc_id_col,lat_col,lon_col]].dropna(subset=[loc_id_col]).copy()
    loc.columns = ["id","lat","lon"]
    coord = {r["id"]: {"lat": float(r["lat"]), "lon": float(r["lon"])} for _, r in loc.iterrows()}

    # Location groups (FG suppliers & DCs)
    if "Location" not in groups_df.columns or "SubLocation" not in groups_df.columns:
        st.error("Location Groups must contain Location & SubLocation"); st.stop()
    FG = set(groups_df[groups_df["Location"]=="FG"]["SubLocation"].dropna().astype(str))
    DC = set(groups_df[groups_df["Location"]=="DC"]["SubLocation"].dropna().astype(str))

    # Supply (Supplier Product)
    sp = supp_prod.rename(columns={"Location":"supplier"}).copy()
    if not {"supplier","Product"}.issubset(sp.columns):
        st.error("Supplier Product must include Location (supplier) and Product"); st.stop()
    cap_col = _find_col(sp, ["Maximum Capacity","Capacity","Cap"])
    if cap_col is None:
        sp["Maximum Capacity"]=0.0; cap_col="Maximum Capacity"
    sp["supply"] = pd.to_numeric(sp[cap_col], errors="coerce").fillna(0.0)
    if "Available" in sp.columns:
        sp["avail"]=pd.to_numeric(sp["Available"], errors="coerce").fillna(1.0)
        sp = sp[sp["avail"]>0]
    sp = sp[["supplier","Product","supply"]].rename(columns={"Product":"product"})
    sp["supplier"]=sp["supplier"].astype(str)
    sp = sp[sp["supplier"].isin(FG)]
    supply = sp.groupby(["supplier","product"])["supply"].sum().reset_index()

    # Products
    products = sorted(demand["product"].dropna().astype(str).unique().tolist())

    # Transport templates
    def _tpl_row(fr,to):
        if "From Location" in lanes_tpl.columns and "To Location" in lanes_tpl.columns:
            hit = lanes_tpl[(lanes_tpl.get("From Location")==fr) & (lanes_tpl.get("To Location")==to)]
        else:
            hit = pd.DataFrame()
        if len(hit)>0:
            r=hit.iloc[0].to_dict()
            return {"cpd": float(r.get("Cost per Distance",1.0) or 1.0),
                    "cpu": float(r.get("Cost Per UOM",0.0) or 0.0)}
        return {"cpd": 1.0 if fr=="FG" else 2.0, "cpu": 5.0 if fr=="FG" else 10.0}
    tpl_fg_dc = _tpl_row("FG","DC")
    tpl_dc_ct = _tpl_row("DC","City")

    # Nodes
    nodes=[]
    warehouses = set(wh_df.get("Location", pd.Series(dtype=str)).dropna().astype(str)) | DC
    for s in FG:
        c=coord.get(s); nodes.append({"id":s,"lat":c["lat"] if c else None,"lon":c["lon"] if c else None,"kind":"supplier"})
    for d in warehouses:
        c=coord.get(d); nodes.append({"id":d,"lat":c["lat"] if c else None,"lon":c["lon"] if c else None,"kind":"dc"})
    for cst in customers:
        c=coord.get(cst); nodes.append({"id":cst,"lat":c["lat"] if c else None,"lon":c["lon"] if c else None,"kind":"customer"})
    nodes_df=pd.DataFrame(nodes).drop_duplicates(subset=["id"])

    # Lanes
    def d_latlon(a,b):
        if a not in coord or b not in coord:
            return None
        return haversine_km(coord[a]["lat"],coord[a]["lon"],coord[b]["lat"],coord[b]["lon"])

    lanes=[]
    # FG->DC
    for s in FG:
        if s not in coord: 
            continue
        for d in warehouses:
            if d not in coord:
                continue
            dist=d_latlon(s,d)
            if dist is None:
                continue
            for p in products:
                sup_p=float(supply[(supply["supplier"]==s)&(supply["product"]==p)]["supply"].sum())
                cap=max(0.0, sup_p)         # cap = available supply for that (s,p)
                cost=tpl_fg_dc["cpu"] + tpl_fg_dc["cpd"]*dist
                lanes.append({"src":s,"dst":d,"product":p,"capacity":cap,"cost_per_unit":cost})
    # DC->City
    for d in warehouses:
        if d not in coord:
            continue
        for cst in customers:
            if cst not in coord:
                continue
            dist=d_latlon(d,cst)
            if dist is None:
                continue
            for p in products:
                dem_p=float(demand[(demand["customer"]==cst)&(demand["product"]==p)]["demand"].sum())
                cap=(dem_p*2.0)+1e6
                cost=tpl_dc_ct["cpu"] + tpl_dc_ct["cpd"]*dist
                lanes.append({"src":d,"dst":cst,"product":p,"capacity":cap,"cost_per_unit":cost})

    return {"nodes":nodes_df,"demand":demand,"supply":supply,
            "lanes":pd.DataFrame(lanes),"products":products}

# ─────────────────────────────────────────────────────────────
# Optimizer (REWRITTEN WITH PULP)
# ─────────────────────────────────────────────────────────────
def solve_min_cost_flow(case: dict, shock: dict | None = None) -> dict:
    nodes = case["nodes"]
    demand_df = case["demand"].copy()
    supply_df = case["supply"].copy()
    lanes_df  = case["lanes"].copy()

    # Apply incident shock
    if shock:
        t = shock.get("type")
        if t == "lane_cap":
            cond = (lanes_df["src"]==shock["src"]) & (lanes_df["dst"]==shock["dst"])
            if shock.get("product") is not None:
                cond &= (lanes_df["product"]==shock["product"])
            lanes_df.loc[cond,"capacity"]=float(shock["new_capacity"])
        elif t == "demand_spike":
            cond = (demand_df["customer"]==shock["customer"])
            demand_df.loc[cond,"demand"]=demand_df.loc[cond,"demand"]*(1.0+float(shock["pct"]))
        elif t == "express_lane":
            newrow={"src":shock["src"],"dst":shock["dst"],"product":shock["product"],
                    "capacity":float(shock["capacity"]),"cost_per_unit":float(shock["cost_per_unit"])}
            lanes_df=pd.concat([lanes_df,pd.DataFrame([newrow])],ignore_index=True)

    # --- PuLP Model ---
    prob = pulp.LpProblem("MinCostFlow", pulp.LpMinimize)

    # Variables
    lane_keys = [(r['src'], r['dst'], r['product']) for _, r in lanes_df.iterrows()]
    flow_vars = pulp.LpVariable.dicts("flow", lane_keys, lowBound=0, cat='Continuous')

    supp_prod_keys = [(r['supplier'], r['product']) for _, r in supply_df.iterrows()]
    disposal_vars = pulp.LpVariable.dicts("disposal", supp_prod_keys, lowBound=0, cat='Continuous')
    
    cust_prod_keys = [(r['customer'], r['product']) for _, r in demand_df.iterrows()]
    shortage_vars = pulp.LpVariable.dicts("shortage", cust_prod_keys, lowBound=0, cat='Continuous')
    
    BIGM = 1e6 # Penalty for slacks

    # Objective Function
    cost_expr = pulp.lpSum(
        lanes_df.loc[i, 'cost_per_unit'] * flow_vars[(r['src'], r['dst'], r['product'])]
        for i, r in lanes_df.iterrows()
    )
    disposal_penalty = BIGM * pulp.lpSum(disposal_vars)
    shortage_penalty = BIGM * pulp.lpSum(shortage_vars)
    prob += cost_expr + disposal_penalty + shortage_penalty, "Total_Cost_with_Penalties"
    
    # Constraints
    # Supply constraints: sum(outflow) + disposal = supply
    for s, p in supp_prod_keys:
        outflow = pulp.lpSum(flow_vars.get((s, d, p)) for d in nodes['id'] if (s, d, p) in flow_vars)
        supply_val = float(supply_df[(supply_df["supplier"]==s)&(supply_df["product"]==p)]["supply"].sum())
        prob += outflow + disposal_vars[(s, p)] == supply_val, f"supply_{s}_{p}"

    # Demand constraints: sum(inflow) + shortage = demand
    for cst, p in cust_prod_keys:
        inflow = pulp.lpSum(flow_vars.get((s, cst, p)) for s in nodes['id'] if (s, cst, p) in flow_vars)
        demand_val = float(demand_df[(demand_df["customer"]==cst)&(demand_df["product"]==p)]["demand"].sum())
        prob += inflow + shortage_vars[(cst, p)] == demand_val, f"demand_{cst}_{p}"

    # Lane capacity constraints: flow <= capacity
    for _, r in lanes_df.iterrows():
        key = (r['src'], r['dst'], r['product'])
        prob += flow_vars[key] <= r['capacity'], f"cap_{r['src']}_{r['dst']}_{r['product']}"
        
    # Solve
    prob.solve()
    
    # --- Extract Results ---
    is_optimal = pulp.LpStatus[prob.status] == 'Optimal'
    
    total_shortage = 0.0
    total_disposal = 0.0
    flows = []
    objective = None

    if is_optimal:
        objective = pulp.value(prob.objective)
        for key, var in flow_vars.items():
            q = var.value()
            if q is not None and q > 1e-6:
                flows.append({"src": key[0], "dst": key[1], "product": key[2], "flow": q})
        
        total_disposal = sum(var.value() for var in disposal_vars.values() if var.value() is not None)
        total_shortage = sum(var.value() for var in shortage_vars.values() if var.value() is not None)

    # Nodes
    nrecs=[]
    for _,r in nodes.iterrows():
        nrecs.append({"id":str(r.get("id")),"kind":str(r.get("kind")),
                      "lat":float(r["lat"]) if pd.notna(r["lat"]) else None,
                      "lon":float(r["lon"]) if pd.notna(r["lon"]) else None})

    products_list = case.get("products", sorted(set([f["product"] for f in flows])))

    return {"ok": is_optimal,
            "objective_cost": objective,
            "used_slacks": (total_disposal > 1e-6) or (total_shortage > 1e-6),
            "total_shortage": total_shortage,
            "total_disposal": total_disposal,
            "nodes": nrecs, "products":products_list,
            "flows": flows,
            "currency":"EUR","flow_unit":"units"}


# ─────────────────────────────────────────────────────────────
# KPIs & narrative
# ─────────────────────────────────────────────────────────────
def compute_kpis(case: dict, payload: dict) -> dict:
    cost = payload.get("objective_after")
    flows = payload.get("flows_after") or payload.get("flows") or []
    total_flow = float(sum(num(f.get("flow")) for f in flows))
    lanes_used = len({(f["src"],f["dst"],f["product"]) for f in flows})
    avg_cpu = (cost/total_flow) if (cost is not None and total_flow>0) else None
    shortage = payload.get("total_shortage")
    disposal = payload.get("total_disposal")
    total_demand = st.session_state.get("total_demand")
    fill_rate=None
    if total_demand is not None and shortage is not None:
        denom=max(1e-9,float(total_demand))
        fill_rate=max(0.0,min(1.0,1.0-float(shortage)/denom))
    return {"total_cost":cost,"total_flow_units":total_flow,"lanes_used":lanes_used,"avg_cost_per_unit":avg_cpu,
            "shortage_units":shortage,"fill_rate":fill_rate,"disposal_units":disposal,
            "currency": payload.get("currency","EUR"), "flow_unit": payload.get("flow_unit","units")}

def kpi_explanation_md(k: dict, label: str) -> str:
    return (
        f"**{label} — KPI Summary**\n"
        f"- **Total transport cost**: {fmt(k.get('total_cost'))} {k.get('currency','')}.\n"
        f"- **Total shipped volume**: {fmt(k.get('total_flow_units'))} {k.get('flow_unit','')}.\n"
        f"- **Lanes used**: {fmt(k.get('lanes_used'),places=0)} lanes with non-zero flow.\n"
        f"- **Avg. cost per unit**: {fmt(k.get('avg_cost_per_unit'))} {k.get('currency','')}/{k.get('flow_unit','')}.\n"
        f"- **Shortage volume**: {fmt(k.get('shortage_units'))} {k.get('flow_unit','')} not fulfilled.\n"
        f"- **Service fill rate**: {fmt(pct(k.get('fill_rate')),places=1)}%.\n"
        f"- **Disposals (waste)**: {fmt(k.get('disposal_units'))} {k.get('flow_unit','')} scrapped.\n\n"
        "**KPI definitions**\n"
        "- *Total transport cost*: Optimizer’s objective (sum of lane costs).\n"
        "- *Total shipped volume*: Sum of all product flows across the network.\n"
        "- *Lanes used*: Count of distinct corridors that carried any volume.\n"
        "- *Avg. cost per unit*: Total cost divided by shipped volume.\n"
        "- *Shortage volume*: Demand left unserved (allowed at heavy penalty).\n"
        "- *Service fill rate*: Served demand / Total demand.\n"
        "- *Disposals*: Unused supply disposed (heavily penalized).\n"
    )

def headline_one_liner(title: str, k_now: dict, k_prev: Optional[dict]) -> str:
    cost = k_now.get("total_cost"); fill = k_now.get("fill_rate")
    cost_txt = f"total transport cost **{fmt(cost)}**" if cost is not None else "transport cost (n/a)"
    fill_txt = f"service fill rate **{fmt(pct(fill),places=1)}%**" if fill is not None else "service (n/a)"
    delta_cost = None; delta_fill = None
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
        parts.append(f"{'↑' if delta_fill>0 else '↓'} fill {fmt(abs(delta_fill),places=1)} pp;")
    parts.append(fill_txt + ".")
    parts.append(f" Driver: *{driver}*.")
    return " ".join(parts)

SYSTEM_PROMPT = (
    "You are a supply-chain control-tower analyst for freight TMS.\n"
    "Use ONLY the JSON facts provided. If missing, say 'insufficient data'.\n"
    "Output:\nVERIFIED\n- objective_before: <v>\n- objective_after: <v>\n- delta: <v>\n- lane: <src> -> <dst>\n"
    "- cost_per_unit: <v>\n- capacity: <v>\n- used_slacks: <true/false>\n- units: currency=<...>, flow_unit=<...>\n\n"
    "EXPLANATION\n- Why did objective move? Options considered (if supported). Decision & reasoning. Risks & next steps.\n"
)

def deterministic_explanation(idx: int, payload: dict) -> str:
    before = payload.get("objective_before"); after = payload.get("objective_after")
    bits = [payload.get("title", f"Incident {idx}"), "."]
    if before is not None and after is not None:
        bits.append(f" Objective moved {fmt(before)} → {fmt(after)} (Δ {safe_delta(after,before)}).")
    l = payload.get("lane")
    if l:
        parts=[]
        if "src" in l and "dst" in l: parts.append(f"{l['src']}→{l['dst']}")
        if "cost_per_unit" in l: parts.append(f"cpu={fmt(l['cost_per_unit'])}")
        if "capacity" in l: parts.append(f"cap={fmt(l['capacity'])}")
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
        model = genai.GenerativeModel(st.session_state.get("gem_model","gemini-1.5-pro"),
                                        system_instruction=SYSTEM_PROMPT)
        cfg = {"temperature": float(st.session_state.get("gem_temperature",0.1)),
               "max_output_tokens": int(st.session_state.get("gem_max_tokens",300))}
        ctx = {"incident_index": idx, "title": payload.get("title"),
               "objective_before": payload.get("objective_before"),
               "objective_after": payload.get("objective_after"),
               "used_slacks": payload.get("used_slacks"),
               "total_shortage": payload.get("total_shortage"),
               "total_disposal": payload.get("total_disposal"),
               "lane": payload.get("lane"), "customer": payload.get("customer"),
               "units":{"currency":payload.get("currency","EUR"),"flow_unit":payload.get("flow_unit","units")}}
        resp = model.generate_content("USER JSON:\n"+json.dumps(ctx), generation_config=cfg)
        text = (resp.text or "").strip()
        if any(w in text.lower() for w in ["passenger","ridership","ticket"]): # guard
            return deterministic_explanation(idx, payload)
        return text if text else deterministic_explanation(idx, payload)
    except Exception:
        return deterministic_explanation(idx, payload)

def ensure_explanation(idx: int, payload: dict):
    k=str(idx)
    if not st.session_state["explanations"].get(k):
        st.session_state["explanations"][k] = gemini_explain(idx, payload)

# ─────────────────────────────────────────────────────────────
# Maps
# ─────────────────────────────────────────────────────────────
def _center_latlon(nodes: List[Dict]) -> Tuple[float,float]:
    lats=[n.get("lat") for n in nodes if n.get("lat") is not None]
    lons=[n.get("lon") for n in nodes if n.get("lon") is not None]
    if not lats or not lons:
        return (20.5937,78.9629)
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def _product_color_map(products: List[str]) -> Dict[str,str]:
    return {p: PALETTE[i % len(PALETTE)] for i,p in enumerate(sorted(products))}

def _node_by_id(nodes: List[Dict]) -> Dict[str,Dict]:
    return {n["id"]: n for n in nodes}

def _flow_weight_scaler(flows: List[Dict], min_w=0.6, max_w=2.0):
    vals=sorted(float(f.get("flow",0.0)) for f in flows if f.get("flow",0.0)>0)
    if not vals:
        return lambda q: min_w
    p90=vals[min(int(0.9*(len(vals)-1)), len(vals)-1)]
    if not p90:
        p90 = max(vals) if vals else 1.0
    def to_w(q):
        if q<=0:
            return 0.0
        w=min_w + (max_w-min_w)*(q/(p90*1.0))
        return max(min_w, min(max_w, w))
    return to_w

def build_map(nodes: List[Dict], flows: List[Dict], products: List[str], key_suffix: str) -> folium.Map:
    node_by_id=_node_by_id(nodes)
    lat0,lon0=_center_latlon(nodes)
    m=folium.Map(location=[lat0,lon0], zoom_start=5, tiles="cartodbpositron")
    # nodes
    for n in nodes:
        if n.get("lat") is None or n.get("lon") is None:
            continue
        kind=n.get("kind","")
        color="#198754" if kind not in ("supplier","dc") else ("#6f42c1" if kind=="supplier" else "#0d6efd")
        radius=6 if kind in ("supplier","dc") else 5
        folium.CircleMarker((n["lat"],n["lon"]), radius=radius, color="#1d1d1d", weight=1,
                            fill=True, fill_color=color, fill_opacity=0.9,
                            popup=folium.Popup(html=f"{n['id']} ({kind})", max_width=250)).add_to(m)
    # flows
    to_w=_flow_weight_scaler(flows); color_by=_product_color_map(products)
    for f in flows:
        q=float(f.get("flow",0))
        if q<=0:
            continue
        u=f["src"]; v=f["dst"]
        urec=node_by_id.get(u); vrec=node_by_id.get(v)
        if not urec or not vrec:
            continue
        if None in (urec.get("lat"),urec.get("lon"),vrec.get("lat"),vrec.get("lon")):
            continue
        folium.PolyLine([(urec["lat"],urec["lon"]),(vrec["lat"],vrec["lon"])],
                        weight=to_w(q), color=color_by.get(f.get("product",""),"#555"),
                        opacity=0.55, tooltip=f"{u} → {v} | {f.get('product','')}: {q:,.2f}").add_to(m)
    return m

def show_flows_table(flows: list[dict], caption="Solved flows"):
    if not flows:
        st.info("No flows to display.")
        return
    df=pd.DataFrame(flows).sort_values(["src","dst","product"])
    st.dataframe(df, use_container_width=True, height=300)
    st.caption(caption)

# ─────────────────────────────────────────────────────────────
# Incidents planner (agent)
# ─────────────────────────────────────────────────────────────
def pick_top_lane(payload: dict) -> Optional[dict]:
    flows = payload.get("flows_after") or payload.get("flows") or []
    if not flows:
        return None
    return sorted(flows, key=lambda x: x["flow"], reverse=True)[0]

def plan_incident_action(iid: int, case: dict, prev: dict) -> dict:
    top = pick_top_lane(prev) or {}
    candidates=[]
    if top:
        candidates.append({"type":"lane_cap","label":"Temporarily restrict a congested corridor",
                           "shock":{"type":"lane_cap","src":top["src"],"dst":top["dst"],"product":top["product"],
                                    "new_capacity": max(0.5*float(top["flow"]),1.0)}})
        candidates.append({"type":"express_lane","label":"Add cheaper express lane on top corridor",
                           "shock":{"type":"express_lane","src":top["src"],"dst":top["dst"],"product":top["product"],
                                    "capacity": max(0.5*float(top["flow"]),1.0),"cost_per_unit":0.5}})
    # hottest customer demand surge
    dest_tot={}
    for f in prev.get("flows_after") or prev.get("flows") or []:
        dest_tot[f["dst"]]=dest_tot.get(f["dst"],0.0)+f["flow"]
    if dest_tot:
        hot=sorted(dest_tot.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        candidates.append({"type":"demand_spike","label":"+25% demand at hottest customer",
                           "shock":{"type":"demand_spike","customer":hot,"pct":0.25}})

    # Optional Gemini selection
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            pref={1:"lane_cap",2:"demand_spike",3:"express_lane"}.get(iid,"lane_cap")
            model=genai.GenerativeModel(st.session_state.get("gem_model","gemini-1.5-pro"),
                                system_instruction=("Pick one candidate for the incident id. Prefer lane_cap for 1, "
                                                    "demand_spike for 2, express_lane for 3. Return JSON {index}."))
            resp=model.generate_content(json.dumps({"incident_id":iid,"preferred":pref,"candidates":candidates}),
                                        generation_config={"temperature":float(st.session_state.get("gem_temperature",0.1)),
                                                           "max_output_tokens":128})
            j=None
            try:
                j=json.loads((resp.text or "").strip())
            except Exception:
                j=None
            if j and isinstance(j.get("index"),int) and 0<=j["index"]<len(candidates):
                choice=candidates[j["index"]]
            else:
                order={"lane_cap":0,"demand_spike":1,"express_lane":2}
                ranked=sorted(candidates,key=lambda c: abs(order.get(c["type"],10)-order.get(pref,0)))
                choice=ranked[0] if ranked else (candidates[0] if candidates else {})
            return {"action": choice.get("shock", {}), "reason": f"Agent selected '{choice.get('label','action')}'."}
        except Exception:
            pass

    # Deterministic preference
    if iid==1 and any(c["type"]=="lane_cap" for c in candidates):
        choice=next(c for c in candidates if c["type"]=="lane_cap")
    elif iid==2 and any(c["type"]=="demand_spike" for c in candidates):
        choice=next(c for c in candidates if c["type"]=="demand_spike")
    elif iid==3 and any(c["type"]=="express_lane" for c in candidates):
        choice=next(c for c in candidates if c["type"]=="express_lane")
    else:
        choice=candidates[0] if candidates else {"shock":{},"label":"No-op"}
    return {"action": choice.get("shock", {}), "reason": f"Selected deterministic: {choice.get('label','action')}"}

# ─────────────────────────────────────────────────────────────
# Baseline & payload wrapper
# ─────────────────────────────────────────────────────────────
def build_baseline() -> tuple[dict, dict]:
    case = read_case()
    base = solve_min_cost_flow(case, None)
    base["title"]="Baseline"
    return base, case

def wrap_payload(payload_after: dict, title: str, before_cost: Optional[float]) -> dict:
    return {"title":title,"objective_before":before_cost,"objective_after":payload_after.get("objective_cost"),
            "used_slacks":payload_after.get("used_slacks"),
            "total_shortage":payload_after.get("total_shortage"),
            "total_disposal":payload_after.get("total_disposal"),
            "flows_after":payload_after.get("flows",[]),
            "currency":payload_after.get("currency"),"flow_unit":payload_after.get("flow_unit"),
            "lane":None,"customer":None}

# ─────────────────────────────────────────────────────────────
# Agent state machine (beats synchronized with logs)
# ─────────────────────────────────────────────────────────────
def start_incident(iid: int):
    if st.session_state["is_running"]:
        st.warning("Another scenario is running. Please wait.")
        return
    if not st.session_state.get("base"):
        st.warning("Initialize baseline first.")
        return
    labels={1:"Incident 1 — Corridor capacity restriction",
            2:"Incident 2 — Demand surge (+25%)",
            3:"Incident 3 — Strategic express lane"}
    st.session_state["current_incident"]={"id":iid,"label":labels.get(iid,f"Incident {iid}")}
    st.session_state["is_running"]=True
    st.session_state["phase"]="thinking"
    st.session_state["tick_count"]=0
    st.session_state["progress_total"]=THINK_TICKS+3
    st.session_state["progress_done"]=0
    st.session_state["acting_stage"]=None
    st.session_state["plan_cache"]=None
    schedule_next_tick()
    log(iid,"Started. Agent is analyzing network KPIs and bottlenecks.")

def advance_agent():
    if not st.session_state["is_running"]:
        return
    if st.session_state["next_tick_at"] is None or time.time()<st.session_state["next_tick_at"]:
        return
    iid=st.session_state["current_incident"]["id"]
    phase=st.session_state["phase"]

    # THINKING
    if phase=="thinking":
        st.session_state["tick_count"]+=1
        st.session_state["progress_done"]+=1
        msg={1:"Scanning top corridors and DC headroom…",
             2:"Estimating re-route costs and service risk…",
             3:"Evaluating candidate interventions…",}.get(st.session_state["tick_count"],"Analyzing…")
        log(iid,msg)
        if st.session_state["tick_count"]>=THINK_TICKS:
            st.session_state["phase"]="acting"
            st.session_state["acting_stage"]="plan"
        schedule_next_tick()
        return

    # ACTING — plan
    if phase=="acting" and st.session_state["acting_stage"]=="plan":
        case=st.session_state["case"]
        prev = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if iid==2:
            prev=st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if iid==3:
            prev=st.session_state["payloads"]["2"] or st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        if not prev:
            prev=wrap_payload(st.session_state["base"],"Baseline",st.session_state["base"].get("objective_cost"))
        plan=plan_incident_action(iid,case,prev)
        st.session_state["plan_cache"]={"prev":prev,"plan":plan}
        log(iid, f"Decision taken: {plan.get('reason','Selected action.')}")
        st.session_state["acting_stage"]="apply"
        st.session_state["progress_done"]+=1
        schedule_next_tick()
        return

    # ACTING — apply
    if phase=="acting" and st.session_state["acting_stage"]=="apply":
        case=st.session_state["case"]
        cached=st.session_state["plan_cache"] or {}
        prev=cached.get("prev",{})
        plan=cached.get("plan",{})
        shock=(plan or {}).get("action",{})
        after=solve_min_cost_flow(case, shock)
        payload={"title":st.session_state["current_incident"]["label"],
                 "objective_before": prev.get("objective_after"),
                 "objective_after": after.get("objective_cost"),
                 "used_slacks": after.get("used_slacks"),
                 "total_shortage": after.get("total_shortage"),
                 "total_disposal": after.get("total_disposal"),
                 "flows_after": after.get("flows", []),
                 "currency": after.get("currency"), "flow_unit": after.get("flow_unit"),
                 "lane": {k: shock[k] for k in ("src","dst","cost_per_unit","capacity") if k in shock} if shock.get("type")!="demand_spike" else None,
                 "customer": shock.get("customer") if shock.get("type")=="demand_spike" else None}
        st.session_state["payloads"][str(iid)]=payload

        base_nodes=st.session_state["base"]["nodes"]
        base_products=st.session_state["base"]["products"]
        st.session_state["maps"][str(iid)]=build_map(base_nodes, payload.get("flows_after", []), base_products, key_suffix=f"i{iid}")

        ensure_explanation(iid,payload)
        log(iid,"Intervention executed. Summarizing results…")
        st.session_state["phase"]="evaluating"
        st.session_state["acting_stage"]=None
        st.session_state["plan_cache"]=None
        st.session_state["progress_done"]+=1
        schedule_next_tick()
        return

    # EVALUATING
    if phase=="evaluating":
        st.session_state["is_running"]=False
        st.session_state["phase"]="complete"
        st.session_state["progress_done"]=min(st.session_state["progress_total"], st.session_state["progress_done"]+1)
        log(iid,"Done. KPIs updated. You can now start the next scenario.")

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title(PAGE_TITLE)

with st.sidebar:
    st.subheader("Agent explanations")
    st.caption("Add GOOGLE_API_KEY in Secrets for Gemini; otherwise deterministic text is used.")
    st.session_state["gem_model"]=st.text_input("Model", value=st.session_state.get("gem_model","gemini-1.5-pro"))
    st.session_state["gem_temperature"]=st.slider("Temperature",0.0,1.0,float(st.session_state.get("gem_temperature",0.1)),0.05)
    st.session_state["gem_max_tokens"]=st.number_input("Max tokens",64,2048,int(st.session_state.get("gem_max_tokens",300)),32)
    st.divider()
    st.subheader("Agent logs")
    tabs=st.tabs(["Incident 1","Incident 2","Incident 3"])
    for idx, tab in enumerate(tabs, start=1):
        with tab:
            logs=st.session_state["logs"].get(str(idx),[])
            if not logs:
                st.caption("No logs yet.")
            else:
                for e in logs[-100:]:
                    st.write(f"**{e['t']}** — {e['msg']}")

st.divider()
agent = st.container()
with agent:
    st.subheader("Agent narrative")

st.divider()
c1,c2,c3,c4 = st.columns([1,1,1,1])

with c1:
    if st.button("🚀 Initialize baseline", disabled=st.session_state["is_running"]):
        case = read_case()
        base = solve_min_cost_flow(case, None)
        st.session_state["case"]=case
        st.session_state["base"]=base
        base_obj=base.get("objective_cost")
        st.session_state["payloads"]["0"]={"title":"Baseline","objective_before":base_obj,"objective_after":base_obj,
                                           "flows_after":base.get("flows",[]),"currency":base.get("currency"),
                                           "flow_unit":base.get("flow_unit")}
        st.session_state["maps"]["0"]=build_map(base.get("nodes",[]), base.get("flows",[]), base.get("products",[]), key_suffix="base")
        st.session_state["phase"]="baseline_ready"
        if base.get("ok"):
            st.success(f"Baseline ready. Total transport cost {fmt(base_obj)}.")
        else:
            st.warning("Baseline infeasible. You can still run scenarios.")

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

# Compare dropdown
ref_options=[]
if st.session_state["payloads"]["0"]: ref_options.append("Baseline")
if st.session_state["payloads"]["1"]: ref_options.append("Incident 1")
if st.session_state["payloads"]["2"]: ref_options.append("Incident 2")
if st.session_state["payloads"]["3"]: ref_options.append("Incident 3")
ref_label = st.selectbox("Compare KPIs against:", options=ref_options or ["Baseline"], index=0, key="compare_ref")

def get_payload_by_label(lbl: str) -> Optional[dict]:
    return {"Baseline":st.session_state["payloads"].get("0"),
            "Incident 1":st.session_state["payloads"].get("1"),
            "Incident 2":st.session_state["payloads"].get("2"),
            "Incident 3":st.session_state["payloads"].get("3")}.get(lbl)

# Agent narrative
with agent:
    ph=st.session_state["phase"]
    if ph=="baseline_ready" and st.session_state["payloads"]["0"]:
        p=st.session_state["payloads"]["0"]
        k=compute_kpis(st.session_state["case"], p)
        st.write(f"**Agent**: Baseline ready: total transport cost **{fmt(k['total_cost'])}**, "
                 f"service fill rate **{fmt(pct(k.get('fill_rate')),places=1)}%**. "
                 "This is our starting point to monitor cost and service.")
        st.markdown(kpi_explanation_md(k,"Baseline"))

    elif st.session_state["current_incident"] and st.session_state["is_running"]:
        iid=st.session_state["current_incident"]["id"]
        label=st.session_state["current_incident"]["label"]
        tick=st.session_state["tick_count"]
        if ph=="thinking":
            msg={0:f"{label}: Starting analysis…",
                 1:"Analyzing corridor utilization and DC buffers…",
                 2:"Comparing re-route options and cost deltas…",
                 3:"Selecting the best intervention…",}.get(tick,"Analyzing…")
        elif ph=="acting" and st.session_state["acting_stage"]=="plan":
            msg="Committing to selected intervention…"
        elif ph=="acting" and st.session_state["acting_stage"]=="apply":
            msg="Applying intervention to network model…"
        else:
            msg="Summarizing results…"
        st.write(f"**Agent**: {msg}")
        done=int(st.session_state.get("progress_done",0))
        total=max(1,int(st.session_state.get("progress_total",THINK_TICKS+3)))
        st.progress(min(done/total,1.0), text=f"Step {min(done,total)}/{total}")

    elif ph=="complete" and st.session_state["current_incident"]:
        iid=st.session_state["current_incident"]["id"]
        p=st.session_state["payloads"].get(str(iid)) or {}
        k_now=compute_kpis(st.session_state["case"], p)
        ref_payload=get_payload_by_label(ref_label) or st.session_state["payloads"]["0"]
        k_prev=compute_kpis(st.session_state["case"], ref_payload) if ref_payload else None
        st.write("**Agent**: " + headline_one_liner(p.get("title", f"Incident {iid}"), k_now, k_prev))
        st.markdown(kpi_explanation_md(k_now, p.get("title", f"Incident {iid}")))
        st.markdown("> " + (st.session_state["explanations"].get(str(iid)) or ""))

    else:
        st.caption("Ready. Initialize baseline, then start a scenario.")

# Baseline block
if st.session_state.get("base"):
    base=st.session_state["base"]
    st.subheader("Baseline")
    st.caption(f"Products: {', '.join(base.get('products', []))} | Nodes: {len(base.get('nodes', []))} | Flows: {len(base.get('flows', []))}")
    with st.container(border=True):
        st.markdown("**Network Map (Baseline)**")
        st_folium(st.session_state["maps"]["0"], width=None, height=520, key="map_base")
    show_flows_table(base.get("flows", []), caption="Baseline solved flows")

# Incident sections
for iid in (1,2,3):
    payload=st.session_state["payloads"].get(str(iid))
    if not payload:
        continue
    st.divider()
    st.subheader(payload.get("title", f"Incident {iid}"))
    before=payload.get("objective_before"); after=payload.get("objective_after")
    ref_payload=get_payload_by_label(ref_label) or st.session_state["payloads"]["0"]
    ref_cost=ref_payload.get("objective_after") if ref_payload else None
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total cost (before)", fmt(before))
    c2.metric("Total cost (after)", fmt(after), delta=safe_delta(after,before))
    c3.metric(f"Δ vs {ref_label}", safe_delta(after,ref_cost))
    c4.metric("Used slacks?", "Yes" if payload.get("used_slacks") else "No")
    if payload.get("lane"):
        l=payload["lane"]
        tag=f"{l.get('src','?')} → {l.get('dst','?')}"
        if "cost_per_unit" in l:
            tag+=f" @ {fmt(l['cost_per_unit'])}"
        st.caption(f"Lane changed: {tag}")
    if payload.get("customer"):
        st.caption(f"Customer impacted: {payload['customer']}")
    with st.container(border=True):
        st.markdown(f"**Network Map ({payload.get('title','Incident')})**")
        st_folium(st.session_state["maps"].get(str(iid)), width=None, height=520, key=f"map_i{iid}")
    show_flows_table(payload.get("flows_after", []), caption=f"Flows after {payload.get('title', f'Incident {iid}')}")

# Tick driver
if st.session_state["is_running"]:
    if st.session_state["next_tick_at"] and time.time() >= st.session_state["next_tick_at"]:
        advance_agent()
        if st.session_state["is_running"]:
            schedule_next_tick()
        st.experimental_rerun()
