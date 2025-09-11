import os
import json
import time
import math
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple

# Maps
import folium
from streamlit_folium import st_folium

# Optimizer (LP)
from scipy.optimize import linprog

# Gemini (optional)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
PAGE_TITLE = "TMS Agent — Control Tower (Gemini)"
DATA_PATH = os.path.join("data", "base_case.xlsx")
BASELINE_WAIT_SECS = 0   # no countdown; keep demo snappy
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
if not os.path.exists(DATA_PATH):
    st.error(f"Data file missing: {DATA_PATH}. Please ensure base_case.xlsx is in the repo under /data/")
    st.stop()
# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
def ensure_state():
    ss = st.session_state
    ss.setdefault("boot", None)
    ss.setdefault("phase", "idle")  # idle, baseline, incident1, incident2, incident3
    ss.setdefault("payloads", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("maps", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("explanations", {"0": None, "1": None, "2": None, "3": None})
    ss.setdefault("logs", {"1": [], "2": [], "3": []})

ensure_state()

# ──────────────────────────────────────────────────────────────────────────────
# Data & Optimizer
# ──────────────────────────────────────────────────────────────────────────────
def read_excel_case(path: str) -> dict:
    xls = pd.ExcelFile(path)
    case = {}
    case["nodes"] = pd.read_excel(xls, "Nodes").fillna("")
    case["demand"] = pd.read_excel(xls, "Demand").fillna(0)
    case["supply"] = pd.read_excel(xls, "Supply").fillna(0)
    case["lanes"]  = pd.read_excel(xls, "Lanes").fillna(0)
    return case

def solve_min_cost_flow(case: dict, shock: dict | None = None) -> dict:
    """
    Balanced min-cost multi-commodity flow with slack (shortage/disposal) penalties to ensure demo feasibility.
    Sheets expected:
      Nodes(id, kind=supplier/dc/customer, lat, lon)
      Demand(customer, product, demand)
      Supply(supplier, product, supply)
      Lanes(src, dst, product, capacity, cost_per_unit)
    shock: lane_cap / demand_spike / express_lane (see incidents below)
    """
    nodes = case["nodes"].copy()
    demand_df = case["demand"].copy()
    supply_df = case["supply"].copy()
    lanes_df  = case["lanes"].copy()

    # Normalize column names
    for df, col_map in [
        (demand_df, {"Demand": "demand"}),
        (supply_df, {"Supply": "supply"}),
    ]:
        for a,b in col_map.items():
            if a in df.columns and b not in df.columns:
                df[b] = df[a]

    products = sorted(set(demand_df["product"].unique()) | set(supply_df["product"].unique()))

    # Apply shock
    if shock:
        t = shock.get("type")
        if t == "lane_cap":
            cond = (lanes_df["src"] == shock["src"]) & (lanes_df["dst"] == shock["dst"]) & (lanes_df["product"] == shock.get("product", lanes_df["product"]))
            lanes_df.loc[cond, "capacity"] = shock["new_capacity"]
        elif t == "demand_spike":
            cond = (demand_df["customer"] == shock["customer"])
            demand_df.loc[cond, "demand"] = demand_df.loc[cond, "demand"] * (1.0 + shock["pct"])
        elif t == "express_lane":
            row = {
                "src": shock["src"], "dst": shock["dst"], "product": shock["product"],
                "capacity": shock["capacity"], "cost_per_unit": shock["cost_per_unit"]
            }
            lanes_df = pd.concat([lanes_df, pd.DataFrame([row])], ignore_index=True)

    # Build variables x[(src,dst,product)]
    lanes = lanes_df[["src","dst","product","capacity","cost_per_unit"]].to_dict(orient="records")
    var_keys = [(l["src"], l["dst"], l["product"]) for l in lanes]
    vidx = {k:i for i,k in enumerate(var_keys)}
    n_x = len(var_keys)

    # Slack vars: disposal per supplier/product, shortage per customer/product
    cust_prod = sorted({(r["customer"], r["product"]) for _, r in demand_df.iterrows()})
    supp_prod = sorted({(r["supplier"], r["product"]) for _, r in supply_df.iterrows()})
    sp_index = {k: n_x + i for i, k in enumerate(supp_prod)}           # disposal
    sh_index = {k: n_x + len(supp_prod) + i for i, k in enumerate(cust_prod)}  # shortage
    n_vars = n_x + len(sp_index) + len(sh_index)

    # Objective
    BIGM = 1e6
    c = np.zeros(n_vars)
    for (src,dst,pr), i in vidx.items():
        row = next(r for r in lanes if r["src"]==src and r["dst"]==dst and r["product"]==pr)
        c[i] = float(row["cost_per_unit"])
    for k,i in sp_index.items():
        c[i] = BIGM
    for k,i in sh_index.items():
        c[i] = BIGM

    # Constraints
    A_eq = []; b_eq = []
    # supply balance: outflow + disposal = supply
    for s,p in supp_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if src == s and pr == p:
                row[i] = 1.0
        row[sp_index[(s,p)]] = 1.0
        sup_val = float(supply_df[(supply_df["supplier"]==s) & (supply_df["product"]==p)]["supply"].sum())
        A_eq.append(row); b_eq.append(sup_val)
    # demand balance: inflow + shortage = demand
    for cst,p in cust_prod:
        row = np.zeros(n_vars)
        for (src,dst,pr), i in vidx.items():
            if dst == cst and pr == p:
                row[i] = 1.0
        row[sh_index[(cst,p)]] = 1.0
        dem_val = float(demand_df[(demand_df["customer"]==cst) & (demand_df["product"]==p)]["demand"].sum())
        A_eq.append(row); b_eq.append(dem_val)

    # capacities: x <= cap
    A_ub = []; b_ub = []
    for (src,dst,pr), i in vidx.items():
        row = np.zeros(n_vars)
        row[i] = 1.0
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
        used_slacks = True  # extreme edge case

    nrecs = []
    for _, r in nodes.iterrows():
        nrecs.append({
            "id": str(r.get("id") or r.get("name") or r.get("node")),
            "kind": str(r.get("kind") or "").lower(),
            "lat": float(r.get("lat")) if pd.notna(r.get("lat")) else None,
            "lon": float(r.get("lon")) if pd.notna(r.get("lon")) else None
        })

    products_list = products
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
# Incidents (deterministic)
# ──────────────────────────────────────────────────────────────────────────────
def build_baseline() -> dict:
    case = read_excel_case(DATA_PATH)
    base = solve_min_cost_flow(case, None)
    return base

def incident_1(prev: dict) -> dict:
    case = read_excel_case(DATA_PATH)
    if not prev.get("flows"):
        return {"title":"Incident 1","objective_before":prev.get("objective_cost"),"objective_after":prev.get("objective_cost"),"flows_after":prev.get("flows",[])}
    biggest = sorted(prev["flows"], key=lambda x: x["flow"], reverse=True)[0]
    shock = {"type": "lane_cap", "src": biggest["src"], "dst": biggest["dst"], "product": biggest["product"], "new_capacity": max(0.5*biggest["flow"], 1.0)}
    before = prev["objective_cost"]
    aft = solve_min_cost_flow(case, shock)
    return {
        "title": "Incident 1 — Corridor capacity restriction",
        "lane": {"src": shock["src"], "dst": shock["dst"]},
        "objective_before": before,
        "objective_after": aft["objective_cost"],
        "used_slacks": aft["used_slacks"],
        "total_shortage": aft["total_shortage"],
        "total_disposal": aft["total_disposal"],
        "flows_after": aft["flows"],
        "currency": aft["currency"], "flow_unit": aft["flow_unit"]
    }

def incident_2(prev: dict) -> dict:
    case = read_excel_case(DATA_PATH)
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

def incident_3(prev: dict) -> dict:
    case = read_excel_case(DATA_PATH)
    if not prev.get("flows"):
        return {"title":"Incident 3","objective_before":prev.get("objective_cost"),"objective_after":prev.get("objective_cost"),"flows_after":prev.get("flows",[])}
    top = sorted(prev["flows"], key=lambda x: x["flow"], reverse=True)[0]
    src, dst, pr = top["src"], top["dst"], top["product"]
    # add cheaper express lane
    avg_cost = 0.0
    # try to get any lane cost for same (src,dst,pr) from excel lanes
    # if not present, choose a small number
    avg_cost = 0.75  # safe demo default
    shock = {"type": "express_lane", "src": src, "dst": dst, "product": pr, "capacity": max(top["flow"]*0.5, 1.0), "cost_per_unit": avg_cost}
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
def _product_color_map(products: List[str]) -> Dict[str,str]:
    return {p: PALETTE[i % len(PALETTE)] for i,p in enumerate(sorted(products))}

def _node_lookup(nodes: List[Dict]) -> Dict[str,Dict]:
    return {n["id"]: n for n in nodes}

def _center_latlon(nodes: List[Dict]) -> Tuple[float,float]:
    lats = [n.get("lat") for n in nodes if n.get("lat") is not None]
    lons = [n.get("lon") for n in nodes if n.get("lon") is not None]
    if not lats or not lons: return (20.5937, 78.9629)
    return (sum(lats)/len(lats), sum(lons)/len(lons))

def _flow_weight_scaler(flows: List[Dict], min_w=0.6, max_w=3.0):
    vals = sorted(float(f.get("flow", 0.0)) for f in flows if f.get("flow", 0.0) > 0)
    if not vals: return lambda x: min_w
    p90 = vals[min(int(0.9*(len(vals)-1)), len(vals)-1)]
    p90 = p90 if p90>0 else (max(vals) if vals else 1.0)
    def to_w(q):
        if q <= 0: return 0
        w = min_w + (max_w - min_w) * (q/(p90*1.0))
        return max(min_w, min(max_w, w))
    return to_w

def build_map(nodes: List[Dict], flows: List[Dict], products: List[str]) -> folium.Map:
    node_by_id = _node_lookup(nodes)
    lat0, lon0 = _center_latlon(nodes)
    m = folium.Map(location=[lat0,lon0], zoom_start=5, tiles="cartodbpositron")
    # nodes
    for n in nodes:
        if n.get("lat") is None or n.get("lon") is None: continue
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
        if u not in node_by_id or v not in node_by_id: continue
        urec, vrec = node_by_id[u], node_by_id[v]
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
# Gemini (guardrailed) — optional, falls back if unavailable
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

        model = genai.GenerativeModel(model_name,
                                      system_instruction=SYSTEM_PROMPT)
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
            # guardrail
            return deterministic_explanation(idx, payload)
        return text if text else deterministic_explanation(idx, payload)
    except Exception:
        return deterministic_explanation(idx, payload)

def ensure_explanation(idx: int, payload: dict):
    key = str(idx)
    if st.session_state["explanations"].get(key):  # cache
        return
    st.session_state["explanations"][key] = gemini_explain(idx, payload)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

with st.sidebar:
    st.subheader("Demo Controls")
    st.caption("Gemini settings (used for explanations; falls back if key missing).")
    st.session_state["gem_model"] = st.text_input("Gemini model", value=st.session_state.get("gem_model","gemini-1.5-pro"))
    st.session_state["gem_temperature"] = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("gem_temperature",0.1)), 0.05)
    st.session_state["gem_max_tokens"] = st.number_input("Max tokens", 64, 2048, int(st.session_state.get("gem_max_tokens",300)), 32)

    st.divider()
    st.caption("Run Incidents")
    if st.button("▶️ Incident 2 (demand surge)"):
        base_or_last = st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        resp2 = incident_2(base_or_last)
        st.session_state["payloads"]["2"] = resp2
        st.session_state["phase"] = "incident2"
        st.session_state["maps"]["2"] = build_map(st.session_state["boot"]["nodes"], resp2.get("flows_after",[]), st.session_state["boot"]["products"])
        ensure_explanation(2, resp2)
        st.toast("Incident 2 executed", icon="🧮")

    if st.button("▶️ Incident 3 (strategic express lane)"):
        base_or_last = st.session_state["payloads"]["2"] or st.session_state["payloads"]["1"] or st.session_state["payloads"]["0"]
        resp3 = incident_3(base_or_last if base_or_last else st.session_state["boot"])
        st.session_state["payloads"]["3"] = resp3
        st.session_state["phase"] = "incident3"
        st.session_state["maps"]["3"] = build_map(st.session_state["boot"]["nodes"], resp3.get("flows_after",[]), st.session_state["boot"]["products"])
        ensure_explanation(3, resp3)
        st.toast("Incident 3 executed", icon="🧭")

st.divider()
agent = st.container()
with agent:
    st.subheader("Agent Narrative")

st.divider()
col1, col2 = st.columns([1,1])
with col1:
    if st.button("🚀 Bootstrap baseline (Excel → optimize)"):
        base = build_baseline()
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
        st.toast("Baseline solved", icon="✅")

with col2:
    if st.button("🚨 Run Incident 1 now"):
        if not st.session_state.get("boot"):
            st.warning("Run baseline first.")
        else:
            resp1 = incident_1(st.session_state["payloads"]["0"])
            st.session_state["payloads"]["1"] = resp1
            st.session_state["phase"] = "incident1"
            st.session_state["maps"]["1"] = build_map(st.session_state["boot"]["nodes"], resp1.get("flows_after",[]), st.session_state["boot"]["products"])
            ensure_explanation(1, resp1)
            st.toast("Incident 1 executed", icon="🚨")

# Render sections
def show_kpis(before: float, after: float):
    delta = after - before
    c1, c2, c3 = st.columns(3)
    c1.metric("Objective (before)", f"{before:,.2f}")
    c2.metric("Objective (after)", f"{after:,.2f}", delta=f"{delta:+,.2f}")
    c3.metric("Δ (after-before)", f"{delta:+,.2f}")

def flows_table(flows: list[dict], label: str):
    show_flows_table(flows, caption=label)

# Agent narrative content
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
    flows_table(base.get("flows", []), "Baseline solved flows")

# Incident 1
if st.session_state["payloads"]["1"]:
    p1 = st.session_state["payloads"]["1"]
    st.divider()
    st.subheader("Incident 1 — Corridor capacity restriction")
    show_kpis(float(p1.get("objective_before", 0.0)), float(p1.get("objective_after", 0.0)))
    if p1.get("lane"):
        st.caption(f"Lane impacted: {p1['lane']['src']} → {p1['lane']['dst']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 1)**")
        st_folium(st.session_state["maps"]["1"], width=None, height=520, key="map_i1")
    flows_table(p1.get("flows_after", []), "Flows after Incident 1")

# Incident 2
if st.session_state["payloads"]["2"]:
    p2 = st.session_state["payloads"]["2"]
    st.divider()
    st.subheader("Incident 2 — Demand surge (+25%)")
    show_kpis(float(p2.get("objective_before", 0.0)), float(p2.get("objective_after", 0.0)))
    if p2.get("customer"): st.caption(f"Customer impacted: {p2['customer']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 2)**")
        st_folium(st.session_state["maps"]["2"], width=None, height=520, key="map_i2")
    flows_table(p2.get("flows_after", []), "Flows after Incident 2")

# Incident 3
if st.session_state["payloads"]["3"]:
    p3 = st.session_state["payloads"]["3"]
    st.divider()
    st.subheader("Incident 3 — Strategic express lane")
    show_kpis(float(p3.get("objective_before", 0.0)), float(p3.get("objective_after", 0.0)))
    if p3.get("lane"):
        l = p3["lane"]
        if "cost_per_unit" in l:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']} @ {l['cost_per_unit']}")
        else:
            st.caption(f"Candidate lane: {l['src']} → {l['dst']}")
    with st.container(border=True):
        st.markdown("**Network Map (Incident 3)**")
        st_folium(st.session_state["maps"]["3"], width=None, height=520, key="map_i3")
    flows_table(p3.get("flows_after", []), "Flows after Incident 3")
