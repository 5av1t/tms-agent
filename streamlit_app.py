import streamlit as st
import pandas as pd
import time
import json
import random
import os
from typing import Dict, List, Optional
import folium
from streamlit_folium import st_folium

# Project imports
from solver import SupplyChainModel
from tools import get_available_tools, AVAILABLE_FUNCTIONS

# Optional Gemini
try:
    import google.generativeai as genai
    from google.generativeai.types import Part
except ImportError:
    genai = None
    Part = None # Define Part as None if import fails

# --- Page Config ---
st.set_page_config(
    page_title="Level 3 TMS Agent — Control Tower",
    layout="wide"
)

# --- Constants ---
DATA_PATH = "base_case.xlsx"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# --- Helper Functions ---
def fmt(x, none="—", places=2):
    if x is None: return none
    try:
        return f"{float(x):,.{places}f}"
    except (ValueError, TypeError):
        return none

# --- Agent Memory Class ---
class AgentMemory:
    """A simple session-based memory for the agent."""
    def __init__(self):
        if "agent_memory" not in st.session_state:
            st.session_state.agent_memory = []

    def add(self, event_type: str, action: dict, outcome: dict):
        """Adds a memory of an event, action, and its outcome."""
        st.session_state.agent_memory.append({
            "event": event_type,
            "action": action,
            "outcome": outcome
        })

    def get_recent_memories(self, n: int = 5) -> List[Dict]:
        """Retrieves the last N memories."""
        return st.session_state.agent_memory[-n:]

    def get_formatted_summary(self) -> str:
        """Returns a string summary of past experiences."""
        if not st.session_state.agent_memory:
            return "No past experiences in this session."
        summary = "Recent Past Experiences:\n"
        for mem in self.get_recent_memories():
            action_desc = mem['action'].get('label', 'N/A')
            cost_delta = mem['outcome'].get('cost_delta_pct', 0)
            fill_delta = mem['outcome'].get('fill_rate_delta_pp', 0)
            summary += f"- Event '{mem['event']}', took action '{action_desc}', "
            summary += f"resulted in cost change of {cost_delta:.1f}% and fill rate change of {fill_delta:.1f}pp.\n"
        return summary

# --- Event Simulator ---
class EventSimulator:
    """Generates random supply chain events."""
    def __init__(self, model: SupplyChainModel):
        self._model = model
        self.customers = [n['id'] for n in model.case['nodes'].to_dict(orient="records") if n['kind'] == 'customer']
        self.lanes = model.case['lanes'][['src', 'dst']].drop_duplicates().to_dict(orient="records")

    def generate_event(self) -> Optional[Dict]:
        """Returns a randomly generated event dictionary."""
        # Removed "none" options to ensure an event is always generated for the demo.
        event_type = random.choice(["demand_spike", "lane_disruption"])
        
        if event_type == "demand_spike" and self.customers:
            customer = random.choice(self.customers)
            pct_increase = random.uniform(0.2, 0.5)
            return {
                "type": "demand_spike",
                "customer": customer,
                "pct": pct_increase,
                "description": f"🚨 Demand Spike: Sudden {pct_increase:.0%} demand increase for {customer}."
            }
        
        if event_type == "lane_disruption" and self.lanes:
            lane = random.choice(self.lanes)
            capacity_reduction = random.uniform(0.5, 0.9)
            return {
                "type": "lane_cap",
                "src": lane['src'],
                "dst": lane['dst'],
                "new_capacity_factor": 1.0 - capacity_reduction,
                "description": f"🚧 Lane Disruption: Capacity on lane {lane['src']} -> {lane['dst']} reduced by {capacity_reduction:.0%}."
            }
        return None

# --- UI Components ---
def draw_map(nodes: List[Dict], flows: List[Dict], products: List[str]) -> folium.Map:
    # (Implementation is similar to the original, simplified for brevity)
    lat0, lon0 = (20.5937, 78.9629) # India
    if nodes:
        df = pd.DataFrame(nodes).dropna(subset=['lat', 'lon'])
        if not df.empty:
            lat0, lon0 = df['lat'].mean(), df['lon'].mean()
            
    m = folium.Map(location=[lat0, lon0], zoom_start=5, tiles="cartodbpositron")
    
    node_by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        if n.get("lat") is not None and n.get("lon") is not None:
            folium.CircleMarker(
                (n["lat"], n["lon"]), radius=5,
                popup=f"{n['id']} ({n['kind']})",
                color="#333", fill=True, fill_opacity=0.8
            ).add_to(m)

    color_map = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(products)}
    for f in flows:
        u, v = node_by_id.get(f["src"]), node_by_id.get(f["dst"])
        if u and v and u.get("lat") and v.get("lat"):
            folium.PolyLine(
                [(u["lat"], u["lon"]), (v["lat"], v.get("lon"))],
                color=color_map.get(f["product"], "#888"),
                weight=max(1, min(8, f["flow"] / 10000)),
                opacity=0.7
            ).add_to(m)
    return m

def display_kpis(kpis: dict, ref_kpis: Optional[dict] = None):
    c1, c2, c3 = st.columns(3)
    
    cost_delta = (kpis['cost'] - ref_kpis['cost']) if ref_kpis and kpis.get('cost') is not None and ref_kpis.get('cost') is not None else None
    c1.metric("Total Transport Cost", f"€{fmt(kpis.get('cost'), places=0)}", delta=f"€{fmt(cost_delta, places=0)}" if cost_delta is not None else None)

    fill_delta = (kpis['fill_rate'] - ref_kpis['fill_rate'])*100 if ref_kpis and kpis.get('fill_rate') is not None and ref_kpis.get('fill_rate') is not None else None
    c2.metric("Service Fill Rate", f"{kpis.get('fill_rate', 0):.2%}", delta=f"{fill_delta:.1f}pp" if fill_delta is not None else None)
    
    c3.metric("Lanes Used", len(kpis.get('flows', [])))

# --- Main Agent Class ---
class TMSAgent:
    def __init__(self, model: SupplyChainModel):
        self.model = model
        self.memory = AgentMemory()
        self.tools = get_available_tools()
        self.tool_functions = AVAILABLE_FUNCTIONS

        if 'gemini_model' not in st.session_state:
            # Prioritize key from user input in session state, then secrets, then environment
            api_key = st.session_state.get("google_api_key") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key and genai:
                try:
                    genai.configure(api_key=api_key)
                    st.session_state.gemini_model = genai.GenerativeModel(
                        'gemini-1.5-pro-latest',
                        system_instruction="You are a world-class supply chain analyst agent. Your goal is to manage a transportation network to minimize costs while maintaining a high service level. Use the provided tools to gather information and make informed decisions. Think step-by-step. First, analyze the situation. Second, use tools if necessary. Third, propose a concrete action from the given candidates. Respond in JSON format.",
                    )
                    st.session_state.is_llm_configured = True
                except Exception as e:
                    st.error(f"Failed to configure Gemini: {e}")
                    st.session_state.gemini_model = None
                    st.session_state.is_llm_configured = False
            else:
                st.session_state.gemini_model = None
                st.session_state.is_llm_configured = False
    
    def _calculate_kpis(self, result: dict) -> dict:
        if not result or not result.get("ok"):
            return {"cost": None, "fill_rate": 0, "flows": []}
        total_demand = self.model.case['demand']['demand'].sum()
        if total_demand == 0: total_demand = 1 # Avoid division by zero
        return {
            "cost": result["objective_cost"],
            "fill_rate": 1 - (result["total_shortage"] / total_demand),
            "flows": result["flows"]
        }

    def _get_llm_response(self, prompt: str, use_tools=True) -> Dict:
        """Gets a response from the Gemini model, optionally using tools."""
        if not st.session_state.get("is_llm_configured") or not Part:
            return {"decision": {"type": "no_op", "label": "No Action (Fallback)", "reason": "Fallback mode: Gemini AI not configured. Provide API key in sidebar."}}
        
        model = st.session_state.gemini_model
        generation_config = {"response_mime_type": "application/json"}
        response_text = "" # Initialize for error reporting
        
        try:
            # First call to the model
            initial_response = model.generate_content(
                prompt,
                tools=self.tools if use_tools else None,
                generation_config=generation_config
            )
            
            # Check for tool call
            if initial_response.candidates and initial_response.candidates[0].content.parts[0].function_call:
                fc = initial_response.candidates[0].content.parts[0].function_call
                tool_name = fc.name
                tool_args = dict(fc.args) # Convert to a standard dict
                st.info(f"🤖 Agent is using tool: `{tool_name}({json.dumps(tool_args)})`")
                
                # Execute the function
                function_to_call = self.tool_functions[tool_name]
                tool_response_content = function_to_call(**tool_args)
                
                # Send the response back to the model correctly
                final_response = model.generate_content(
                    [
                        initial_response.candidates[0].content, # Include the model's previous turn
                        Part.from_function_response(           # Add the tool result as a new part
                            name=tool_name,
                            response={
                                "content": tool_response_content,
                            }
                        )
                    ],
                    tools=self.tools,
                    generation_config=generation_config
                )
                response_text = final_response.text
            else:
                # No tool call, just use the first response
                response_text = initial_response.text

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            st.error(f"LLM Error: Failed to decode JSON response from model.")
            st.code(response_text, language="text") # Show the raw text for debugging
            return {"decision": {"type": "no_op", "reason": f"Error parsing LLM JSON response: {e}"}}
        except Exception as e:
            st.error(f"LLM Error: An unexpected error occurred: {e}")
            return {"decision": {"type": "no_op", "reason": f"Error during generation: {e}"}}

    def plan_and_act(self, event: Dict, current_state: Dict):
        """Main agent loop: Perceive, Plan, Act."""
        with st.status("🤖 Agent is processing the event...", expanded=True) as status:
            # 1. PERCEIVE: Analyze the current situation
            status.write("Step 1: Analyzing the event and current network state.")
            current_kpis = self._calculate_kpis(current_state)
            
            if not current_kpis.get("flows"):
                st.warning("Cannot plan action: current state has no flows to analyze.")
                status.update(label="⚠️ Analysis failed: No flows in current state.", state="error")
                return current_state, current_kpis

            top_flow = sorted(current_kpis['flows'], key=lambda x: x['flow'], reverse=True)[0]

            # 2. PLAN: Formulate candidates and prompt the LLM
            status.write("Step 2: Consulting with Gemini AI for a decision...")
            lane_cost_series = self.model.case['lanes'][(self.model.case['lanes']['src']==top_flow['src']) & (self.model.case['lanes']['dst']==top_flow['dst'])]['cost_per_unit']
            avg_lane_cost = lane_cost_series.mean() if not lane_cost_series.empty else 10.0

            candidates = [
                {"type": "no_op", "label": "Take no action and monitor."},
                {"type": "reroute_congested", "label": f"Reduce capacity on busiest lane ({top_flow['src']} -> {top_flow['dst']}) to force rerouting.", "shock": {"type": "lane_cap", "src": top_flow['src'], "dst": top_flow['dst'], "new_capacity": top_flow['flow'] * 0.5}},
                {"type": "add_express_lane", "label": f"Add a temporary express lane from {top_flow['src']} to {top_flow['dst']}.", "shock": {"type": "express_lane", "src": top_flow['src'], "dst": top_flow['dst'], "product": top_flow['product'], "capacity": top_flow['flow'], "cost_per_unit": 0.8 * avg_lane_cost}},
            ]

            prompt = f"""
            **Situation Analysis**
            An event has occurred: {event['description']}
            Current KPIs: Total Cost={fmt(current_kpis['cost'])}, Fill Rate={current_kpis.get('fill_rate', 0):.2%}
            Busiest Lane: {top_flow['src']} -> {top_flow['dst']} carrying {fmt(top_flow['flow'])} units.
            
            **Memory**
            {self.memory.get_formatted_summary()}

            **Task**
            Analyze the situation and the event. Use the available tools if you need more context (e.g., check weather for a demand spike location, or traffic on a disrupted lane). Then, decide on the best course of action from the candidates below. Your response must be a JSON object with a single key 'decision', which is one of the candidate objects. Add a 'reason' to your chosen decision.

            **Action Candidates:**
            {json.dumps(candidates, indent=2)}
            """
            st.code(prompt, language="markdown")
            
            response_json = self._get_llm_response(prompt)
            decision = response_json.get("decision", candidates[0])
            
            # Store the agent's reasoning for display in the main UI
            st.session_state.agent_narrative = decision.get("reason", "No detailed reasoning was provided by the agent.")

            status.write("Step 3: Gemini AI has made a decision.")
            # Displaying the raw JSON decision is still useful here for debugging/transparency
            st.write("##### Agent's Decision (JSON)")
            st.json(decision)

            # 3. ACT: Apply the decision and store memory
            status.write("Step 4: Applying decision and re-optimizing the network...")
            shock = decision.get("shock")
            new_state = self.model.solve(shock)
            new_kpis = self._calculate_kpis(new_state)

            status.write("Step 5: Storing the outcome in memory for future learning.")
            if current_kpis.get("cost") is not None and new_kpis.get("cost") is not None and current_kpis['cost'] > 0:
                 cost_delta_pct = (new_kpis['cost'] - current_kpis['cost']) / current_kpis['cost'] * 100
            else:
                 cost_delta_pct = 0

            outcome = {
                "cost_delta_pct": cost_delta_pct,
                "fill_rate_delta_pp": (new_kpis.get('fill_rate',0) - current_kpis.get('fill_rate',0)) * 100,
            }
            self.memory.add(event['type'], decision, outcome)
            
            status.update(label="✅ Agent action complete!", state="complete", expanded=False)
        
        return new_state, new_kpis

# --- Main App Logic ---
def main():
    st.title("Level 3 Agentic TMS Control Tower")
    st.caption("A proactive, tool-using, learning agent for supply chain management.")

    # --- Initialization ---
    if 'model' not in st.session_state:
        try:
            st.session_state.model = SupplyChainModel(DATA_PATH)
        except FileNotFoundError as e:
            st.error(f"Fatal Error: {e}. Please upload 'base_case.xlsx'.")
            uploaded_file = st.file_uploader("Upload base_case.xlsx", type=["xlsx"])
            if uploaded_file:
                st.session_state.model = SupplyChainModel(uploaded_file)
                st.rerun()
            return
            
    if 'agent' not in st.session_state:
        st.session_state.agent = TMSAgent(st.session_state.model)

    if 'baseline' not in st.session_state:
        with st.spinner("Calculating baseline optimization..."):
            st.session_state.baseline = st.session_state.model.solve()
            st.session_state.baseline_kpis = st.session_state.agent._calculate_kpis(st.session_state.baseline)
            st.session_state.current_state = st.session_state.baseline
            st.session_state.current_kpis = st.session_state.baseline_kpis
            st.session_state.agent_narrative = "The network baseline has been established. The agent is now monitoring for events."


    model = st.session_state.model
    agent = st.session_state.agent
    event_sim = EventSimulator(model)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Agent Configuration")
        api_key_input = st.text_input("Enter your Google API Key", type="password", key="api_key_input")
        if st.button("Configure Agent"):
            if api_key_input:
                st.session_state.google_api_key = api_key_input
                # Force re-initialization of the agent and its model
                if 'agent' in st.session_state:
                    del st.session_state['agent']
                if 'gemini_model' in st.session_state:
                    del st.session_state['gemini_model']
                st.success("API Key set! Agent will now use Gemini.")
                st.rerun()
            else:
                st.warning("Please enter a valid API key.")

        st.header("Live Simulation Control")
        if st.button("Simulate Next Event", type="primary"):
            st.session_state.last_event = event_sim.generate_event()
            # If the event is "none", we just log it and do nothing else.
            if st.session_state.last_event:
                new_state, new_kpis = agent.plan_and_act(st.session_state.last_event, st.session_state.current_state)
                st.session_state.current_state = new_state
                st.session_state.current_kpis = new_kpis
            else:
                # This case will no longer be reached with the updated code, but is safe to keep.
                st.toast("Network stable, no new events generated this tick.")


        st.divider()
        st.header("Agent Memory")
        st.text_area("Recent Experiences", agent.memory.get_formatted_summary(), height=200, disabled=True)

    # --- Main Display ---
    st.header("Network Status")

    # Add a persistent warning if the LLM is not configured
    if not st.session_state.get("is_llm_configured"):
        st.warning("Warning: Gemini AI is not configured. The agent is running in a deterministic fallback mode. Please enter your Google API key in the sidebar to enable intelligent decision-making.", icon="⚠️")

    if 'last_event' in st.session_state and st.session_state.last_event:
        st.info(st.session_state.last_event['description'])
    else:
        st.info("Network is stable. Simulate an event from the sidebar.")

    # New section for Agent's textual analysis
    st.header("Agent's Analysis")
    if 'agent_narrative' in st.session_state and st.session_state.agent_narrative:
        st.markdown(f"> {st.session_state.agent_narrative}")
    else:
        st.caption("Awaiting the first event to generate an analysis...")

    # KPI Display
    st.header("KPI Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline KPIs")
        display_kpis(st.session_state.baseline_kpis)
    with col2:
        st.subheader("Current KPIs")
        display_kpis(st.session_state.current_kpis, st.session_state.baseline_kpis)
    
    # Map and Data Display
    st.header("Network Flow Visualization")
    map_data = st.session_state.current_state
    if map_data and map_data.get('ok'):
        folium_map = draw_map(map_data['nodes'], map_data['flows'], map_data['products'])
        st_folium(folium_map, width=None, height=500)
        
        with st.expander("Show Current Flow Data"):
            st.dataframe(pd.DataFrame(map_data.get('flows', [])))
    else:
        st.warning("Current state is not optimal. Cannot display map or flows.")


if __name__ == "__main__":
    main()

