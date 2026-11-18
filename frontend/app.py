"""app.py - Streamlit Frontend for PathSnap Evaluator Demonstration.

Connects over HTTP to the FastAPI ML backend on Laptop 1.
Renders interactive Folium maps of GPS fixes and reconstructed routes,
performance metrics, timing breakdowns, and raw API response inspection.
"""
import sys
import time
from pathlib import Path
import folium
from folium.plugins import MeasureControl
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-err {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Backend Connection & Trip Selection
# ---------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/47/PES_University_logo.png", width=140)
st.sidebar.title("PathSnap Control")
st.sidebar.markdown("**Team 115** | PES University  \n*Distributed Route Reconstruction*")
st.sidebar.markdown("---")

api_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value="http://localhost:8000",
    help="Enter the IP and port of Laptop 1 (e.g., http://192.168.1.15:8000)"
).rstrip("/")

# Health check test
health_status = None
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2.0)
    if health_resp.status_code == 200:
        health_status = health_resp.json()
except Exception:
    health_status = None

if health_status:
    st.sidebar.markdown(f'<span class="status-badge-ok">● Connected ({health_status.get("status")})</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Model: {health_status.get('model_loaded')}")
else:
    st.sidebar.markdown('<span class="status-badge-err">● Disconnected</span>', unsafe_allow_html=True)
    st.sidebar.warning(f"Cannot connect to backend at `{api_url}`. Ensure Laptop 1 has FastAPI running on `0.0.0.0:8000`.")

st.sidebar.markdown("---")

# Query available trips from backend
available_trips = []
partition_name = "unknown"
if health_status:
    try:
        trips_resp = requests.get(f"{api_url}/trips", timeout=3.0)
        if trips_resp.status_code == 200:
            trips_data = trips_resp.json()
            available_trips = trips_data.get("trips", [])
            partition_name = trips_data.get("partition", "unknown")
    except Exception as e:
        st.sidebar.error(f"Error fetching trips: {e}")

if not available_trips:
    # Default fallback list for offline UI viewing
    available_trips = ["00000000", "00000007", "00000009", "00000040", "00000052",
                       "00000056", "00000065", "00000076", "00000077", "00000091"]

selected_trip = st.sidebar.selectbox(
    "Select Evaluation Trip",
    options=available_trips,
    index=0,
    help="Select from available evaluation trip IDs retrieved from Laptop 1 backend."
)

custom_override = st.sidebar.text_input("Or Enter Custom Trip ID", value="", max_chars=8)
if custom_override.strip():
    selected_trip = custom_override.strip()

st.sidebar.caption(f"Active Partition: `{partition_name}` ({len(available_trips)} trips)")
predict_btn = st.sidebar.button("🚀 Match & Reconstruct Route", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Deployment Architecture")
st.sidebar.info("""
**Laptop 1 (MacBook Air M1):**
- FastAPI Backend (`0.0.0.0:8000`)
- P1 Candidates & 13 Features
- P2 Trained ML Pipeline
- P3 HMM/Viterbi Sequential Decoder
- Confidential Server-Side Evaluation

**Laptop 2 (Evaluator UI):**
- Streamlit Web Interface
- Real HTTP REST requests
- Folium Interactive Maps
""")


# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UE24CS352A Machine Learning Project | End-to-End Trajectory Reconstruction System</div>', unsafe_allow_html=True)

# Store prediction results in session state to persist across interactions
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
    st.session_state.active_trip = None

if predict_btn:
    if not health_status:
        st.error(f"Cannot execute prediction: Backend API at `{api_url}` is unreachable. Please verify Laptop 1 network connection.")
    else:
        with st.spinner(f"Matching trajectory for Trip `{selected_trip}` across road network..."):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{api_url}/predict",
                    json={"trip_id": selected_trip},
                    timeout=30.0
                )
                req_latency = time.perf_counter() - t0

                if resp.status_code == 200:
                    st.session_state.prediction_data = resp.json()
                    st.session_state.active_trip = selected_trip
                    st.session_state.req_latency = req_latency
                    st.success(f"Trip `{selected_trip}` successfully matched in {req_latency:.2f} s!")
                elif resp.status_code == 404:
                    st.error(f"Trip `{selected_trip}` not found in any dataset partition on Laptop 1 backend.")
                else:
                    st.error(f"Prediction failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Network error communicating with Laptop 1: {e}")

# Render Results Dashboard if data is available
data = st.session_state.prediction_data

if data and st.session_state.active_trip:
    st.markdown(f"### Results for Trip `{st.session_state.active_trip}`")

    # Metrics Row
    metrics = data.get("metrics")
    timing = data.get("timing_breakdown", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        val = f"{metrics['top1_accuracy']*100:.2f}%" if metrics and metrics.get("top1_accuracy") is not None else "N/A"
        st.metric("Observation Top-1", val, delta="vs Aligned Reference" if metrics else None)

    with col2:
        val = f"{metrics['route_continuity']*100:.2f}%" if metrics and metrics.get("route_continuity") is not None else "N/A"
        st.metric("Route Continuity", val, delta="Road Topology" if metrics else None)

    with col3:
        val = f"{metrics['observation_macro_f1']*100:.2f}%" if metrics and metrics.get("observation_macro_f1") is not None else "N/A"
        st.metric("Observation Macro F1", val)

    with col4:
        val = f"{metrics['route_jaccard']*100:.2f}%" if metrics and metrics.get("route_jaccard") is not None else "N/A"
        st.metric("Route Overlap", val, delta="Ground Truth Match" if metrics else None)

    with col5:
        total_s = timing.get("total_inference_s", data.get("inference_time_seconds", 0.0))
        st.metric("Total Latency", f"{total_s:.3f} s", delta=f"{st.session_state.get('req_latency', 0.0):.3f}s HTTP")

    # Detailed Latency Breakdown
    with st.expander("⏱️ Subsystem Timing Breakdown", expanded=False):
        tb_cols = st.columns(4)
        tb_cols[0].metric("P1 Candidate & Feature Gen", f"{timing.get('feature_generation_s', 0)*1000:.1f} ms")
        tb_cols[1].metric("P2 Classifier Inference", f"{timing.get('model_scoring_s', 0)*1000:.1f} ms")
        tb_cols[2].metric("P3 HMM/Viterbi Decoder", f"{timing.get('viterbi_decoding_s', 0)*1000:.1f} ms")
        tb_cols[3].metric("Total Server Processing", f"{timing.get('total_inference_s', 0)*1000:.1f} ms")

    # Map and Route Inspection Tabs
    tab_map, tab_arcs, tab_network, tab_raw = st.tabs(["🗺️ Interactive Map", "🛣️ Reconstructed Arcs", "🌐 Two-Laptop Demo Guide", "📄 Raw API Response"])

    with tab_map:
        gps_coords = data.get("gps_coordinates", [])
        route_coords = data.get("route_coordinates", [])

        if not gps_coords and not route_coords:
            st.warning("No geographical coordinates returned for this trip.")
        else:
            # Center map on midpoint of GPS fixes
            center_lat = sum(p[0] for p in gps_coords) / len(gps_coords) if gps_coords else route_coords[0][0]
            center_lon = sum(p[1] for p in gps_coords) / len(gps_coords) if gps_coords else route_coords[0][1]

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles="OpenStreetMap",
            )

            # Layer 1: Raw GPS Fixes
            gps_group = folium.FeatureGroup(name="Raw GPS Observations (Noisy Fixes)")
            for i, (lat, lon) in enumerate(gps_coords):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color="#2563EB",
                    fill=True,
                    fill_color="#3B82F6",
                    fill_opacity=0.7,
                    popup=f"GPS Fix #{i} ({lat:.5f}, {lon:.5f})",
                ).add_to(gps_group)
            gps_group.add_to(m)

            # Layer 2: Reconstructed Road Route
            if route_coords:
                route_group = folium.FeatureGroup(name="Reconstructed Vehicle Route (Viterbi)")
                folium.PolyLine(
                    locations=route_coords,
                    color="#10B981",
                    weight=5,
                    opacity=0.85,
                    tooltip="Decoded Traversed Path",
                ).add_to(route_group)
                route_group.add_to(m)

                # Start and End Markers
                folium.Marker(
                    location=route_coords[0],
                    popup="<b>Route Start</b>",
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                ).add_to(m)

                folium.Marker(
                    location=route_coords[-1],
                    popup="<b>Route Destination</b>",
                    icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                ).add_to(m)

            folium.LayerControl(position="topright", collapsed=False).add_to(m)
            m.add_child(MeasureControl())

            # Render map in Streamlit
            st_folium(m, width="100%", height=560, returned_objects=[])

            st.caption("🟢 **Green Line:** Reconstructed vehicle route on road network | 🔵 **Blue Dots:** Noisy raw GPS observations | ⛳ **Pins:** Trajectory origin and destination")

    with tab_arcs:
        st.markdown(f"#### Traversed Road Segments ({len(data.get('predicted_route', []))} total arcs)")
        arcs = data.get("predicted_route", [])
        if arcs:
            df_arcs = pd.DataFrame({
                "Sequence Step": list(range(len(arcs))),
                "Traversed Arc ID": arcs,
            })
            st.dataframe(df_arcs, use_container_width=True, height=350)
        else:
            st.info("No road arcs decoded.")

    with tab_network:
        st.markdown("### Two-Laptop Deployment & Live Verification Instructions")
        st.markdown("""
        #### Architecture Overview
        ```
        +-----------------------------------+         Wi-Fi LAN HTTP REST         +-----------------------------------+
        |       Laptop 1 (MacBook Air)      | <=================================> |        Laptop 2 (Evaluator UI)    |
        |   FastAPI Service on 0.0.0.0:8000 |         GET /trips                  |      Streamlit on localhost:8501   |
        |   ML Models, HMM, Server Eval     |         POST /predict {"trip_id"}   |     Interactive Folium Visualizer |
        +-----------------------------------+                                     +-----------------------------------+
        ```

        #### Laptop 1 (MacBook Air - Server Host):
        1. Open Terminal and navigate to the project directory:
           ```bash
           cd ~/Downloads/115-Pathsnap
           source .venv/bin/activate
           uvicorn backend.main:app --host 0.0.0.0 --port 8000
           ```
        2. Find your local Wi-Fi IP address:
           ```bash
           ipconfig getifaddr en0
           # Example output: 192.168.1.15
           ```
        3. Keep the terminal visible to show real-time incoming request logs to evaluators.

        #### Laptop 2 (Client / Evaluator UI):
        1. Open browser to the Streamlit app.
        2. In the sidebar under **FastAPI Backend URL**, enter:
           `http://<Laptop_1_IP>:8000` (e.g. `http://192.168.1.15:8000`).
        3. Verify connection badge turns **Green (Connected)**.
        4. Select any evaluator trip and click **Match & Reconstruct Route**.

        #### Testing via Command Line (CMD / PowerShell on Laptop 2):
        ```bash
        # 1. Health check:
        curl -X GET http://<Laptop_1_IP>:8000/health

        # 2. Query available trips:
        curl -X GET http://<Laptop_1_IP>:8000/trips

        # 3. Predict route for Trip 00000000:
        curl -X POST http://<Laptop_1_IP>:8000/predict -H "Content-Type: application/json" -d '{"trip_id": "00000000"}'
        ```
        """)

    with tab_raw:
        st.markdown("#### Raw Server Response Payload")
        st.json(data)

else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'Match & Reconstruct Route'** to run the complete map-matching pipeline.")
    
    # Overview cards when no trip is active
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### System Architecture")
        st.markdown("""
        - **Person 1:** Spatial KDTree Candidate Generator & 13 Kinematic Features.
        - **Person 2:** Histogram Gradient Boosting with Bayesian Hyperparameter Optimization.
        - **Person 3:** Hidden Markov Model with Log-Space Viterbi Sequential Decoding.
        - **Person 4:** Distributed FastAPI Backend, Evaluator UI, and Folium Visualizer.
        """)
    with col_b:
        st.markdown("### Key Technical Highlights")
        st.markdown("""
        - **Inference Latency:** Sub-second per trip (< 0.87 s for > 2,500 fixes).
        - **Observation Accuracy:** **90.92%** Top-1 matching across test observations.
        - **Route Continuity:** **97.61%** physically continuous road network transitions.
        - **Privacy Guarantee:** Ground truth `.route` data remains strictly server-side.
        """)
