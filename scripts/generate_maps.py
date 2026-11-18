"""generate_maps.py - Generate comprehensive interactive Folium maps for technical report & evidence."""
import sys
from pathlib import Path
import folium
from folium.plugins import MeasureControl
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from common import load_trip, RoadGraph
from features import generate_candidates, CandConfig
from align import align_to_route
from viterbi import ViterbiDecoder, HMMConfig

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")
OUT_MAPS = REPO_ROOT / "reports" / "maps"
OUT_MAPS.mkdir(parents=True, exist_ok=True)


def create_folium_map(trip_id: str = "00000000"):
    print(f"Generating comprehensive Folium map for Trip {trip_id}...")
    trip = load_trip(DATA_ROOT, "student_test", trip_id, with_route=True)
    g = RoadGraph(trip.nodes, trip.arcs)

    # Candidate generation & model scoring
    cands, _, obs = generate_candidates(trip, CandConfig(), graph=g)
    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    cands["score"] = artifact["pipeline"].predict_proba(cands[artifact["features"]])[:, 1]

    # Decode route with Viterbi
    decoder = ViterbiDecoder(HMMConfig())
    res = decoder.decode(obs, cands, g)
    reconstructed_arcs = res["reconstructed_route"]

    # Aligned reference route
    ref_df = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g, trip.route)

    # Center map on average coordinate
    avg_lat = float(trip.track.lat.mean())
    avg_lon = float(trip.track.lon.mean())

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # 1. Road Network Layer (subset of arcs near trajectory)
    road_layer = folium.FeatureGroup(name="1. Road Network Topology (Indexed Arcs)", show=False)
    # Collect unique arcs near track
    nearby_arcs = np.unique(cands.arc_id)
    for a in nearby_arcs[:1000]:  # limit to 1000 nearby arcs to keep file size responsive
        u, v = g.u[a], g.v[a]
        line = [[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])],
                [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]]
        folium.PolyLine(line, color="#9CA3AF", weight=1.5, opacity=0.5).add_to(road_layer)
    road_layer.add_to(m)

    # 2. Reference Route (Ground Truth)
    ref_layer = folium.FeatureGroup(name="2. Ground Truth Route (.route)", show=True)
    ref_coords = []
    for a in trip.route:
        u, v = g.u[a], g.v[a]
        ref_coords.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
        ref_coords.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
    folium.PolyLine(ref_coords, color="#10B981", weight=6, opacity=0.7, tooltip="Ground Truth Reference Route").add_to(ref_layer)
    ref_layer.add_to(m)

    # 3. Predicted Route (HMM / Viterbi Reconstructed)
    pred_layer = folium.FeatureGroup(name="3. Predicted Reconstructed Route (PathSnap)", show=True)
    pred_coords = []
    for a in reconstructed_arcs:
        u, v = g.u[a], g.v[a]
        pred_coords.append([float(trip.nodes[u, 1]), float(trip.nodes[u, 0])])
        pred_coords.append([float(trip.nodes[v, 1]), float(trip.nodes[v, 0])])
    folium.PolyLine(pred_coords, color="#2563EB", weight=3.5, opacity=0.9, tooltip="Predicted Traversed Route", dash_array="5, 5").add_to(pred_layer)
    pred_layer.add_to(m)

    # 4. Raw GPS Observations Layer
    gps_layer = folium.FeatureGroup(name="4. Raw GPS Observations (Noisy Fixes)", show=True)
    for i, row in trip.track.iterrows():
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=2.5,
            color="#EF4444",
            fill=True,
            fill_color="#EF4444",
            fill_opacity=0.6,
            popup=f"Fix #{i} t={row.t_s:.1f}s",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Start and End Markers
    if pred_coords:
        folium.Marker(
            location=pred_coords[0],
            popup="<b>Trajectory Origin</b>",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            location=pred_coords[-1],
            popup="<b>Trajectory Destination</b>",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.add_child(MeasureControl())

    out_file = OUT_MAPS / f"map_trip_{trip_id}.html"
    m.save(str(out_file))
    print(f"Folium map saved successfully: {out_file}")


if __name__ == "__main__":
    create_folium_map("00000000")
    create_folium_map("00000007")
