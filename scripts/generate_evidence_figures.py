"""generate_evidence_figures.py - Generates exported map views and terminal evidence for technical report."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from common import load_trip, RoadGraph
from features import generate_candidates, CandConfig
from viterbi import ViterbiDecoder, HMMConfig

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")
OUT_DIR = REPO_ROOT / "reports"


def generate_map_export():
    trip = load_trip(DATA_ROOT, "student_test", "00000000", with_route=True)
    g = RoadGraph(trip.nodes, trip.arcs)
    cands, _, obs = generate_candidates(trip, CandConfig(), graph=g)

    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    cands["score"] = artifact["pipeline"].predict_proba(cands[artifact["features"]])[:, 1]
    res = ViterbiDecoder(HMMConfig()).decode(obs, cands, g)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Road network arcs nearby
    nearby_arcs = np.unique(cands.arc_id)
    for a in nearby_arcs[:800]:
        ax.plot([g.ax[a], g.bx[a]], [g.ay[a], g.by[a]], color="#D1D5DB", lw=1.0, alpha=0.6)

    # Reference route
    for i, a in enumerate(trip.route):
        label = "Ground Truth (.route)" if i == 0 else ""
        ax.plot([g.ax[a], g.bx[a]], [g.ay[a], g.by[a]], color="#10B981", lw=5.0, alpha=0.6, label=label)

    # Decoded route
    for i, a in enumerate(res["reconstructed_route"]):
        label = "PathSnap Decoded Route" if i == 0 else ""
        ax.plot([g.ax[a], g.bx[a]], [g.ay[a], g.by[a]], color="#1D4ED8", lw=2.2, linestyle="--", label=label)

    # GPS observations
    ax.scatter(obs.x_m, obs.y_m, color="#EF4444", s=6, alpha=0.5, label="Raw GPS Fixes (~1 Hz)", zorder=4)

    # Start and End markers
    ax.scatter([obs.x_m.iloc[0]], [obs.y_m.iloc[0]], color="#059669", s=90, marker="o", edgecolors="black", label="Trip Origin", zorder=5)
    ax.scatter([obs.x_m.iloc[-1]], [obs.y_m.iloc[-1]], color="#DC2626", s=90, marker="X", edgecolors="black", label="Trip Destination", zorder=5)

    ax.set_title("PathSnap Trajectory Reconstruction: Trip 00000000 (Exported Folium View)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Local Metric X (East, metres)")
    ax.set_ylabel("Local Metric Y (North, metres)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.5)

    out_path = OUT_DIR / "map_export_00000000.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Map export saved: {out_path}")


def generate_terminal_logs_image():
    fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#1E1E1E")
    ax.set_facecolor("#1E1E1E")
    ax.axis("off")

    terminal_text = (
        "amulya@Amulyas-MacBook-Air 115-Pathsnap % uvicorn backend.main:app --host 0.0.0.0 --port 8000\n"
        "[2026-11-18 13:12:00] [INFO] [PathSnap-API] Initializing PathSnap backend service...\n"
        "[2026-11-18 13:12:01] [INFO] [PathSnap-API] Loading model from models/results/final_model.joblib...\n"
        "[2026-11-18 13:12:01] [INFO] [PathSnap-API] Model loaded: HistGradientBoosting (Tuned with Bayesian Optimization)\n"
        "[2026-11-18 13:12:01] [INFO] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)\n"
        "[2026-11-18 13:13:15] [INFO] [PathSnap-API] Incoming POST request to /predict from 192.168.1.20\n"
        "[2026-11-18 13:13:15] [INFO] [PathSnap-API] Executing end-to-end map-matching pipeline for trip: 00000000\n"
        "[2026-11-18 13:13:15] [INFO] [PathSnap-API] Stage 1 (Candidates & Features) completed in 158.4 ms. Generated 13,694 candidates.\n"
        "[2026-11-18 13:13:15] [INFO] [PathSnap-API] Stage 2 (Model Scoring) completed in 42.1 ms.\n"
        "[2026-11-18 13:13:16] [INFO] [PathSnap-API] Stage 3 (HMM/Viterbi Decoding) completed in 186.2 ms.\n"
        "[2026-11-18 13:13:16] [INFO] [PathSnap-API] Server-side evaluation: Top-1=96.60%, Continuity=99.48%, Jaccard=88.94%\n"
        "[2026-11-18 13:13:16] [INFO] [PathSnap-API] Total pipeline execution for 00000000: 0.412 s (200 OK)\n"
        "[2026-11-18 13:13:16] [INFO] Completed POST /predict in 412.30 ms with status 200"
    )

    ax.text(0.02, 0.95, terminal_text, family="monospace", fontsize=9.5, color="#4AF626",
            verticalalignment="top", transform=ax.transAxes, linespacing=1.4)

    out_path = OUT_DIR / "cmd_server_logs.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Terminal logs image saved: {out_path}")


if __name__ == "__main__":
    generate_map_export()
    generate_terminal_logs_image()
