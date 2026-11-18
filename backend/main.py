"""main.py - FastAPI Backend Service for PathSnap Map-Matching and Route Reconstruction.

Hosts the trained classifier pipeline, HMM/Viterbi sequential decoder, road network topology,
and server-side evaluation. Binds to 0.0.0.0:8000 for distributed two-laptop communication.
"""
from __future__ import annotations
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import f1_score

# Add backend, data_processing, models, and hmm to path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from common import load_trip, RoadGraph, list_trips, TripFormatError
from features import generate_candidates, CandConfig
from align import align_to_route
from viterbi import ViterbiDecoder, HMMConfig
from schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionMetrics,
    TimingBreakdown,
    HealthResponse,
    TripsResponse,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-API] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap")

# ---------------------------------------------------------------------------
# Data and Model Initialization
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "/Users/amulya/Downloads/Student folder"))
MODEL_PATH = REPO_ROOT / "models" / "results" / "final_model.joblib"

logger.info(f"Initializing PathSnap backend. Repository root: {REPO_ROOT}")
logger.info(f"Target data directory: {DEFAULT_DATA_ROOT}")

if not MODEL_PATH.exists():
    logger.error(f"FATAL: Final model artifact not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

logger.info(f"Loading trained model pipeline from {MODEL_PATH}...")
model_artifact = joblib.load(MODEL_PATH)
model_pipeline = model_artifact["pipeline"]
model_features = model_artifact["features"]
logger.info(f"Model pipeline successfully loaded: {model_pipeline.named_steps}")
logger.info(f"Required features ({len(model_features)}): {model_features}")

# Initialize HMM / Viterbi Decoder
hmm_decoder = ViterbiDecoder(HMMConfig())

# ---------------------------------------------------------------------------
# FastAPI App Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction Backend Service",
    version="1.0.0",
)

# Allow CORS for two-laptop communication from any browser or frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method
    logger.info(f"Incoming {method} request to {path} from {client_host}")

    response = await call_next(request)
    duration = time.perf_counter() - start_time
    logger.info(f"Completed {method} {path} in {duration*1000:.2f} ms with status {response.status_code}")
    return response


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def resolve_trip_partition(data_root: Path, trip_id: str) -> Tuple[str, Path]:
    """Find which partition contains the requested trip_id."""
    for partition in ["evaluator_test", "evaluator", "student_test", "train"]:
        trip_dir = data_root / partition / trip_id
        if trip_dir.is_dir() and (trip_dir / f"{trip_id}.track").exists():
            return partition, trip_dir
    raise FileNotFoundError(f"Trip ID '{trip_id}' not found in any dataset partition under {data_root}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "PathSnap Distributed Map-Matching API",
        "team": "Team 115",
        "institution": "PES University",
        "course": "UE24CS352A Machine Learning",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "trips": "/trips",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok",
        service="PathSnap API",
        version="1.0.0",
        model_loaded="HistGradientBoosting (Tuned with Bayesian Optimization)",
        features_count=len(model_features),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/trips", response_model=TripsResponse, tags=["Trips"])
def get_trips():
    """Returns available trips for evaluation.
    Prioritizes evaluator partition if present, otherwise returns student_test trips.
    """
    data_root = DEFAULT_DATA_ROOT
    if not data_root.exists():
        logger.warning(f"Data root {data_root} does not exist.")
        return TripsResponse(trips=[], partition="none", total_trips=0, message="Data root directory not found.")

    for candidate_partition in ["evaluator_test", "evaluator"]:
        p_dir = data_root / candidate_partition
        if p_dir.is_dir():
            trips = sorted([d.name for d in p_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()])
            if trips:
                logger.info(f"Serving {len(trips)} evaluator trips from {candidate_partition}")
                return TripsResponse(
                    trips=trips,
                    partition=candidate_partition,
                    total_trips=len(trips),
                    message=f"Loaded {len(trips)} official evaluator testing trips.",
                )

    # Fallback to permitted development student_test trips
    student_dir = data_root / "student_test"
    if student_dir.is_dir():
        trips = sorted([d.name for d in student_dir.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists()])
        logger.info(f"Serving {len(trips)} student_test trips")
        return TripsResponse(
            trips=trips,
            partition="student_test",
            total_trips=len(trips),
            message=f"Loaded {len(trips)} permitted development student-test trips.",
        )

    return TripsResponse(trips=[], partition="none", total_trips=0, message="No trips found in dataset root.")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    trip_id = request.trip_id.strip()
    logger.info(f"Executing end-to-end map-matching pipeline for trip: {trip_id}")
    t_start = time.perf_counter()

    # 1. Locate trip partition
    try:
        partition, trip_dir = resolve_trip_partition(DEFAULT_DATA_ROOT, trip_id)
        logger.info(f"Trip {trip_id} found in partition: '{partition}'")
    except FileNotFoundError as e:
        logger.error(f"Trip lookup failed: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # 2. Load Trip Data (Inference-safe: with_route=False)
    # The .route file is NEVER loaded for the prediction pipeline!
    try:
        trip = load_trip(DEFAULT_DATA_ROOT, partition, trip_id, with_route=False)
    except Exception as e:
        logger.error(f"Failed to load raw trip files for {trip_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Data loading error: {e}")

    # Build road graph
    g = RoadGraph(trip.nodes, trip.arcs)

    # 3. Person 1: Candidate Generation and 13 Features
    t_p1 = time.perf_counter()
    try:
        cand_cfg = CandConfig()
        cands, _, obs = generate_candidates(trip, cand_cfg, graph=g)
    except Exception as e:
        logger.error(f"Person 1 feature extraction failed for {trip_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Feature extraction error: {e}")
    time_p1 = time.perf_counter() - t_p1
    logger.info(f"Stage 1 (Candidates & Features) completed in {time_p1*1000:.1f} ms. Generated {len(cands)} candidates for {len(obs)} fixes.")

    # 4. Person 2: Machine Learning Model Scoring
    t_p2 = time.perf_counter()
    try:
        if len(cands) > 0:
            scores = model_pipeline.predict_proba(cands[model_features])[:, 1]
            cands["score"] = scores
        else:
            cands["score"] = pd.Series(dtype=float)
    except Exception as e:
        logger.error(f"Person 2 model inference failed for {trip_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Model inference error: {e}")
    time_p2 = time.perf_counter() - t_p2
    logger.info(f"Stage 2 (Model Scoring) completed in {time_p2*1000:.1f} ms.")

    # 5. Person 3: HMM / Viterbi Sequential Decoding & Route Reconstruction
    t_p3 = time.perf_counter()
    try:
        decode_result = hmm_decoder.decode(obs, cands, g)
    except Exception as e:
        logger.error(f"Person 3 HMM/Viterbi decoding failed for {trip_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Viterbi decoder error: {e}")
    time_p3 = time.perf_counter() - t_p3
    logger.info(f"Stage 3 (HMM/Viterbi Decoding) completed in {time_p3*1000:.1f} ms.")

    decoded_arcs = decode_result["decoded_arcs"]
    reconstructed_route = decode_result["reconstructed_route"]
    continuity = decode_result["continuity"]

    # 6. Extract Map Coordinates for Client Visualization
    # Predicted route polylines: list of [lat, lon] coordinates connecting node endpoints
    route_coords: List[List[float]] = []
    for arc_id in reconstructed_route:
        if arc_id >= 0 and arc_id < g.n_arcs:
            u_node = g.u[arc_id]
            v_node = g.v[arc_id]
            # Nodes are stored as [lon, lat], Folium expects [lat, lon]
            tail_lat = float(trip.nodes[u_node, 1])
            tail_lon = float(trip.nodes[u_node, 0])
            head_lat = float(trip.nodes[v_node, 1])
            head_lon = float(trip.nodes[v_node, 0])
            
            if not route_coords:
                route_coords.append([tail_lat, tail_lon])
            route_coords.append([head_lat, head_lon])

    # Raw GPS coordinates: [lat, lon]
    gps_coords: List[List[float]] = [
        [float(row.lat), float(row.lon)] for _, row in trip.track.iterrows()
    ]

    # 7. Server-Side Evaluation (Confidential: reference geometry is NEVER returned to client)
    metrics_obj = None
    route_file = trip_dir / f"{trip_id}.route"
    if route_file.exists():
        try:
            # Server reads reference route for internal accuracy calculation only
            ref_trip = load_trip(DEFAULT_DATA_ROOT, partition, trip_id, with_route=True)
            ref_df = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g, ref_trip.route)
            ref_arcs = ref_df["ref_arc_id"].to_numpy()
            pred_arcs = np.array(decoded_arcs)

            # Top-1 observation accuracy
            top1_acc = float((pred_arcs == ref_arcs).mean())

            # Multi-class Observation Macro F1
            all_classes = np.unique(np.concatenate([ref_arcs, pred_arcs]))
            obs_macro_f1 = float(f1_score(ref_arcs, pred_arcs, labels=all_classes, average="macro", zero_division=0))

            # Sequence-level Route Jaccard similarity
            set_rec = set(reconstructed_route)
            set_true = set(ref_trip.route)
            jaccard = float(len(set_rec & set_true) / max(len(set_rec | set_true), 1))

            metrics_obj = PredictionMetrics(
                top1_accuracy=top1_acc,
                route_continuity=continuity,
                observation_macro_f1=obs_macro_f1,
                route_jaccard=jaccard,
                n_observations=len(obs),
                n_decoded_arcs=len(reconstructed_route),
            )
            logger.info(f"Server-side evaluation: Top-1={top1_acc*100:.2f}%, Continuity={continuity*100:.2f}%, Jaccard={jaccard*100:.2f}%")
        except Exception as eval_err:
            logger.warning(f"Server-side evaluation skipped or failed: {eval_err}")

    total_time = time.perf_counter() - t_start
    logger.info(f"Total pipeline execution for {trip_id}: {total_time:.3f} s (P1: {time_p1*1000:.1f}ms, P2: {time_p2*1000:.1f}ms, P3: {time_p3*1000:.1f}ms)")

    return PredictionResponse(
        trip_id=trip_id,
        status="success",
        metrics=metrics_obj,
        predicted_route=reconstructed_route,
        route_coordinates=route_coords,
        gps_coordinates=gps_coords,
        inference_time_seconds=total_time,
        timing_breakdown=TimingBreakdown(
            feature_generation_s=time_p1,
            model_scoring_s=time_p2,
            viterbi_decoding_s=time_p3,
            total_inference_s=total_time,
        ),
        message=f"Trajectory map-matched successfully across {len(reconstructed_route)} reconstructed road arcs.",
    )


if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0:8000 for distributed two-laptop communication
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
