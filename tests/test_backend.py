"""test_backend.py - Unit and integration tests for FastAPI backend service."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "models"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify GET / returns 200 and status online."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    """Verify GET /health returns status ok and model information."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert data["features_count"] == 13


def test_trips_endpoint():
    """Verify GET /trips returns list of available evaluation trips."""
    res = client.get("/trips")
    assert res.status_code == 200
    data = res.json()
    assert "trips" in data
    assert len(data["trips"]) >= 10
    assert "00000000" in data["trips"]


def test_predict_invalid_trip_id_format():
    """Verify POST /predict validates 8-digit trip ID length."""
    res = client.post("/predict", json={"trip_id": "123"})
    assert res.status_code == 422, "Trip ID shorter than 8 chars must be rejected"


def test_predict_trip_not_found():
    """Verify POST /predict returns 404 for non-existent trip."""
    res = client.post("/predict", json={"trip_id": "99999999"})
    assert res.status_code == 404
    data = res.json()
    assert "detail" in data


def test_predict_valid_student_trip():
    """Verify POST /predict executes full pipeline on student_test trip 00000000."""
    res = client.post("/predict", json={"trip_id": "00000000"})
    assert res.status_code == 200
    data = res.json()

    assert data["trip_id"] == "00000000"
    assert data["status"] == "success"
    assert "predicted_route" in data
    assert len(data["predicted_route"]) > 0
    assert "route_coordinates" in data
    assert len(data["route_coordinates"]) > 0
    assert "gps_coordinates" in data
    assert len(data["gps_coordinates"]) > 0

    # Verify timing breakdown
    timing = data["timing_breakdown"]
    assert timing["feature_generation_s"] > 0
    assert timing["model_scoring_s"] > 0
    assert timing["viterbi_decoding_s"] > 0
    assert timing["total_inference_s"] > 0

    # Verify server-side evaluation metrics
    metrics = data["metrics"]
    assert metrics is not None
    assert metrics["top1_accuracy"] > 0.90
    assert metrics["route_continuity"] > 0.95
    assert metrics["n_observations"] == 2503

    # CRITICAL EVALUATOR SECURITY CHECK:
    # Ensure reference geometry, reference labels, and private data are NOT in response
    assert "reference_route" not in data
    assert "reference_labels" not in data
    assert "ref_arc_id" not in data
