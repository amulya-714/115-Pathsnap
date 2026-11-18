"""Focused regression tests for evaluator inference data loading."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / name) for name in ("backend", "data_processing", "models", "hmm")]

import common
import main


def _write_trip(root, partition="evaluator_test", trip_id="00000001"):
    trip_dir = root / partition / trip_id
    trip_dir.mkdir(parents=True)
    (trip_dir / f"{trip_id}.track").write_text("12.0 77.0 0\n12.001 77.001 1\n")
    (trip_dir / f"{trip_id}.nodes").write_text("12.0 77.0\n12.001 77.001\n")
    (trip_dir / f"{trip_id}.arcs").write_text("0 1\n")
    # Deliberately invalid route contents prove inference does not parse this file.
    (trip_dir / f"{trip_id}.route").write_text("must never be read\n")
    return trip_dir


def test_load_trip_accepts_evaluator_test_without_reading_route(tmp_path):
    _write_trip(tmp_path)

    trip = common.load_trip(tmp_path, "evaluator_test", "00000001")

    assert trip.partition == "evaluator_test"
    assert trip.route is None
    assert len(trip.track) == 2
    assert trip.arcs.tolist() == [[0, 1]]


def test_predict_evaluator_trip_never_checks_or_loads_reference_route(tmp_path, monkeypatch):
    trip_dir = _write_trip(tmp_path)
    monkeypatch.setattr(main, "DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr(main, "model_features", ["feature"])
    monkeypatch.setattr(main, "model_pipeline", type("Model", (), {
        "predict_proba": lambda self, frame: np.array([[0.1, 0.9]])
    })())
    monkeypatch.setattr(main, "generate_candidates", lambda trip, cfg, graph: (
        pd.DataFrame({"feature": [1.0]}), graph,
        pd.DataFrame({"lat": [77.0, 77.001], "lon": [12.0, 12.001]}),
    ))
    monkeypatch.setattr(main.hmm_decoder, "decode", lambda obs, cands, graph: {
        "decoded_arcs": [0], "reconstructed_route": [0], "continuity": 1.0,
    })

    original_exists = Path.exists

    def guarded_exists(path):
        if path == trip_dir / "00000001.route":
            raise AssertionError("prediction inspected evaluator reference route")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    response = TestClient(main.app).post("/predict", json={"trip_id": "00000001"})

    assert response.status_code == 200, response.text
    assert response.json()["metrics"] is None
    assert response.json()["predicted_route"] == [0]
