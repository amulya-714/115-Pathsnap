import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add data_processing to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from common import load_trip, RoadGraph, LocalProjector, seg_point_dist, TripFormatError
from features import generate_candidates, CandConfig, FEATURE_COLUMNS, KEY_COLUMNS

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")


def test_load_trip_safe():
    """Verify inference-safe loading does not open .route."""
    trip = load_trip(DATA_ROOT, "student_test", "00000000", with_route=False)
    assert trip.trip_id == "00000000"
    assert trip.partition == "student_test"
    assert trip.route is None, "Inference trip loading must not open .route"
    assert len(trip.track) > 0
    assert len(trip.nodes) > 0
    assert len(trip.arcs) > 0


def test_road_graph_topology():
    """Verify road graph directed edges, successors, and coordinate projection."""
    trip = load_trip(DATA_ROOT, "student_test", "00000000")
    g = RoadGraph(trip.nodes, trip.arcs)
    assert g.n_nodes == len(trip.nodes)
    assert g.n_arcs == len(trip.arcs)
    assert len(g.length_m) == g.n_arcs
    assert len(g.bearing_deg) == g.n_arcs

    # Verify successors function
    succ_0 = g.successors(0)
    assert isinstance(succ_0, np.ndarray)
    if len(succ_0) > 0:
        head_node = g.v[0]
        for s in succ_0:
            assert g.u[s] == head_node, "Successor must start at head node of arc"


def test_local_projector():
    """Verify forward and inverse projection consistency."""
    proj = LocalProjector(12.9716, 77.5946)
    x, y = proj.fwd(12.9720, 77.5950)
    lon_back, lat_back = proj.inv(x, y)
    assert np.isclose(lon_back, 12.9720, atol=1e-6)
    assert np.isclose(lat_back, 77.5950, atol=1e-6)


def test_candidate_generation_contract():
    """Verify candidate generation produces the exact 13 features contract."""
    trip = load_trip(DATA_ROOT, "student_test", "00000000")
    cfg = CandConfig(radius_m=50.0, k_max=50)
    cands, graph, obs = generate_candidates(trip, cfg)

    assert isinstance(cands, pd.DataFrame)
    assert isinstance(obs, pd.DataFrame)
    assert len(obs) == len(trip.track)

    # Check key columns
    for key in KEY_COLUMNS:
        assert key in cands.columns

    # Check all 13 feature columns
    for feat in FEATURE_COLUMNS:
        assert feat in cands.columns

    # Check column order
    expected_order = KEY_COLUMNS + FEATURE_COLUMNS
    assert list(cands.columns) == expected_order

    # Check value bounds
    assert (cands["dist_m"] >= 0.0).all()
    assert (cands["dist_m"] <= cfg.radius_m + 1e-3).all()
    assert (cands["heading_diff_deg"] >= 0.0).all()
    assert (cands["heading_diff_deg"] <= 180.0).all()
    assert (cands["along_frac"] >= 0.0).all()
    assert (cands["along_frac"] <= 1.0).all()


def test_feature_code_route_independence():
    """Verify features.py never contains references to .route or route file."""
    features_py = (REPO_ROOT / "data_processing" / "features.py").read_text(encoding="utf-8")
    import ast
    tree = ast.parse(features_py)
    # Check that no attribute lookup or identifier named 'route' exists outside docstrings
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "route" not in names, "features.py must not reference route identifier"
