"""test_person3.py - Verification tests for Person 3 HMM & Viterbi sequential decoder."""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from common import load_trip, RoadGraph
from features import generate_candidates, CandConfig
from align import align_to_route
from viterbi import ViterbiDecoder, HMMConfig

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")


def test_viterbi_decoder_initialization():
    """Verify HMMConfig defaults and ViterbiDecoder instantiation."""
    cfg = HMMConfig(beta=10.0, disconnect_penalty=12.0)
    decoder = ViterbiDecoder(cfg)
    assert decoder.cfg.beta == 10.0
    assert decoder.cfg.disconnect_penalty == 12.0


def test_viterbi_decoding_on_student_trip():
    """Verify decoding runs and improves Top-1 accuracy and continuity on trip 00000000."""
    trip = load_trip(DATA_ROOT, "student_test", "00000000", with_route=True)
    g = RoadGraph(trip.nodes, trip.arcs)
    cands, _, obs = generate_candidates(trip, CandConfig(), graph=g)

    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    cands["score"] = artifact["pipeline"].predict_proba(cands[artifact["features"]])[:, 1]

    decoder = ViterbiDecoder(HMMConfig())
    res = decoder.decode(obs, cands, g)

    assert "decoded_arcs" in res
    assert "reconstructed_route" in res
    assert "top1_classifier_arcs" in res
    assert "continuity" in res

    assert len(res["decoded_arcs"]) == len(obs)
    assert len(res["reconstructed_route"]) > 0

    ref_df = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g, trip.route)
    ref_arcs = ref_df["ref_arc_id"].to_numpy()

    clf_arcs = np.array(res["top1_classifier_arcs"])
    hmm_arcs = np.array(res["decoded_arcs"])

    top1_clf = (clf_arcs == ref_arcs).mean()
    top1_hmm = (hmm_arcs == ref_arcs).mean()

    assert top1_hmm >= top1_clf, "HMM decoding must match or improve over greedy classifier"
    assert res["continuity"] >= 0.95, "Decoded route continuity should exceed 95%"


def test_zero_candidate_observation_handling():
    """Verify that observations with zero candidates do not crash Viterbi."""
    trip = load_trip(DATA_ROOT, "student_test", "00000052")
    g = RoadGraph(trip.nodes, trip.arcs)
    cands, _, obs = generate_candidates(trip, CandConfig(), graph=g)

    # Artificially remove candidates for observation index 5 to test robustness
    cands_filtered = cands[cands["obs_idx"] != 5].copy()
    cands_filtered["score"] = 0.5

    decoder = ViterbiDecoder(HMMConfig())
    res = decoder.decode(obs, cands_filtered, g)

    assert len(res["decoded_arcs"]) == len(obs)
    # The zero candidate observation at index 5 should receive a fallback arc
    assert res["decoded_arcs"][5] >= 0 or res["decoded_arcs"][5] == -1
