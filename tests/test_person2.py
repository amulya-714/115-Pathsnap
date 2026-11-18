import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "models"))

from common import load_trip
from features import generate_candidates, CandConfig, FEATURE_COLUMNS
from pathsnap_models import score_candidates

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")


def test_saved_model_artifact_loading():
    """Verify final_model.joblib and other model artifacts load properly."""
    model_path = REPO_ROOT / "models" / "results" / "final_model.joblib"
    assert model_path.exists(), "final_model.joblib must exist"

    artifact = joblib.load(model_path)
    assert isinstance(artifact, dict)
    assert "pipeline" in artifact
    assert "features" in artifact
    assert artifact["features"] == FEATURE_COLUMNS, "Model feature order must match Person 1 FEATURE_COLUMNS"


def test_model_inference_on_student_test():
    """Verify model scoring pipeline on student_test trip 00000000."""
    trip = load_trip(DATA_ROOT, "student_test", "00000000")
    cands, graph, obs = generate_candidates(trip, CandConfig())

    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    scores_df = score_candidates(pipeline, cands, features)
    assert isinstance(scores_df, pd.DataFrame)
    assert len(scores_df) == len(cands)
    assert "score" in scores_df.columns
    assert (scores_df["score"] >= 0.0).all()
    assert (scores_df["score"] <= 1.0).all()


def test_top1_parity_with_person2_handoff():
    """Verify top-1 predictions match Person 2 pre-scored handoff table."""
    handoff_path = REPO_ROOT / "models" / "handoff" / "final_student_scores.csv.gz"
    assert handoff_path.exists(), "Pre-scored handoff table must exist"

    pre = pd.read_csv(handoff_path, dtype={"trip_id": str})
    pre_0 = pre[pre["trip_id"] == "00000000"].copy()

    trip = load_trip(DATA_ROOT, "student_test", "00000000")
    cands, graph, obs = generate_candidates(trip, CandConfig())

    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    cands["score"] = artifact["pipeline"].predict_proba(cands[artifact["features"]])[:, 1]

    best_pre = pre_0.sort_values(["score", "arc_id"], ascending=[False, True]).drop_duplicates("obs_idx").sort_values("obs_idx")
    best_cur = cands.sort_values(["score", "arc_id"], ascending=[False, True]).drop_duplicates("obs_idx").sort_values("obs_idx")

    assert len(best_pre) == len(best_cur)
    diff = (best_pre["arc_id"].values != best_cur["arc_id"].values).sum()
    assert diff == 0, f"Expected 100% top-1 agreement with Person 2 scores, found {diff} differences"
