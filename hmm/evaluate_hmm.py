"""evaluate_hmm.py - Systematic evaluation of Classifier-Only vs HMM/Viterbi.

Evaluates all 10 student-test trips and produces:
1. Per-trip and aggregate comparison metrics (Top-1, Continuity, Macro F1, Latency)
2. Comparison CSV table: hmm/results/hmm_vs_classifier.csv
3. Visualization plot: hmm/results/hmm_comparison.png
"""
import sys
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Ensure data_processing and hmm are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data_processing"))
sys.path.insert(0, str(REPO_ROOT / "hmm"))

from common import load_trip, RoadGraph, list_trips
from features import generate_candidates, CandConfig
from align import align_to_route
from viterbi import ViterbiDecoder, HMMConfig

DATA_ROOT = Path("/Users/amulya/Downloads/Student folder")
OUT_DIR = REPO_ROOT / "hmm" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_all():
    print("Loading Person 2 final model artifact...")
    artifact = joblib.load(REPO_ROOT / "models" / "results" / "final_model.joblib")
    pipeline = artifact["pipeline"]
    features = artifact["features"]

    decoder = ViterbiDecoder(HMMConfig())
    trips = list_trips(DATA_ROOT, "student_test")
    print(f"Evaluating {len(trips)} student_test trips: {trips}")

    rows = []
    total_obs = 0
    total_clf_correct = 0
    total_hmm_correct = 0

    for trip_id in trips:
        t_start = time.perf_counter()
        trip = load_trip(DATA_ROOT, "student_test", trip_id, with_route=True)
        g = RoadGraph(trip.nodes, trip.arcs)

        # 1. Feature generation
        cand_cfg = CandConfig()
        cands, _, obs = generate_candidates(trip, cand_cfg, graph=g)

        # 2. Model scoring
        t_model_start = time.perf_counter()
        cands["score"] = pipeline.predict_proba(cands[features])[:, 1]
        model_time = time.perf_counter() - t_model_start

        # 3. Ground truth alignment for dev evaluation
        ref_df = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g, trip.route)
        ref_arcs = ref_df["ref_arc_id"].to_numpy()

        # 4. HMM / Viterbi decoding
        t_hmm_start = time.perf_counter()
        decode_res = decoder.decode(obs, cands, g)
        hmm_time = time.perf_counter() - t_hmm_start

        total_time = time.perf_counter() - t_start

        clf_arcs = np.array(decode_res["top1_classifier_arcs"])
        hmm_arcs = np.array(decode_res["decoded_arcs"])
        n_obs = len(obs)

        # Accuracies
        clf_correct = int((clf_arcs == ref_arcs).sum())
        hmm_correct = int((hmm_arcs == ref_arcs).sum())
        top1_clf = clf_correct / n_obs
        top1_hmm = hmm_correct / n_obs

        total_obs += n_obs
        total_clf_correct += clf_correct
        total_hmm_correct += hmm_correct

        # Continuities
        def calc_cont(arcs):
            if len(arcs) <= 1:
                return 1.0
            valid = (arcs[:-1] >= 0) & (arcs[1:] >= 0)
            b, a = arcs[:-1][valid], arcs[1:][valid]
            connected = (b == a) | (g.v[b] == g.u[a])
            return float(connected.mean()) if len(connected) > 0 else 1.0

        cont_clf = calc_cont(clf_arcs)
        cont_hmm = calc_cont(hmm_arcs)

        # Observation-level Macro F1
        all_arcs_unique = np.unique(np.concatenate([ref_arcs, clf_arcs, hmm_arcs]))
        macro_f1_clf = f1_score(ref_arcs, clf_arcs, labels=all_arcs_unique, average="macro", zero_division=0)
        macro_f1_hmm = f1_score(ref_arcs, hmm_arcs, labels=all_arcs_unique, average="macro", zero_division=0)

        # Sequence route metrics
        reconstructed = decode_res["reconstructed_route"]
        ground_truth_route = list(trip.route)
        
        # Jaccard set overlap of route arcs
        set_rec = set(reconstructed)
        set_true = set(ground_truth_route)
        route_jaccard = len(set_rec & set_true) / max(len(set_rec | set_true), 1)

        rows.append({
            "trip_id": trip_id,
            "n_obs": n_obs,
            "clf_top1": top1_clf,
            "hmm_top1": top1_hmm,
            "top1_delta": top1_hmm - top1_clf,
            "clf_continuity": cont_clf,
            "hmm_continuity": cont_hmm,
            "continuity_delta": cont_hmm - cont_clf,
            "clf_macro_f1": macro_f1_clf,
            "hmm_macro_f1": macro_f1_hmm,
            "route_jaccard": route_jaccard,
            "model_time_s": model_time,
            "hmm_time_s": hmm_time,
            "total_time_s": total_time,
        })
        print(f"[{trip_id}] obs={n_obs} | Clf Top1={top1_clf*100:.2f}% -> HMM Top1={top1_hmm*100:.2f}% | "
              f"Clf Cont={cont_clf*100:.2f}% -> HMM Cont={cont_hmm*100:.2f}% ({total_time:.2f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "hmm_vs_classifier.csv", index=False)

    overall_clf_top1 = total_clf_correct / total_obs
    overall_hmm_top1 = total_hmm_correct / total_obs
    overall_clf_cont = df["clf_continuity"].mean()
    overall_hmm_cont = df["hmm_continuity"].mean()

    print("\n" + "=" * 60)
    print("OVERALL STUDENT-TEST HMM EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Observations:         {total_obs}")
    print(f"Classifier-Only Top-1:      {overall_clf_top1*100:.2f}%")
    print(f"HMM/Viterbi Top-1:          {overall_hmm_top1*100:.2f}% (Delta: +{(overall_hmm_top1 - overall_clf_top1)*100:.2f}%)")
    print(f"Classifier Route Continuity:{overall_clf_cont*100:.2f}%")
    print(f"HMM Route Continuity:       {overall_hmm_cont*100:.2f}% (Delta: +{(overall_hmm_cont - overall_clf_cont)*100:.2f}%)")
    print(f"Average Route Jaccard:      {df['route_jaccard'].mean()*100:.2f}%")
    print(f"Average Inference Time:     {df['total_time_s'].mean():.3f} s / trip")
    print("=" * 60)

    # Generate visual comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df))
    width = 0.35

    axes[0].bar(x - width/2, df["clf_top1"] * 100, width, label="Classifier Only", color="#4A90E2")
    axes[0].bar(x + width/2, df["hmm_top1"] * 100, width, label="HMM / Viterbi", color="#50E3C2")
    axes[0].set_ylabel("Observation Top-1 Accuracy (%)")
    axes[0].set_title("Pointwise Top-1 Accuracy per Trip")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["trip_id"], rotation=45)
    axes[0].set_ylim(70, 100)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    axes[1].bar(x - width/2, df["clf_continuity"] * 100, width, label="Classifier Only", color="#E94E77")
    axes[1].bar(x + width/2, df["hmm_continuity"] * 100, width, label="HMM / Viterbi", color="#F8E71C")
    axes[1].set_ylabel("Route Continuity (%)")
    axes[1].set_title("Road Network Continuity per Trip")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["trip_id"], rotation=45)
    axes[1].set_ylim(80, 102)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = OUT_DIR / "hmm_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    evaluate_all()
