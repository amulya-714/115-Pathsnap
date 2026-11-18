"""Train, compare, tune and export PathSnap candidate classifiers."""

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

KEY = ["trip_id", "obs_idx"]
SEED = 42


def models():
    return {
        "logistic": (LogisticRegression(max_iter=400), "C", [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30]),
        "naive_bayes": (GaussianNB(), "var_smoothing", np.logspace(-12, -3, 10).tolist()),
        "decision_tree": (DecisionTreeClassifier(max_depth=8, random_state=SEED), "max_depth", [2, 3, 5, 8, 12, 16, 24, 32]),
        "random_forest": (RandomForestClassifier(n_estimators=40, max_depth=12, max_samples=0.25, n_jobs=4, random_state=SEED),
                          "max_depth", [3, 5, 8, 12, 16, 20, 24, 32]),
        "hist_boosting": (HistGradientBoostingClassifier(max_iter=80, random_state=SEED, early_stopping=False),
                         "learning_rate", [0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3]),
    }


def load_split(candidates_path, reference_path, features):
    candidates = pd.read_csv(candidates_path, dtype={"trip_id": str, **dict.fromkeys(features, "float32")})
    truth = pd.read_csv(reference_path, dtype={"trip_id": str}).rename(columns={"ref_arc_id": "reference_arc_id"})
    if "label_ok" not in truth:
        truth["label_ok"] = 1
    truth = truth[KEY + ["reference_arc_id", "label_ok"]]
    candidates = candidates[KEY + ["arc_id"] + features].copy()
    columns = KEY + ["arc_id"]
    if candidates[columns].isna().any().any() or truth.isna().any().any():
        raise ValueError("Candidate identifiers and reference records must be complete.")
    if candidates.duplicated(columns).any() or truth.duplicated(KEY).any():
        raise ValueError("Expected unique candidates per observation and one reference row per observation.")
    for feature in features:
        candidates[feature] = pd.to_numeric(candidates[feature], errors="raise").replace(
            [np.inf, -np.inf], np.nan
        )
    labeled = candidates.merge(truth, on=KEY, how="left", validate="many_to_one")
    if labeled["reference_arc_id"].isna().any():
        raise ValueError("Every candidate observation needs a matching development reference row.")
    labeled["label"] = (labeled["arc_id"] == labeled["reference_arc_id"]).astype(int)
    return labeled, truth


def score_candidates(estimator, candidates, features):
    frame = candidates[KEY + ["arc_id"]].copy()
    if frame.empty:
        frame["score"] = pd.Series(dtype=float)
        return frame
    numeric = candidates[features].apply(pd.to_numeric, errors="raise").astype("float32")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    frame["score"] = estimator.predict_proba(numeric)[:, 1]
    return frame


def evaluate(estimator, candidates, truth, features):
    start = perf_counter()
    scores = score_candidates(estimator, candidates, features)
    inference_seconds = perf_counter() - start
    best = (scores.sort_values(["score", "arc_id"], ascending=[False, True])
            .drop_duplicates(KEY)[KEY + ["arc_id"]])
    result = truth.merge(best, on=KEY, how="left")
    actual_arcs = pd.MultiIndex.from_arrays([result.trip_id, result.reference_arc_id])
    predicted_arcs = pd.MultiIndex.from_arrays([result.trip_id, result.arc_id.fillna(-1)])
    arc_index = actual_arcs.unique().union(predicted_arcs.unique())
    hits = candidates.loc[candidates["label"].eq(1), KEY].drop_duplicates()
    observed = len(truth)
    labels = candidates["label"]
    predicted = scores["score"].ge(0.5)
    metrics = {
        "candidate_macro_f1": f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "candidate_macro_precision": precision_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "candidate_macro_recall": recall_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0),
        "candidate_log_loss": log_loss(labels, scores["score"], labels=[0, 1]) if len(labels) else np.nan,
        "observation_top1": result["arc_id"].eq(result["reference_arc_id"]).mean(),
        "observation_macro_f1": f1_score(
            arc_index.get_indexer(actual_arcs), arc_index.get_indexer(predicted_arcs),
            labels=arc_index.get_indexer(actual_arcs.unique()), average="macro", zero_division=0,
        ),
        "candidate_miss_rate": (observed - len(hits)) / observed,
        "inference_seconds": inference_seconds,
    }
    return metrics, scores


def fit_and_evaluate(model, train, validation, truth, features):
    train = train.loc[train["label_ok"].eq(1)]
    if train["label"].nunique() != 2:
        raise ValueError("A training fold needs both positive and negative candidates.")
    estimator = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(model))
    start = perf_counter()
    estimator.fit(train[features], train["label"])
    training_seconds = perf_counter() - start
    metrics, scores = evaluate(estimator, validation, truth, features)
    metrics["training_seconds"] = training_seconds
    return estimator, metrics, scores


def bayesian_search(values, trials, objective):
    # Gaussian process over a discrete grid with an upper-confidence-bound acquisition.
    grid = np.log(np.asarray(values, dtype=float))
    grid = ((grid - grid.min()) / (grid.max() - grid.min())).reshape(-1, 1)
    used, results = [], []
    for step in range(trials):
        if step < 2:
            index = [0, len(values) - 1][step]
        else:
            remaining = [i for i in range(len(values)) if i not in used]
            gp = GaussianProcessRegressor(kernel=Matern(), alpha=1e-5, normalize_y=True,
                                          optimizer=None, random_state=SEED)
            gp.fit(grid[used], results)
            mean, std = gp.predict(grid[remaining], return_std=True)
            index = remaining[int(np.argmax(mean + 1.96 * std))]
        used.append(index)
        results.append(objective(step + 1, values[index]))
    return values[used[int(np.argmax(results))]]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("train-candidates", "train-reference", "test-candidates", "test-reference", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    cli.add_argument("--features", nargs="+", required=True)
    cli.add_argument("--folds", type=int, default=3)
    cli.add_argument("--trials", type=int, choices=range(3, 9), default=6)
    cli.add_argument("--baseline-only", action="store_true")
    cli.add_argument("--student-graph-root", type=Path, help="Raw student_test folder containing trip directories.")
    args = cli.parse_args()
    features = args.features
    reserved = {*KEY, "arc_id", "reference_arc_id", "ref_arc_id", "label", "label_ok",
                "route_pos", "align_dist_m", "nearest_route_dist_m", "score"}
    if reserved.intersection(features) or len(set(features)) != len(features):
        cli.error("Use distinct inference features only. Exclude identifiers, labels and scores.")
    train, train_truth = load_split(args.train_candidates, args.train_reference, features)
    test, test_truth = load_split(args.test_candidates, args.test_reference, features)
    if set(train_truth.trip_id).intersection(test_truth.trip_id):
        cli.error("Training and student-testing trip IDs overlap.")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    specs = models()
    if args.baseline_only:
        specs = {"logistic": specs["logistic"]}
    folds = list(GroupKFold(n_splits=args.folds).split(train_truth, groups=train_truth.trip_id))
    fold_rows, comparison, trials = [], [], []
    graphs = {}
    if args.student_graph_root:
        for trip in test_truth.trip_id.unique():
            if not (len(trip) == 8 and trip.isascii() and trip.isdecimal()):
                cli.error("Expected eight-digit raw trip IDs.")
            graphs[trip] = np.loadtxt(args.student_graph_root / trip / f"{trip}.arcs", dtype=int, ndmin=2)

    def cv(name, model):
        for number, (fit_idx, val_idx) in enumerate(folds, start=1):
            fit_ids = train_truth.iloc[fit_idx].trip_id
            val_truth = train_truth.iloc[val_idx]
            _, metrics, _ = fit_and_evaluate(
                model, train[train.trip_id.isin(fit_ids)],
                train[train.trip_id.isin(val_truth.trip_id)], val_truth, features
            )
            fold_rows.append({"run": name, "fold": number, **metrics})
        return float(np.mean([row["candidate_macro_f1"] for row in fold_rows[-len(folds):]]))

    def record(name, model, cv_f1):
        fitted, metrics, scores = fit_and_evaluate(model, train, test, test_truth, features)
        if graphs:
            best = (scores.sort_values(["score", "arc_id"], ascending=[False, True])
                    .drop_duplicates(KEY)[KEY + ["arc_id"]])
            obs = test_truth[KEY].merge(best, on=KEY, how="left").sort_values(KEY)
            continuous, pairs = 0, 0
            for trip_id, group in obs.groupby("trip_id"):
                arcs = group.arc_id.fillna(-1).to_numpy(dtype=int)
                before, after = arcs[:-1], arcs[1:]
                observed = (before >= 0) & (after >= 0)
                graph = graphs[trip_id]
                connected = (before[observed] == after[observed]) | (
                    graph[before[observed], 1] == graph[after[observed], 0]
                )
                continuous += int(connected.sum())
                pairs += len(before)
            metrics["direct_route_continuity"] = continuous / pairs if pairs else np.nan
        comparison.append({"run": name, "cv_candidate_macro_f1": cv_f1, **metrics})
        scores.to_csv(out / f"{name}_student_scores.csv", index=False)
        joblib.dump({"pipeline": fitted, "features": features}, out / f"{name}.joblib")
        pd.DataFrame(comparison).to_csv(out / "comparison.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "cv_folds.csv", index=False)
        print(name, comparison[-1], flush=True)

    for name, (model, _, _) in specs.items():
        record(name, model, cv(name, model))
    winner = max(comparison, key=lambda row: row["cv_candidate_macro_f1"])["run"]
    model, parameter, grid = specs[winner]
    best = None
    final_name = winner
    if not args.baseline_only:
        def objective(trial, value):
            started = perf_counter()
            f1 = cv(f"trial_{trial}", clone(model).set_params(**{parameter: value}))
            trials.append({"trial": trial, parameter: value, "cv_candidate_macro_f1": f1,
                           "seconds": perf_counter() - started})
            pd.DataFrame(trials).to_csv(out / "bayesian_trials.csv", index=False)
            return f1

        best = bayesian_search(grid, args.trials, objective)
        record(f"{winner}_tuned", clone(model).set_params(**{parameter: best}),
               max(row["cv_candidate_macro_f1"] for row in trials))
        if comparison[-1]["cv_candidate_macro_f1"] >= max(
            row["cv_candidate_macro_f1"] for row in comparison[:-1]
        ):
            final_name = f"{winner}_tuned"
        import matplotlib.pyplot as plt
        table = pd.DataFrame(comparison).set_index("run")
        for name, columns in {
            "f1_and_top1": ["cv_candidate_macro_f1", "observation_macro_f1", "observation_top1"],
            "student_log_loss": ["candidate_log_loss"],
        }.items():
            axes = table[columns].plot.bar(rot=25, figsize=(10, 5))
            axes.figure.tight_layout()
            axes.figure.savefig(out / f"{name}.png", dpi=150)
            plt.close(axes.figure)

    shutil.copyfile(out / f"{final_name}.joblib", out / "final_model.joblib")
    (out / "run_config.json").write_text(json.dumps({
        "features": features, "seed": SEED, "folds": args.folds,
        "models": {name: model.get_params() for name, (model, _, _) in specs.items()},
        "selected_by_cv": winner, "exported_model": final_name, "search_parameter": parameter,
        "search_grid": grid, "best_tested_value": best, "baseline_only": args.baseline_only,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
