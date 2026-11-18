"""Stage 5 - writes out/HANDOFF.md for Persons 2 and 3 from the ACTUAL tables (schema, order, dtypes, sample rows, counts).
python s5_handoff.py --data-root "Student folder" --out out
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from common import load_trip, RoadGraph
from features import FEATURE_COLUMNS, KEY_COLUMNS, DTYPES, CandConfig
from align import attach_labels

DOC = {
 "trip_id": "8-char zero-padded string; folder name. Partition is NOT in the table - keep partition in a separate column when you concatenate.",
 "obs_idx": "0-based row in that trip's .track",
 "arc_id": "0-based row in that trip's .arcs. LOCAL to trip_id; (trip_id, arc_id) is the only valid arc key",
 "dist_m": "metres, point-to-segment, local projection", "heading_diff_deg": "deg 0..180; 90 when heading_valid=0",
 "heading_cos": "cos(heading_diff); 0 when heading_valid=0", "heading_valid": "1 if GPS heading estimable (>= 6 m moved over +-2 fixes)",
 "along_frac": "0..1 foot-point position along arc", "arc_len_m": "metres", "speed_mps": "m/s, same for all candidates of an obs",
 "dist_rank": "0 = nearest candidate", "dist_minus_min_m": "metres above nearest candidate", "n_cand": "candidates kept for this obs",
 "dt_prev_s": "s since previous fix (0 first)", "dt_next_s": "s to next fix (0 last)", "has_reverse_arc": "1 if opposite-direction arc exists"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", default="Student folder"); ap.add_argument("--out", default="out")
    a = ap.parse_args(); out = Path(a.out); cfg = CandConfig()
    cand = {p: pd.read_csv(out / f"candidates_{p}.csv.gz", dtype={"trip_id": str}).astype({k: v for k, v in DTYPES.items() if k != "trip_id"}) for p in ("train", "student_test")}
    lab = {p: pd.read_csv(out / f"labels_{p}.csv.gz", dtype={"trip_id": str}) for p in ("train", "student_test")}
    lines = []; w = lines.append
    w("# PathSnap Person 1 -> Persons 2 & 3 handoff\n")
    w(f"Generated from actual files. Config: `{cfg}`. Partitions used: train (80 trips), student_test (10 trips). No evaluator data.\n")
    w("## Files (all in `out/`)\n| file | rows | contains | uses .route? |\n|---|---|---|---|")
    for p in cand:
        w(f"| `candidates_{p}.csv.gz` | {len(cand[p]):,} | keys + 13 features, one row per (obs, candidate arc) | **No** (inference-safe) |")
    for p in lab:
        w(f"| `labels_{p}.csv.gz` | {len(lab[p]):,} | one row per GPS fix: dev label from reference-route alignment | yes - DEVELOPMENT ONLY |")
    w("\n## Candidate table schema (column order is the contract)\n| # | column | dtype | meaning |\n|---|---|---|---|")
    for i, c in enumerate(KEY_COLUMNS + FEATURE_COLUMNS): w(f"| {i} | `{c}` | {DTYPES[c]} | {DOC[c]} |")
    w("\n**FEATURE_COLUMNS (model input order, 13):**\n```python\n" + repr(FEATURE_COLUMNS) + "\n```")
    w("Import them instead of retyping: `from features import FEATURE_COLUMNS, KEY_COLUMNS`. Fit any scaler on TRAIN rows only. Group CV by `trip_id` (never split a trip).\n")
    w("## Label table schema (DEV ONLY - never present at inference)\n| column | dtype | meaning |\n|---|---|---|")
    for c, d, m in [("trip_id", "str", "as above"), ("obs_idx", "int", "as above"), ("route_pos", "int", "index into that trip's .route (monotone non-decreasing)"),
                    ("ref_arc_id", "int", "arc_id from .route at route_pos = development label"), ("align_dist_m", "float", "GPS -> labelled arc distance, m"),
                    ("nearest_route_dist_m", "float", "diagnostic: distance to nearest route arc ignoring order"),
                    ("label_ok", "int8", "1 if align_dist_m <= 100 m; 0 = label suspect (reference route probably does not explain this fix)")]:
        w(f"| `{c}` | {d} | {m} |")
    w("\nJoin: `from align import attach_labels; df = attach_labels(cand, labels)` -> adds `y` (1 if arc_id == ref_arc_id) and `label_ok`; candidate rows are never dropped.\n")
    # real numbers
    tr = attach_labels(cand["train"], lab["train"]); st = attach_labels(cand["student_test"], lab["student_test"])
    for nm, d, L in (("train", tr, lab["train"]), ("student_test", st, lab["student_test"])):
        has = d.groupby(["trip_id", "obs_idx"]).y.max()
        n_obs = len(L)
        w(f"- **{nm}**: {len(d):,} candidate rows; positive rows y=1: {int(d.y.sum()):,} ({d.y.mean():.3%}); observations with a positive in the candidate set: {int(has.sum()):,} of {n_obs:,} "
          f"-> candidate miss rate {1 - has.sum() / n_obs:.4f} (natural set, before any training-only correction).")
    w("\n## Sample rows (real rows: train trip 00000003, obs 100 and 101; top 4 candidates each)")
    s = tr[(tr.trip_id == "00000003") & (tr.obs_idx.isin([100, 101])) & (tr.dist_rank < 4)]
    w("```\n" + s[KEY_COLUMNS + FEATURE_COLUMNS + ["y", "label_ok"]].round(3).to_string(index=False) + "\n```")
    s2 = lab["train"][(lab["train"].trip_id == "00000003") & lab["train"].obs_idx.isin([100, 101])]
    w("```\n" + s2.round(3).to_string(index=False) + "\n```")
    t = load_trip(a.data_root, "train", "00000003"); g = RoadGraph(t.nodes, t.arcs)
    w("\n## Road-graph format (per trip; LOCAL ids)\n- Raw: `<trip>.nodes` (lon lat; row = node_id), `<trip>.arcs` (from_node to_node; row = arc_id), whitespace/TAB separated, no header.")
    w("- Arcs are **directed** u -> v; a two-way street is two arcs (arcs with a reverse twin: see EDA). Some arcs are exact duplicates (same u,v, different arc_id) and some have zero length - kept, not removed.")
    w("- Never merge graphs of different trips: node_id and arc_id restart at 0 in each trip, and trips are in different cities.")
    w("- Load: `from common import load_trip, RoadGraph; trip = load_trip(root, 'train', '00000003'); g = RoadGraph(trip.nodes, trip.arcs)`  (`.route` is NOT opened unless `with_route=True`).")
    w("- `g.successors(arc_id)` -> np.array of arc_ids that can follow (leave the head node `g.v[arc_id]`); includes the U-turn (reverse arc). Use for HMM transitions; `g.reverse_id[arc_id]` is the opposite arc or -1.")
    w("- Arrays on `g` (length n_arcs): `u, v, ax, ay, bx, by` (tail/head node ids and local metres), `length_m`, `bearing_deg` (clockwise from north, NaN for zero-length). Node arrays: `nx, ny`.")
    w("- Projection: per-trip local equirectangular metres `g.proj.fwd(lon, lat) -> (x, y)`; keep the SAME projector for observations and arcs of a trip.")
    w(f"\nExample (trip 00000003): n_nodes={g.n_nodes}, n_arcs={g.n_arcs}; first 3 arcs:")
    w("```\n" + pd.DataFrame({"arc_id": range(3), "u": g.u[:3], "v": g.v[:3], "length_m": g.length_m[:3].round(2), "bearing_deg": g.bearing_deg[:3].round(1), "reverse_id": g.reverse_id[:3]}).to_string(index=False) + "\n```")
    w("\n## Known data caveats you must respect\n- Observations with zero candidates exist (nothing within 50 m): see `s2_summary.csv`; Person 3 must handle an observation with no candidate state (gap/skip) in Viterbi.")
    chk = pd.read_csv(out / "s3_trip_checks.csv", dtype={"trip_id": str}); sus = chk[chk.frac_gt100 > .05]
    w("- Reference route does not explain the GPS in these trips (>5% of fixes >100 m from aligned arc): " + ", ".join(f"{r.partition}/{r.trip_id} ({r.frac_gt100:.0%})" for r in sus.itertuples())
      + ". `label_ok=0` marks fixes. Decide as a team how to treat them for TRAINING; keep them visible (and reported) in student_test evaluation.")
    w("- A repeated arc id in the route is normal (loops); labels are position-based. Duplicate-geometry twin arcs have identical features; only one of them is the label.")
    (out / "HANDOFF.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    sys.exit(main())
