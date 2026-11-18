# PathSnap Person 1 -> Persons 2 & 3 handoff

Generated from actual files. Config: `CandConfig(radius_m=50.0, k_max=50, piece_m=50.0, min_move_m=6.0, half_window=2)`. Partitions used: train (80 trips), student_test (10 trips). No evaluator data.

## Files (all in `out/`)
| file | rows | contains | uses .route? |
|---|---|---|---|
| `candidates_train.csv.gz` | 1,843,151 | keys + 13 features, one row per (obs, candidate arc) | **No** (inference-safe) |
| `candidates_student_test.csv.gz` | 208,057 | keys + 13 features, one row per (obs, candidate arc) | **No** (inference-safe) |
| `labels_train.csv.gz` | 208,417 | one row per GPS fix: dev label from reference-route alignment | yes - DEVELOPMENT ONLY |
| `labels_student_test.csv.gz` | 19,160 | one row per GPS fix: dev label from reference-route alignment | yes - DEVELOPMENT ONLY |

## Candidate table schema (column order is the contract)
| # | column | dtype | meaning |
|---|---|---|---|
| 0 | `trip_id` | str | 8-char zero-padded string; folder name. Partition is NOT in the table - keep partition in a separate column when you concatenate. |
| 1 | `obs_idx` | int32 | 0-based row in that trip's .track |
| 2 | `arc_id` | int32 | 0-based row in that trip's .arcs. LOCAL to trip_id; (trip_id, arc_id) is the only valid arc key |
| 3 | `dist_m` | float32 | metres, point-to-segment, local projection |
| 4 | `heading_diff_deg` | float32 | deg 0..180; 90 when heading_valid=0 |
| 5 | `heading_cos` | float32 | cos(heading_diff); 0 when heading_valid=0 |
| 6 | `heading_valid` | int8 | 1 if GPS heading estimable (>= 6 m moved over +-2 fixes) |
| 7 | `along_frac` | float32 | 0..1 foot-point position along arc |
| 8 | `arc_len_m` | float32 | metres |
| 9 | `speed_mps` | float32 | m/s, same for all candidates of an obs |
| 10 | `dist_rank` | int16 | 0 = nearest candidate |
| 11 | `dist_minus_min_m` | float32 | metres above nearest candidate |
| 12 | `n_cand` | int16 | candidates kept for this obs |
| 13 | `dt_prev_s` | float32 | s since previous fix (0 first) |
| 14 | `dt_next_s` | float32 | s to next fix (0 last) |
| 15 | `has_reverse_arc` | int8 | 1 if opposite-direction arc exists |

**FEATURE_COLUMNS (model input order, 13):**
```python
['dist_m', 'heading_diff_deg', 'heading_cos', 'heading_valid', 'along_frac', 'arc_len_m', 'speed_mps', 'dist_rank', 'dist_minus_min_m', 'n_cand', 'dt_prev_s', 'dt_next_s', 'has_reverse_arc']
```
Import them instead of retyping: `from features import FEATURE_COLUMNS, KEY_COLUMNS`. Fit any scaler on TRAIN rows only. Group CV by `trip_id` (never split a trip).

## Label table schema (DEV ONLY - never present at inference)
| column | dtype | meaning |
|---|---|---|
| `trip_id` | str | as above |
| `obs_idx` | int | as above |
| `route_pos` | int | index into that trip's .route (monotone non-decreasing) |
| `ref_arc_id` | int | arc_id from .route at route_pos = development label |
| `align_dist_m` | float | GPS -> labelled arc distance, m |
| `nearest_route_dist_m` | float | diagnostic: distance to nearest route arc ignoring order |
| `label_ok` | int8 | 1 if align_dist_m <= 100 m; 0 = label suspect (reference route probably does not explain this fix) |

Join: `from align import attach_labels; df = attach_labels(cand, labels)` -> adds `y` (1 if arc_id == ref_arc_id) and `label_ok`; candidate rows are never dropped.

- **train**: 1,843,151 candidate rows; positive rows y=1: 185,277 (10.052%); observations with a positive in the candidate set: 185,277 of 208,417 -> candidate miss rate 0.1110 (natural set, before any training-only correction).
- **student_test**: 208,057 candidate rows; positive rows y=1: 19,133 (9.196%); observations with a positive in the candidate set: 19,133 of 19,160 -> candidate miss rate 0.0014 (natural set, before any training-only correction).

## Sample rows (real rows: train trip 00000003, obs 100 and 101; top 4 candidates each)
```
 trip_id  obs_idx  arc_id  dist_m  heading_diff_deg  heading_cos  heading_valid  along_frac  arc_len_m  speed_mps  dist_rank  dist_minus_min_m  n_cand  dt_prev_s  dt_next_s  has_reverse_arc  y  label_ok
00000003      100    1320   2.433        102.089996       -0.209              1       0.332   9.123000     13.789          0             0.000      19        1.0        1.0                1  0         1
00000003      100    6446   2.433         77.910004        0.209              1       0.668   9.123000     13.789          1             0.000      19        1.0        1.0                1  0         1
00000003      100   11362   2.965         10.667000        0.983              1       0.888  22.424999     13.789          2             0.531      19        1.0        1.0                0  0         1
00000003      100     226   3.883        100.727997       -0.186              1       1.000   8.806000     13.789          3             1.450      19        1.0        1.0                1  0         1
00000003      101   11361   3.321          4.784000        0.996              1       0.136  71.949997     10.596          0             0.000      16        1.0        1.0                0  1         1
00000003      101    2585   5.534        175.115997       -0.996              1       0.970 113.302002     10.596          1             2.214      16        1.0        1.0                0  0         1
00000003      101   11359   6.492        175.774002       -0.997              1       0.000   6.380000     10.596          2             3.172      16        1.0        1.0                0  0         1
00000003      101    1322  10.330          4.997000        0.996              1       1.000  11.431000     10.596          3             7.009      16        1.0        1.0                0  0         1
```
```
 trip_id  obs_idx  route_pos  ref_arc_id  align_dist_m  nearest_route_dist_m  label_ok
00000003      100          7        1322         3.883                 2.965         1
00000003      101          8       11361         3.321                 3.321         1
```

## Road-graph format (per trip; LOCAL ids)
- Raw: `<trip>.nodes` (lon lat; row = node_id), `<trip>.arcs` (from_node to_node; row = arc_id), whitespace/TAB separated, no header.
- Arcs are **directed** u -> v; a two-way street is two arcs (arcs with a reverse twin: see EDA). Some arcs are exact duplicates (same u,v, different arc_id) and some have zero length - kept, not removed.
- Never merge graphs of different trips: node_id and arc_id restart at 0 in each trip, and trips are in different cities.
- Load: `from common import load_trip, RoadGraph; trip = load_trip(root, 'train', '00000003'); g = RoadGraph(trip.nodes, trip.arcs)`  (`.route` is NOT opened unless `with_route=True`).
- `g.successors(arc_id)` -> np.array of arc_ids that can follow (leave the head node `g.v[arc_id]`); includes the U-turn (reverse arc). Use for HMM transitions; `g.reverse_id[arc_id]` is the opposite arc or -1.
- Arrays on `g` (length n_arcs): `u, v, ax, ay, bx, by` (tail/head node ids and local metres), `length_m`, `bearing_deg` (clockwise from north, NaN for zero-length). Node arrays: `nx, ny`.
- Projection: per-trip local equirectangular metres `g.proj.fwd(lon, lat) -> (x, y)`; keep the SAME projector for observations and arcs of a trip.

Example (trip 00000003): n_nodes=9138, n_arcs=18028; first 3 arcs:
```
 arc_id  u    v  length_m  bearing_deg  reverse_id
      0  0 7838    153.19        123.8          -1
      1  0 3594     12.44        205.5        7192
      2  0  154     14.26          3.8         261
```

## Known data caveats you must respect
- Observations with zero candidates exist (nothing within 50 m): see `s2_summary.csv`; Person 3 must handle an observation with no candidate state (gap/skip) in Viterbi.
- Reference route does not explain the GPS in these trips (>5% of fixes >100 m from aligned arc): train/00000002 (10%), train/00000033 (51%), train/00000054 (85%). `label_ok=0` marks fixes. Decide as a team how to treat them for TRAINING; keep them visible (and reported) in student_test evaluation.
- A repeated arc id in the route is normal (loops); labels are position-based. Duplicate-geometry twin arcs have identical features; only one of them is the label.
