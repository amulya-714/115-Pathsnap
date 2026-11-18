# PathSnap - Person 1 pipeline (loader, candidates, features, reference alignment, EDA)

Data used: `Student folder/train` (80 trips) and `Student folder/student_test` (10 trips) only.
`common.PARTITIONS` does not list the evaluator partition, so it cannot be read by this code.

## Setup
    pip install -r requirements.txt
    # run from this folder; DATA = the unzipped "Student folder"
    DATA="path/to/Student folder"

## Stages (each stage depends only on earlier ones; ~4 s, 44 s, 32 s, 3 s, 4 s in my sandbox)
| stage | command | outputs in `out/` | what to check |
|---|---|---|---|
| 1 inventory | `python s1_inventory.py --data-root "$DATA" --out out` | inventory_trips.csv, inventory_partitions.csv, quality_flags.csv | "file-level errors: none"; 90 trips (80 + 10) |
| 2 candidates | `python s2_candidates.py --data-root "$DATA" --out out` | candidates_{train,student_test}.csv.gz, s2_summary.csv | 3 `[check]` lines pass; "table-contract errors: none". Quick form: add `--self-test` |
| 3 labels + miss rate | `python s3_align.py --data-root "$DATA" --out out --map-partition train --map-trip 00000003` | labels_*.csv.gz, s3_trip_checks.csv, s3_miss_rate.csv, maps/align_check_*.html | all_monotone True; "equal: True" in the cross-check; open the map |
| 4 EDA | `python s4_eda.py --data-root "$DATA" --out out --map-trip 00000003` | eda_summary.md, eda_figs.png, maps/eda_train_00000003.html | open the map and the PNG |
| 5 handoff | `python s5_handoff.py --data-root "$DATA" --out out` | HANDOFF.md | paste into the team chat |

Large trips: for a trip with >3000 fixes the map shows the first 3000 by default; use `--obs-range 6000 7500` (e.g. train/00000002 around its bad run starting at obs 6730).

## How to check the alignment on the Folium map (maps/align_check_*.html)
1. Layer control: toggle *Road network*, *Reference route*, *GPS fixes*, *Alignment links*.
2. Fixes are green (<=15 m from the aligned arc), orange (<=50 m), red (>50 m). Red clusters = the reference route does not explain the GPS there.
3. Along the GPS path, hover fixes: `route pos` must never decrease; the green route must be followed in order.
4. Look at junctions and parallel roads: links should end on the arc the vehicle is on, not on a neighbouring street.
5. Use trip 00000002 with `--obs-range 6500 7243` to see what a failing alignment looks like.

## Assumptions of the alignment (align.py) - these can be wrong
A1 vehicle follows the route list in order; A2 the route list is complete; A3 GPS noise << spacing of non-adjacent route arcs; A4 skipping an arc is cheap but not free (1 m per skipped arc).
Output is a **development label**, not verified ground truth. `label_ok=0` (aligned distance > 100 m) marks fixes where the reference route probably does not explain the GPS.

## Independence of inference features from .route
`features.py` never reads `.route`; `load_trip()` opens `.route` only with `with_route=True`; stage 2 checks (a) no `route` identifier in features.py (AST) and (b) identical candidate tables with the `.route` file deleted.

## Not executed in my sandbox
Folium could not be installed there (no network). `viz.py` data preparation ran on real data; the Folium calls were exercised only against a recording stub. Run stage 3/4 on your machine and look at the HTML before you rely on it.
