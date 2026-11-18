# PathSnap EDA summary (Person 1) - generated from actual outputs

Partitions: **train** (80 trips) = fitting/CV; **student_test** (10 trips) = development comparison only. Evaluator data not present or used.

## Inventory
| item | train | student_test |
|---|---|---|
| trips | 80 | 10 |
| GPS observations (total) | 208417 | 19160 |
| obs / trip median | 1529.0 | 1420.0 |
| obs / trip min-max | 361-19313 | 262-3958 |
| nodes / trip median | 9608.0 | 6596.5 |
| arcs / trip median | 19526.5 | 13145.0 |
| arcs / trip max | 307392 | 56027 |
| route arcs / trip median | 280.0 | 250.0 |
| bbox diagonal km (median) | 9.5 | 7.2 |

All 90 trips have all four files; missing values in track/nodes/arcs/route: **0**; invalid lon/lat: **0**; fixes outside node bbox+1 km: **0**.

## Timestamps and sampling (pooled over consecutive fixes)
```
partition              train  student_test
steps            208337.0000    19150.0000
dt_eq_1               0.9650        0.9862
dt_eq_2               0.0208        0.0066
dt_3_4                0.0070        0.0047
dt_ge5                0.0070        0.0025
dt_ge60               0.0003        0.0001
speed_p50_mps        11.1166       14.1424
speed_p95_mps        33.3096       31.3891
speed_gt30_frac       0.0751        0.0671
```


## Gaps per trip
| item | train | student_test |
|---|---|---|
| trips with a gap >= 5 s | 45 | 2 |
| gaps >= 5 s (total) | 1468 | 48 |
| largest gap, s | 153064.0 | 210.0 |
| implied-speed jumps > 50 m/s (total) | 1524 | 8 |

## Road graph (per trip)
| item | train | student_test |
|---|---|---|
| trips with >1 weakly connected component | 79 | 10 |
| largest weak component, share of nodes (median) | 0.981 | 0.971 |
| largest weak component share (min) | 0.693 | 0.874 |
| largest strongly connected share (median) | 0.964 | 0.961 |
| isolated nodes (total) | 0 | 0 |
| duplicate arcs (total) | 1775 | 128 |
| zero-length arcs (total) | 100 | 6 |
| arcs with a reverse arc (median share) | 0.951 | 0.933 |
| arc length median of medians, m | 24.5 | 31.9 |
| longest arc, m | 9297.0 | 7007.0 |

## Reference route (development reference, NOT verified truth)
| item | train | student_test |
|---|---|---|
| disconnected consecutive route arcs (total) | 13 | 1 |
| repeated arcs inside a route (total) | 67 | 255 |
| trips with route entirely in one weak component | 80 | 10 |

## Alignment quality (route positions aligned by monotone DP)
| item | train | student_test |
|---|---|---|
| trips with monotone alignment | 80 | 10 |
| aligned distance, median of trip medians (m) | 4.63 | 4.75 |
| trips with >5% of fixes > 100 m from aligned arc | 3 | 0 |
| route-position coverage, median | 0.96 | 0.948 |
| route-position coverage, min | 0.713 | 0.849 |
| median |heading - aligned arc bearing|, median of trips (deg) | 6.0 | 5.7 |

Trips where the reference route does not explain the GPS track (>5% of fixes >100 m from the aligned arc):

```
partition  trip_id  n_obs  frac_gt100  track_km  route_km
    train 00000002   7243       0.099    55.880     7.965
    train 00000033   5844       0.510    47.832     6.379
    train 00000054  19313       0.853   169.156    31.361
```

## Candidate generation at default (radius 50 m, k_max 50)
| item | train | student_test |
|---|---|---|
| candidate rows | 1843151 | 208057 |
| mean candidates / observation | 8.84 | 10.86 |
| observations with zero candidates | 18229 | 0 |
| candidate miss rate, all obs (natural set, before any correction) | 0.1110 | 0.0014 |
| candidate miss rate, label_ok obs only | 0.0138 | 0.0005 |
