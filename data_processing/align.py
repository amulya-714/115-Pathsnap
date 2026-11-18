"""align.py - DEVELOPMENT-ONLY. Align each GPS fix to the ordered reference route (.route).

Method: monotone dynamic programme over route POSITIONS j = 0..J-1 (not arc ids, so a route that
re-uses an arc - e.g. a loop driven twice - is handled).
  cost(i, j)   = min(dist(fix i, route arc j), cap_m)               [metres]
  transition   : stay on j (free) or advance j_prev -> j with  skip_pen_m * (j - j_prev - 1)
  start / end  : virtual positions -1 and J, same skip penalty (the first/last fix need not be on arc 0 / J-1)
Assumptions (all can fail - see s3 checks):
  A1 the vehicle traverses the reference route in the listed order, never going back in the list;
  A2 the reference route is complete enough that each fix's true arc is in the list;
  A3 GPS noise is small relative to the spacing between route arcs that are far apart in the list;
  A4 skipping a route arc (short arc, 1 Hz fix, or a time gap) is cheap but not free.
The output is a DEVELOPMENT LABEL, not physical ground truth.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from common import RoadGraph, seg_point_dist

ALIGN_DEFAULTS = dict(cap_m=100.0, skip_pen_m=1.0)


def align_to_route(x, y, g: RoadGraph, route, cap_m=100.0, skip_pen_m=1.0):
    n, J = len(x), len(route)
    ax, ay, bx, by = g.ax[route], g.ay[route], g.bx[route], g.by[route]
    lam = float(skip_pen_m)
    ramp = lam * np.arange(J)
    D = np.empty((n, J), dtype=np.float32)
    dmin_any = np.empty(n)                      # distance to nearest route arc ignoring order (diagnostic)
    dist_rows = np.empty((n, J), dtype=np.float32)
    for i in range(n):
        d, _, _, _ = seg_point_dist(x[i], y[i], ax, ay, bx, by)
        dist_rows[i] = d; dmin_any[i] = d.min()
    C = np.minimum(dist_rows, cap_m)
    D[0] = C[0] + ramp
    for i in range(1, n):
        prev = D[i - 1].astype(np.float64)
        cm = np.minimum.accumulate(prev - ramp)           # min_{k<=j} prev[k] - lam*k
        adv = np.full(J, np.inf); adv[1:] = lam * (np.arange(1, J) - 1) + cm[:-1]
        D[i] = C[i] + np.minimum(prev, adv)
    end = D[n - 1] + lam * (J - 1 - np.arange(J))
    pos = np.empty(n, dtype=np.int64); pos[-1] = int(np.argmin(end))
    for i in range(n - 1, 0, -1):
        j = pos[i]; prev = D[i - 1].astype(np.float64)
        best_k, best_v = j, prev[j]
        if j > 0:
            v = prev[:j] + lam * (j - 1 - np.arange(j))
            k = int(np.argmin(v))
            if v[k] < best_v - 1e-9:
                best_k, best_v = k, v[k]
        pos[i - 1] = best_k
    dist_al = dist_rows[np.arange(n), pos].astype(np.float64)
    return pd.DataFrame({"obs_idx": np.arange(n), "route_pos": pos, "ref_arc_id": np.asarray(route)[pos],
                         "align_dist_m": dist_al, "nearest_route_dist_m": dmin_any})


def attach_labels(cand: pd.DataFrame, labels: pd.DataFrame):
    """DEV ONLY. Adds y=1 where arc_id == ref_arc_id for that (trip_id, obs_idx). Keeps all candidates."""
    m = cand.merge(labels[["trip_id", "obs_idx", "ref_arc_id", "label_ok"]], on=["trip_id", "obs_idx"], how="left")
    m["y"] = (m["arc_id"] == m["ref_arc_id"]).astype("int8")
    return m.drop(columns=["ref_arc_id"])
