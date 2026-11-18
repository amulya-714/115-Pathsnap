"""features.py - candidate generation + features. INFERENCE-SAFE: uses track + nodes + arcs only.
This module must never import or read .route (checked by s2_candidates.py --self-test).

One row per (observation, candidate arc). Keys: trip_id, obs_idx, arc_id (arc_id is LOCAL to trip_id).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import Trip, RoadGraph, LocalProjector, seg_point_dist

# ---- the contract handed to Persons 2/3: order and dtypes are fixed here -------------------------
FEATURE_COLUMNS = [
    "dist_m",             # float32  point-to-segment distance, metres
    "heading_diff_deg",   # float32  smallest angle between GPS heading and arc bearing, 0..180 (90 if heading_valid=0)
    "heading_cos",        # float32  cos(heading_diff); 0 if heading_valid=0
    "heading_valid",      # int8     1 if GPS heading could be estimated (moved >= min_move_m), else 0
    "along_frac",         # float32  0..1 position of the foot point along the arc (0 = tail node, 1 = head node)
    "arc_len_m",          # float32  arc length, metres
    "speed_mps",          # float32  GPS speed estimate, metres/second (same for all candidates of an observation)
    "dist_rank",          # int16    0 = nearest candidate of this observation
    "dist_minus_min_m",   # float32  dist_m minus the nearest candidate's dist_m for this observation
    "n_cand",             # int16    number of candidates kept for this observation
    "dt_prev_s",          # float32  seconds since previous fix (0 for first fix)
    "dt_next_s",          # float32  seconds to next fix (0 for last fix)
    "has_reverse_arc",    # int8     1 if the opposite-direction arc exists in this trip's graph
]
KEY_COLUMNS = ["trip_id", "obs_idx", "arc_id"]
DTYPES = {"trip_id": "str", "obs_idx": "int32", "arc_id": "int32", "dist_m": "float32",
          "heading_diff_deg": "float32", "heading_cos": "float32", "heading_valid": "int8",
          "along_frac": "float32", "arc_len_m": "float32", "speed_mps": "float32", "dist_rank": "int16",
          "dist_minus_min_m": "float32", "n_cand": "int16", "dt_prev_s": "float32", "dt_next_s": "float32",
          "has_reverse_arc": "int8"}


@dataclass(frozen=True)
class CandConfig:
    radius_m: float = 50.0      # a-priori default; tune on TRAIN trips only (s3 prints the sweep)
    k_max: int = 50             # keep at most this many nearest candidates per observation
    piece_m: float = 50.0       # arcs are indexed in pieces <= this long (exactness for long arcs)
    min_move_m: float = 6.0     # heading valid only if GPS moved >= this far over the window (chosen on 26 sound TRAIN trips)
    half_window: int = 2        # heading/speed use fixes i-h .. i+h (chosen on TRAIN trips only)


def obs_kinematics(x, y, t, cfg: CandConfig):
    """Speed (m/s), heading (deg clockwise from north, NaN if not valid) per fix - track only."""
    n = len(x); i = np.arange(n)
    lo, hi = np.maximum(i - cfg.half_window, 0), np.minimum(i + cfg.half_window, n - 1)
    dx, dy, dt = x[hi] - x[lo], y[hi] - y[lo], t[hi] - t[lo]
    disp = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, disp / dt, 0.0)
    heading = np.where(disp >= cfg.min_move_m, np.degrees(np.arctan2(dx, dy)) % 360.0, np.nan)
    return speed, heading


def _index_pieces(g: RoadGraph, piece_m):
    """Midpoints of arc pieces (each <= piece_m long) -> (points Nx2, arc id per point)."""
    k = np.maximum(1, np.ceil(g.length_m / piece_m).astype(np.int64))
    arc = np.repeat(np.arange(g.n_arcs), k)
    start = np.cumsum(k) - k
    j = np.arange(len(arc)) - np.repeat(start, k)
    f = (j + 0.5) / np.repeat(k, k)
    px = g.ax[arc] + f * (g.bx[arc] - g.ax[arc]); py = g.ay[arc] + f * (g.by[arc] - g.ay[arc])
    return np.column_stack([px, py]), arc


def generate_candidates(trip: Trip, cfg: CandConfig = CandConfig(), graph: RoadGraph | None = None):
    """Returns (candidate DataFrame, graph, projected obs DataFrame). Never reads trip.route."""
    g = graph or RoadGraph(trip.nodes, trip.arcs)
    tr = trip.track
    x, y = g.proj.fwd(tr.lon.to_numpy(), tr.lat.to_numpy()); t = tr.t_s.to_numpy()
    speed, heading = obs_kinematics(x, y, t, cfg)
    n = len(x)
    pts, parc = _index_pieces(g, cfg.piece_m)
    tree = cKDTree(pts); otree = cKDTree(np.column_stack([x, y]))
    # any point within radius of an arc is within radius + piece/2 of that arc's nearest piece midpoint
    sdm = otree.sparse_distance_matrix(tree, cfg.radius_m + cfg.piece_m / 2, output_type="ndarray")
    oi, aj = sdm["i"].astype(np.int64), parc[sdm["j"].astype(np.int64)]
    key = np.unique(oi * g.n_arcs + aj)                        # de-duplicate (obs, arc) pairs
    oi, aj = key // g.n_arcs, key % g.n_arcs
    d, frac, _, _ = seg_point_dist(x[oi], y[oi], g.ax[aj], g.ay[aj], g.bx[aj], g.by[aj])
    keep = d <= cfg.radius_m
    oi, aj, d, frac = oi[keep], aj[keep], d[keep], frac[keep]
    df = pd.DataFrame({"obs_idx": oi, "arc_id": aj, "dist_m": d, "along_frac": frac})
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="stable")
    df["dist_rank"] = df.groupby("obs_idx").cumcount()
    df = df[df.dist_rank < cfg.k_max].copy()
    grp = df.groupby("obs_idx")["dist_m"]
    df["n_cand"] = grp.transform("size"); df["dist_minus_min_m"] = df.dist_m - grp.transform("min")
    o, a = df.obs_idx.to_numpy(), df.arc_id.to_numpy()
    # heading agreement
    h, ab = heading[o], g.bearing_deg[a]
    valid = ~np.isnan(h) & ~np.isnan(ab)
    diff = np.abs((np.nan_to_num(h) - np.nan_to_num(ab) + 180.0) % 360.0 - 180.0)
    df["heading_valid"] = valid.astype(np.int8)
    df["heading_diff_deg"] = np.where(valid, diff, 90.0)
    df["heading_cos"] = np.where(valid, np.cos(np.radians(diff)), 0.0)
    df["arc_len_m"] = g.length_m[a]; df["speed_mps"] = speed[o]
    dtp = np.r_[0.0, np.diff(t)]; dtn = np.r_[np.diff(t), 0.0]
    df["dt_prev_s"] = dtp[o]; df["dt_next_s"] = dtn[o]
    df["has_reverse_arc"] = (g.reverse_id[a] >= 0).astype(np.int8)
    df.insert(0, "trip_id", trip.trip_id)
    df = df.astype({k: v for k, v in DTYPES.items() if k != "trip_id"}).reset_index(drop=True)
    obs = pd.DataFrame({"obs_idx": np.arange(n), "lon": tr.lon.to_numpy(), "lat": tr.lat.to_numpy(), "t_s": t,
                        "x_m": x, "y_m": y, "speed_mps": speed, "heading_deg": heading})
    return df[KEY_COLUMNS + FEATURE_COLUMNS], g, obs
