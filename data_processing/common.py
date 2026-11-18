"""common.py - loader, local metric projection, directed road graph.

Design rules (from the PathSnap documents + what the files actually look like):
  * Raw files are headerless, whitespace (TAB) separated; .track has CRLF endings.
  * .track = lon, lat, t_s   .nodes = lon, lat   .arcs = from_node, to_node   .route = arc_id
  * node_id = 0-based row in .nodes; arc_id = 0-based row in .arcs. BOTH ARE LOCAL TO ONE TRIP.
    Never concatenate nodes/arcs of two trips; always carry trip_id next to arc_id.
  * Arcs are DIRECTED (u -> v). A two-way road appears as two arcs.
  * Only the partitions in PARTITIONS can be read. Evaluator trips may be loaded for inference,
    but their reference routes must not be read during prediction.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

EARTH_R_M = 6371008.8
PARTITIONS = ("train", "student_test", "evaluator_test")
_TRIP_RE = re.compile(r"^\d{8}$")


class TripFormatError(Exception):
    pass


def list_trips(root, partition):
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}, got {partition!r}")
    base = Path(root) / partition
    return sorted(p.name for p in base.iterdir() if p.is_dir() and _TRIP_RE.match(p.name))


def _read(path, names, dtype):
    if not path.exists():
        raise TripFormatError(f"missing file {path.name}")
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)
    except pd.errors.EmptyDataError:
        raise TripFormatError(f"empty file {path.name}")
    except ValueError as e:
        raise TripFormatError(f"{path.name}: {e}")
    return df


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame        # lon, lat, t_s  (float64), row order == obs_idx
    nodes: np.ndarray          # (N,2) float64 lon, lat ; row == node_id
    arcs: np.ndarray           # (M,2) int64  u, v    ; row == arc_id
    route: np.ndarray | None   # (R,) int64 arc_ids, chronological; None unless with_route=True


def load_trip(root, partition, trip_id, with_route=False) -> Trip:
    """with_route=False is the INFERENCE-SAFE default: the .route file is never opened."""
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    d = Path(root) / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t_s"], "float64")
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], "float64").to_numpy()
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], "int64").to_numpy()
    route = None
    if with_route:
        route = _read(d / f"{trip_id}.route", ["arc_id"], "int64")["arc_id"].to_numpy()
    return Trip(trip_id, partition, track, nodes, arcs, route)


class LocalProjector:
    """Equirectangular projection about (lon0, lat0) -> metres (x east, y north).
    Valid because each trip spans a few km; error is measured in stage 1 against haversine."""
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_R_M * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R_M

    def fwd(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def inv(self, x, y):
        return self.lon0 + np.asarray(x) / self.kx, self.lat0 + np.asarray(y) / self.ky


def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


class RoadGraph:
    """Directed road graph for ONE trip, in local metres.
    arc a goes node u[a] -> v[a].  arc_id == row in .arcs (local to the trip)."""
    def __init__(self, nodes_lonlat, arcs, proj: LocalProjector | None = None):
        n = len(nodes_lonlat)
        self.n_nodes, self.n_arcs = n, len(arcs)
        if proj is None:
            proj = LocalProjector(nodes_lonlat[:, 0].mean(), nodes_lonlat[:, 1].mean())
        self.proj = proj
        self.nx, self.ny = proj.fwd(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.u, self.v = arcs[:, 0].astype(np.int64), arcs[:, 1].astype(np.int64)
        if self.n_arcs and (self.u.min() < 0 or self.v.min() < 0 or max(self.u.max(), self.v.max()) >= n):
            raise TripFormatError("arcs reference node ids outside .nodes")
        self.ax, self.ay = self.nx[self.u], self.ny[self.u]
        self.bx, self.by = self.nx[self.v], self.ny[self.v]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.length_m = np.hypot(dx, dy)
        bearing = np.degrees(np.arctan2(dx, dy)) % 360.0           # clockwise from north
        self.bearing_deg = np.where(self.length_m > 0, bearing, np.nan)
        # successor lookup: arcs leaving node v[a]
        self._order = np.argsort(self.u, kind="stable")
        self._start = np.searchsorted(self.u[self._order], np.arange(n + 1))
        self.out_deg = np.diff(self._start)                        # out-degree per node
        # reverse arc (v->u) if one exists, else -1 (lowest arc_id wins when duplicated)
        key = self.u * n + self.v
        rkey = self.v * n + self.u
        srt = np.argsort(key, kind="stable"); sk = key[srt]
        pos = np.searchsorted(sk, rkey)
        pos_c = np.minimum(pos, len(sk) - 1)
        found = (sk[pos_c] == rkey) & (len(sk) > 0)
        self.reverse_id = np.where(found, srt[pos_c], -1)

    def successors(self, arc_id):
        """arc ids that can follow arc_id (leave its head node). Includes the reverse arc (U-turn)."""
        h = self.v[arc_id]
        return self._order[self._start[h]:self._start[h + 1]]


def seg_point_dist(px, py, ax, ay, bx, by):
    """Distance (m), along-fraction in [0,1] and foot point from points P to segments A->B (all same shape)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    fx, fy = ax + t * dx, ay + t * dy
    return np.hypot(px - fx, py - fy), t, fx, fy
