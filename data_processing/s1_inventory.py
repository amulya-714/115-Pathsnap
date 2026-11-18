"""Stage 1 - dataset inventory + data-quality checks (permitted partitions only).

python s1_inventory.py --data-root "Student folder" --out out
Writes: out/inventory_trips.csv, out/inventory_partitions.csv, out/quality_flags.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from common import (PARTITIONS, list_trips, load_trip, RoadGraph, LocalProjector,
                    haversine_m, TripFormatError)

MARGIN_M = 1000.0      # GPS allowed this far outside the node bbox before it is flagged
MAX_SPEED_MPS = 50.0   # implied speed above this between consecutive fixes = jump
GAP_S = 5.0            # time step >= this = "large gap"


def trip_row(root, part, tid):
    row = dict(partition=part, trip_id=tid)
    d = Path(root) / part / tid
    for ext in ("track", "nodes", "arcs", "route"):
        p = d / f"{tid}.{ext}"
        row[f"size_{ext}_B"] = p.stat().st_size if p.exists() else np.nan
    flags = []
    t = load_trip(root, part, tid, with_route=True)
    tr, nodes, arcs, route = t.track, t.nodes, t.arcs, t.route
    n = len(tr)
    row.update(n_obs=n, n_nodes=len(nodes), n_arcs=len(arcs), n_route=len(route))

    # ---- GPS validity
    nan_cnt = int(tr.isna().sum().sum())
    bad_range = int(((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).sum())
    zero0 = int(((tr.lon == 0) & (tr.lat == 0)).sum())
    nb = dict(lo0=nodes[:, 0].min(), lo1=nodes[:, 0].max(), la0=nodes[:, 1].min(), la1=nodes[:, 1].max())
    mlat = MARGIN_M / 111_195.0
    mlon = mlat / np.cos(np.deg2rad(nodes[:, 1].mean()))
    outside = int(((tr.lon < nb["lo0"] - mlon) | (tr.lon > nb["lo1"] + mlon) |
                   (tr.lat < nb["la0"] - mlat) | (tr.lat > nb["la1"] + mlat)).sum())
    row.update(gps_nan=nan_cnt, gps_bad_range=bad_range, gps_zero_zero=zero0, gps_outside_node_bbox=outside,
               lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max())
    # ---- timestamps
    dt = np.diff(tr.t_s.to_numpy())
    row.update(t_first=tr.t_s.iloc[0], t_last=tr.t_s.iloc[-1],
               dt_frac_1s=float(np.mean(dt == 1.0)) if len(dt) else np.nan,
               dt_max_s=dt.max() if len(dt) else np.nan,
               n_dt_nonpositive=int((dt <= 0).sum()), n_dt_gap_ge5=int((dt >= GAP_S).sum()),
               n_dt_not_integer=int((np.abs(dt - np.round(dt)) > 1e-6).sum()))
    # ---- geometry / jumps
    proj = LocalProjector(nodes[:, 0].mean(), nodes[:, 1].mean())
    x, y = proj.fwd(tr.lon.to_numpy(), tr.lat.to_numpy())
    step = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(dt > 0, step / dt, np.nan)
    row.update(track_len_km=step.sum() / 1000, duration_s=dt.sum(),
               n_stationary_steps=int((step < 0.5).sum()),
               speed_p50=np.nanmedian(spd), speed_max=np.nanmax(spd),
               n_speed_jumps=int((spd > MAX_SPEED_MPS).sum()),
               n_dup_coords=int((step == 0).sum()),
               bbox_diag_km=haversine_m(tr.lon.min(), tr.lat.min(), tr.lon.max(), tr.lat.max()) / 1000)
    # projection check against haversine on consecutive fixes
    hv = haversine_m(tr.lon.values[:-1], tr.lat.values[:-1], tr.lon.values[1:], tr.lat.values[1:])
    m = hv > 1
    row["proj_max_rel_err"] = float(np.max(np.abs(step[m] - hv[m]) / hv[m])) if m.any() else np.nan

    # ---- graph
    g = RoadGraph(nodes, arcs, proj)
    N = g.n_nodes
    A = coo_matrix((np.ones(len(arcs)), (g.u, g.v)), shape=(N, N)).tocsr()
    nw, lw = connected_components(A, directed=True, connection="weak")
    ns, ls = connected_components(A, directed=True, connection="strong")
    used = np.zeros(N, bool); used[g.u] = True; used[g.v] = True
    cw = np.bincount(lw); cs = np.bincount(ls)
    key = g.u * N + g.v
    row.update(isolated_nodes=int((~used).sum()), weak_components=int(nw),
               largest_weak_frac=cw.max() / N, strong_components=int(ns), largest_strong_frac=cs.max() / N,
               self_loops=int((g.u == g.v).sum()), duplicate_arcs=int(len(key) - len(np.unique(key))),
               zero_len_arcs=int((g.length_m == 0).sum()),
               arcs_with_reverse_frac=float((g.reverse_id >= 0).mean()),
               arc_len_p50_m=float(np.median(g.length_m)), arc_len_max_m=float(g.length_m.max()),
               out_deg_max=int(g.out_deg.max()))
    # ---- reference route (development-only)
    in_range = (route >= 0) & (route < len(arcs))
    row["route_ids_out_of_range"] = int((~in_range).sum())
    if in_range.all() and len(route) > 1:
        conn = g.v[route[:-1]] == g.u[route[1:]]
        row["route_breaks"] = int((~conn).sum())
        row["route_repeated_arcs"] = int(len(route) - len(np.unique(route)))
        row["route_len_km"] = float(g.length_m[route].sum() / 1000)
        row["route_in_one_weak_comp"] = bool(len(np.unique(lw[g.u[route]])) == 1)
    else:
        row.update(route_breaks=np.nan, route_repeated_arcs=np.nan, route_len_km=np.nan, route_in_one_weak_comp=False)

    # ---- flags
    def f(c, msg):
        if c: flags.append(msg)
    f(nan_cnt, f"{nan_cnt} NaN in track"); f(bad_range, "lon/lat out of range")
    f(zero0, "(0,0) fixes"); f(outside, f"{outside} fixes >{MARGIN_M:.0f} m outside node bbox")
    f(row["n_dt_nonpositive"], "non-monotonic/duplicate timestamps"); f(row["n_dt_gap_ge5"], f"{row['n_dt_gap_ge5']} gaps >= {GAP_S}s")
    f(row["n_speed_jumps"], f"{row['n_speed_jumps']} jumps > {MAX_SPEED_MPS} m/s")
    f(row["isolated_nodes"], f"{row['isolated_nodes']} isolated nodes")
    f(row["weak_components"] > 1, f"{row['weak_components']} weak components")
    f(row["duplicate_arcs"], f"{row['duplicate_arcs']} duplicate arcs"); f(row["zero_len_arcs"], f"{row['zero_len_arcs']} zero-length arcs")
    f(row["route_ids_out_of_range"], "route ids out of range")
    rb, rr = row["route_breaks"], row["route_repeated_arcs"]
    f(rb == rb and rb > 0, f"route has {rb} disconnected steps")
    f(rr == rr and rr > 0, f"{rr} repeated route arcs")
    row["quality_flags"] = "; ".join(flags)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Student folder"); ap.add_argument("--out", default="out")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    rows, errs = [], []
    for part in PARTITIONS:
        for tid in list_trips(a.data_root, part):
            try: rows.append(trip_row(a.data_root, part, tid))
            except (TripFormatError, ValueError) as e: errs.append((part, tid, str(e)))
    inv = pd.DataFrame(rows); inv.to_csv(out / "inventory_trips.csv", index=False)
    inv[inv["quality_flags"] != ""][["partition", "trip_id", "quality_flags"]].to_csv(out / "quality_flags.csv", index=False)
    agg = inv.groupby("partition").agg(
        trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_min=("n_obs", "min"), obs_median=("n_obs", "median"),
        obs_max=("n_obs", "max"), nodes_median=("n_nodes", "median"), nodes_max=("n_nodes", "max"),
        arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"), route_len_median=("n_route", "median"),
        gps_nan=("gps_nan", "sum"), gps_bad_range=("gps_bad_range", "sum"), outside_bbox=("gps_outside_node_bbox", "sum"),
        nonpositive_dt=("n_dt_nonpositive", "sum"), gaps_ge5=("n_dt_gap_ge5", "sum"), trips_with_gap_ge5=("n_dt_gap_ge5", lambda s: int((s > 0).sum())),
        dt_max_s=("dt_max_s", "max"), speed_jumps=("n_speed_jumps", "sum"),
        isolated_nodes=("isolated_nodes", "sum"), trips_weak_comp_gt1=("weak_components", lambda s: int((s > 1).sum())),
        trips_strong_comp_gt1=("strong_components", lambda s: int((s > 1).sum())),
        dup_arcs=("duplicate_arcs", "sum"), zero_len_arcs=("zero_len_arcs", "sum"),
        route_breaks=("route_breaks", "sum"), route_repeated=("route_repeated_arcs", "sum"),
        proj_max_rel_err=("proj_max_rel_err", "max"))
    agg.T.to_csv(out / "inventory_partitions.csv")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    print(agg.T.to_string()); print("\nfile-level errors:", errs or "none")
    print("trips with any flag:", int((inv["quality_flags"] != "").sum()), "of", len(inv))
    print(inv["quality_flags"].replace("", np.nan).dropna().str.split("; ").explode().str.replace(r"\d+", "N", regex=True).value_counts().to_string())
    return 0

if __name__ == "__main__":
    sys.exit(main())
