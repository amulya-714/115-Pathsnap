"""Stage 4 - EDA summary (actual numbers) + figures + ONE Folium map (GPS, road geometry, development reference route).

python s4_eda.py --data-root "Student folder" --out out --map-partition train --map-trip 00000003
Needs outputs of stages 1-3. Writes out/eda_summary.md, out/eda_figs.png, out/maps/eda_<part>_<trip>.html
Every number is labelled with its partition. Nothing here is used to fit a model.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import PARTITIONS, list_trips, load_trip, LocalProjector
from features import generate_candidates, CandConfig
from align import align_to_route


def pooled_track_stats(root):
    rows = []
    for part in PARTITIONS:
        dts, sps = [], []
        for tid in list_trips(root, part):
            t = load_trip(root, part, tid); tr = t.track
            pr = LocalProjector(t.nodes[:, 0].mean(), t.nodes[:, 1].mean()); x, y = pr.fwd(tr.lon.to_numpy(), tr.lat.to_numpy())
            dt = np.diff(tr.t_s.to_numpy()); st = np.hypot(np.diff(x), np.diff(y))
            dts.append(dt); sps.append(np.where(dt > 0, st / np.maximum(dt, 1e-9), np.nan)[dt <= 5])
        dt = np.concatenate(dts); sp = np.concatenate(sps)
        rows.append(dict(partition=part, steps=len(dt), dt_eq_1=(dt == 1).mean(), dt_eq_2=(dt == 2).mean(), dt_3_4=((dt >= 3) & (dt < 5)).mean(),
                         dt_ge5=(dt >= 5).mean(), dt_ge60=(dt >= 60).mean(), speed_p50_mps=np.nanmedian(sp), speed_p95_mps=np.nanpercentile(sp, 95),
                         speed_gt30_frac=np.nanmean(sp > 30)))
        globals().setdefault("_DT", {})[part] = dt
    return pd.DataFrame(rows).set_index("partition")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Student folder"); ap.add_argument("--out", default="out")
    ap.add_argument("--map-partition", default="train"); ap.add_argument("--map-trip", default=None); ap.add_argument("--obs-range", nargs=2, type=int)
    a = ap.parse_args(); out = Path(a.out); (out / "maps").mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(out / "inventory_trips.csv", dtype={"trip_id": str}); chk = pd.read_csv(out / "s3_trip_checks.csv", dtype={"trip_id": str})
    s2 = pd.read_csv(out / "s2_summary.csv", dtype={"trip_id": str}); miss = pd.read_csv(out / "s3_miss_rate.csv")
    pool = pooled_track_stats(a.data_root)
    P = lambda df, col, fn: {p: fn(df[df.partition == p][col]) for p in PARTITIONS}
    L = []
    w = L.append
    w("# PathSnap EDA summary (Person 1) - generated from actual outputs\n")
    w("Partitions: **train** (80 trips) = fitting/CV; **student_test** (10 trips) = development comparison only. Evaluator data not present or used.\n")
    w("## Inventory\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("trips", "trip_id", len), ("GPS observations (total)", "n_obs", np.sum), ("obs / trip median", "n_obs", np.median),
                         ("obs / trip min-max", "n_obs", lambda s: f"{s.min()}-{s.max()}"), ("nodes / trip median", "n_nodes", np.median),
                         ("arcs / trip median", "n_arcs", np.median), ("arcs / trip max", "n_arcs", np.max), ("route arcs / trip median", "n_route", np.median),
                         ("bbox diagonal km (median)", "bbox_diag_km", lambda s: round(float(np.median(s)), 1))]:
        d = P(inv, col, fn); w(f"| {lab} | {d['train']} | {d['student_test']} |")
    w(f"\nAll 90 trips have all four files; missing values in track/nodes/arcs/route: **{int(inv.gps_nan.sum())}**; invalid lon/lat: **{int(inv.gps_bad_range.sum() + inv.gps_zero_zero.sum())}**; "
      f"fixes outside node bbox+1 km: **{int(inv.gps_outside_node_bbox.sum())}**.\n")
    w("## Timestamps and sampling (pooled over consecutive fixes)\n```\n" + pool.round(4).T.to_string() + "\n```")
    g = inv.groupby("partition")
    w("\n\n## Gaps per trip\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("trips with a gap >= 5 s", "n_dt_gap_ge5", lambda s: int((s > 0).sum())), ("gaps >= 5 s (total)", "n_dt_gap_ge5", np.sum), ("largest gap, s", "dt_max_s", np.max),
                         ("implied-speed jumps > 50 m/s (total)", "n_speed_jumps", np.sum)]:
        d = P(inv, col, fn); w(f"| {lab} | {d['train']} | {d['student_test']} |")
    w("\n## Road graph (per trip)\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("trips with >1 weakly connected component", "weak_components", lambda s: int((s > 1).sum())),
                         ("largest weak component, share of nodes (median)", "largest_weak_frac", lambda s: round(float(np.median(s)), 3)),
                         ("largest weak component share (min)", "largest_weak_frac", lambda s: round(float(np.min(s)), 3)),
                         ("largest strongly connected share (median)", "largest_strong_frac", lambda s: round(float(np.median(s)), 3)),
                         ("isolated nodes (total)", "isolated_nodes", np.sum), ("duplicate arcs (total)", "duplicate_arcs", np.sum),
                         ("zero-length arcs (total)", "zero_len_arcs", np.sum), ("arcs with a reverse arc (median share)", "arcs_with_reverse_frac", lambda s: round(float(np.median(s)), 3)),
                         ("arc length median of medians, m", "arc_len_p50_m", lambda s: round(float(np.median(s)), 1)), ("longest arc, m", "arc_len_max_m", lambda s: round(float(np.max(s)), 0))]:
        d = P(inv, col, fn); w(f"| {lab} | {d['train']} | {d['student_test']} |")
    w("\n## Reference route (development reference, NOT verified truth)\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("disconnected consecutive route arcs (total)", "route_breaks", np.sum), ("repeated arcs inside a route (total)", "route_repeated_arcs", np.sum),
                         ("trips with route entirely in one weak component", "route_in_one_weak_comp", lambda s: int(s.sum()))]:
        d = P(inv, col, fn); w(f"| {lab} | {d['train']} | {d['student_test']} |")
    c = chk.copy()
    w("\n## Alignment quality (route positions aligned by monotone DP)\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("trips with monotone alignment", "monotone", lambda s: int(s.sum())), ("aligned distance, median of trip medians (m)", "dist_p50", lambda s: round(float(np.median(s)), 2)),
                         ("trips with >5% of fixes > 100 m from aligned arc", "frac_gt100", lambda s: int((s > .05).sum())),
                         ("route-position coverage, median", "route_pos_coverage", lambda s: round(float(np.median(s)), 3)),
                         ("route-position coverage, min", "route_pos_coverage", lambda s: round(float(np.min(s)), 3)),
                         ("median |heading - aligned arc bearing|, median of trips (deg)", "heading_median_deg", lambda s: round(float(np.median(s)), 1))]:
        d = P(c, col, fn); w(f"| {lab} | {d['train']} | {d['student_test']} |")
    sus = c[c.frac_gt100 > .05][["partition", "trip_id", "n_obs", "frac_gt100", "track_km", "route_km"]]
    w("\nTrips where the reference route does not explain the GPS track (>5% of fixes >100 m from the aligned arc):\n")
    w("```\n" + sus.round(3).to_string(index=False) + "\n```")
    w("\n## Candidate generation at default (radius 50 m, k_max 50)\n| item | train | student_test |\n|---|---|---|")
    for lab, col, fn in [("candidate rows", "n_rows", np.sum), ("mean candidates / observation", None, None), ("observations with zero candidates", "obs_zero_cand", np.sum)]:
        if col: d = P(s2, col, fn)
        else: d = {p: round(float(s2[s2.partition == p].n_rows.sum() / s2[s2.partition == p].n_obs.sum()), 2) for p in PARTITIONS}
        w(f"| {lab} | {d['train']} | {d['student_test']} |")
    m = miss[(miss.radius_m == 50) & (miss.k_max == 50)].set_index("partition")
    w(f"| candidate miss rate, all obs (natural set, before any correction) | {m.loc['train','miss_all']:.4f} | {m.loc['student_test','miss_all']:.4f} |")
    w(f"| candidate miss rate, label_ok obs only | {m.loc['train','miss_label_ok']:.4f} | {m.loc['student_test','miss_label_ok']:.4f} |")
    (out / "eda_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for p, col in zip(PARTITIONS, ["tab:blue", "tab:orange"]):
        ax[0, 0].hist(inv[inv.partition == p].n_obs, bins=30, alpha=.6, color=col, label=p)
        ax[0, 1].hist(np.log10(inv[inv.partition == p].n_arcs), bins=20, alpha=.6, color=col, label=p)
        dt = globals()["_DT"][p]; ax[0, 2].hist(np.clip(dt, 0, 20), bins=np.arange(0, 21, 1), alpha=.6, color=col, label=p, density=True)
    ax[0, 0].set(title="GPS observations per trip", xlabel="observations"); ax[0, 1].set(title="road arcs per trip", xlabel="log10(arcs)")
    ax[0, 2].set(title="time step between fixes (clipped at 20 s)", xlabel="seconds", yscale="log"); [x.legend() for x in ax[0]]
    ax[1, 0].scatter(chk.route_km, chk.track_km, c=np.where(chk.partition == "train", "tab:blue", "tab:orange"), s=14)
    mx = max(chk.route_km.max(), chk.track_km.max()); ax[1, 0].plot([0, mx], [0, mx], "k--", lw=.8); ax[1, 0].set(title="GPS track length vs reference route length", xlabel="route km", ylabel="track km", xscale="log", yscale="log")
    ax[1, 1].hist(np.clip(chk.frac_gt100 * 100, 0, 100), bins=20, color="tab:red"); ax[1, 1].set(title="% of fixes >100 m from aligned arc, per trip (both partitions)", xlabel="%", yscale="log")
    ax[1, 2].scatter(inv.n_arcs, inv.n_obs, s=12, c=np.where(inv.partition == "train", "tab:blue", "tab:orange")); ax[1, 2].set(title="graph size vs track size", xlabel="arcs", ylabel="observations", xscale="log", yscale="log")
    fig.suptitle("PathSnap EDA (blue = train, orange = student_test)"); fig.tight_layout(); fig.savefig(out / "eda_figs.png", dpi=110); print("wrote", out / "eda_figs.png")

    tid = a.map_trip or list_trips(a.data_root, a.map_partition)[2]
    t = load_trip(a.data_root, a.map_partition, tid, with_route=True)
    _, g_, obs = generate_candidates(t, CandConfig())
    lab = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g_, t.route)
    try:
        from viz import map_layers, write_map
        ML = map_layers(t, g_, obs, lab, tuple(a.obs_range) if a.obs_range else (0, min(len(obs), 3000)))
        fn = out / "maps" / f"eda_{a.map_partition}_{tid}.html"
        write_map(ML, fn, f"EDA {a.map_partition}/{tid}", False)
        print("wrote", fn, "| gps", len(ML["gps"]), "road arcs", len(ML["roads"]), "route arcs", len(ML["route"]))
    except ImportError:
        print("folium not installed -> run: pip install folium   (map skipped)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
