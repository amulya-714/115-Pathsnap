"""Stage 3 - DEVELOPMENT labels (reference-route alignment), alignment checks, candidate miss rate, check map.

python s3_align.py --data-root "Student folder" --out out
python s3_align.py ... --map-partition train --map-trip 00000003            # + Folium alignment-check map
python s3_align.py ... --map-partition train --map-trip 00000002 --obs-range 0 1500
Needs out/candidates_<partition>.csv.gz from stage 2 for the default-config cross-check (skipped if absent).
Writes: out/labels_<part>.csv.gz, out/s3_trip_checks.csv, out/s3_miss_rate.csv, out/maps/align_check_*.html
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from common import PARTITIONS, list_trips, load_trip
from features import CandConfig, generate_candidates
from align import align_to_route, ALIGN_DEFAULTS

RADII = [30, 50, 75, 100, 150]
KS = [10, 20, 50, 10**9]
LABEL_OK_M = 100.0       # fixed a priori: aligned arc farther than this => label flagged suspect (label_ok = 0)
DEFAULT = CandConfig()   # radius 50, k_max 50 : the miss rate reported "before any correction"


def longest_run(mask):
    if not mask.any(): return 0, -1
    d = np.diff(np.r_[0, mask.astype(int), 0]); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax(e - s)); return int(e[k] - s[k]), int(s[k])


def process_trip(root, part, tid):
    t = load_trip(root, part, tid, with_route=True)
    big, g, obs = generate_candidates(t, CandConfig(radius_m=max(RADII), k_max=10**9))   # inference-safe call
    L = align_to_route(obs.x_m.to_numpy(), obs.y_m.to_numpy(), g, t.route, **ALIGN_DEFAULTS)
    L.insert(0, "trip_id", tid); L["label_ok"] = (L.align_dist_m <= LABEL_OK_M).astype("int8")
    ref = big.merge(L[["obs_idx", "ref_arc_id"]], left_on=["obs_idx", "arc_id"], right_on=["obs_idx", "ref_arc_id"])
    L["ref_rank"] = L.obs_idx.map(ref.set_index("obs_idx").dist_rank).astype("float64")
    for r in RADII:
        L[f"n_within_{r}"] = L.obs_idx.map((big.dist_m <= r).groupby(big.obs_idx).sum()).fillna(0).astype("int32")
    p = L.route_pos.to_numpy(); d = L.align_dist_m.to_numpy()
    hd = np.abs((obs.heading_deg.to_numpy() - g.bearing_deg[L.ref_arc_id.to_numpy()] + 180) % 360 - 180)
    ok = (~np.isnan(hd)) & (d <= LABEL_OK_M) & (obs.speed_mps.to_numpy() >= 2.0)
    run, run_at = longest_run(d > LABEL_OK_M)
    ex = d - L.nearest_route_dist_m.to_numpy()
    twin = int(sum(1 for a in set(t.route.tolist()) if (g.u == g.u[a]).any() and ((g.u == g.u[a]) & (g.v == g.v[a])).sum() > 1)) if g.n_arcs < 60000 else -1
    chk = dict(partition=part, trip_id=tid, n_obs=len(L), n_route=len(t.route),
               monotone=bool((np.diff(p) >= 0).all()), route_pos_coverage=len(np.unique(p)) / len(t.route),
               dist_p50=np.median(d), dist_p95=np.percentile(d, 95), dist_max=d.max(),
               frac_gt50=(d > 50).mean(), frac_gt100=(d > LABEL_OK_M).mean(),
               frac_forced_worse_10m=(ex > 10).mean(),
               heading_n=int(ok.sum()), heading_median_deg=float(np.median(hd[ok])) if ok.any() else np.nan,
               heading_frac_gt90=float((hd[ok] > 90).mean()) if ok.any() else np.nan,
               longest_bad_run=run, longest_bad_run_start=run_at,
               track_km=float(np.hypot(np.diff(obs.x_m), np.diff(obs.y_m)).sum() / 1000), route_km=float(g.length_m[t.route].sum() / 1000),
               route_arcs_with_geometric_twin=twin)
    return L, chk, t, g, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Student folder"); ap.add_argument("--out", default="out")
    ap.add_argument("--map-partition"); ap.add_argument("--map-trip"); ap.add_argument("--obs-range", nargs=2, type=int)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True); (out / "maps").mkdir(exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    allL, checks = [], []
    for part in PARTITIONS:
        Ls = []
        for tid in list_trips(a.data_root, part):
            L, chk, t, g, obs = process_trip(a.data_root, part, tid)
            Ls.append(L); checks.append(chk)
            if a.map_trip == tid and a.map_partition == part:
                from viz import map_layers, write_map
                ML = map_layers(t, g, obs, L, tuple(a.obs_range) if a.obs_range else (0, min(len(obs), 3000)))
                fn = out / "maps" / f"align_check_{part}_{tid}.html"
                try:
                    write_map(ML, fn, f"{part}/{tid} alignment check", True)
                    print("wrote", fn, "| road arcs drawn:", len(ML["roads"]), "truncated:", ML["road_arcs_truncated"])
                except ImportError:
                    print("folium not installed -> run: pip install folium   (map skipped)")
        P = pd.concat(Ls, ignore_index=True); P["partition"] = part; allL.append(P)
        P[["trip_id", "obs_idx", "route_pos", "ref_arc_id", "align_dist_m", "nearest_route_dist_m", "label_ok"]].to_csv(
            out / f"labels_{part}.csv.gz", index=False, compression="gzip", float_format="%.4f")
    A = pd.concat(allL, ignore_index=True); C = pd.DataFrame(checks); C.to_csv(out / "s3_trip_checks.csv", index=False)

    print("\n=== ALIGNMENT CHECKS (per partition) ===")
    g_ = C.groupby("partition").agg(trips=("trip_id", "count"), all_monotone=("monotone", "all"),
        cov_median=("route_pos_coverage", "median"), cov_min=("route_pos_coverage", "min"),
        trips_frac_gt100_over_5pct=("frac_gt100", lambda s: int((s > .05).sum())),
        trips_forced_worse_over_5pct=("frac_forced_worse_10m", lambda s: int((s > .05).sum())),
        heading_median_of_trip_medians=("heading_median_deg", "median"), trips_heading_gt90_over_20pct=("heading_frac_gt90", lambda s: int((s > .2).sum())))
    print(g_.T.to_string())
    d = A.groupby("partition").agg(obs=("obs_idx", "size"), dist_p50=("align_dist_m", "median"),
        dist_p95=("align_dist_m", lambda s: s.quantile(.95)), gt50=("align_dist_m", lambda s: (s > 50).mean()),
        gt100=("align_dist_m", lambda s: (s > LABEL_OK_M).mean()), label_ok=("label_ok", "mean"))
    print(d.T.to_string())
    print("\ntrips needing attention (label_ok < 95%, or coverage < 0.8, or >20% headings opposed):")
    bad = C[(C.frac_gt100 > .05) | (C.route_pos_coverage < .8) | (C.heading_frac_gt90 > .2)]
    print(bad[["partition", "trip_id", "n_obs", "frac_gt100", "route_pos_coverage", "heading_frac_gt90", "longest_bad_run", "longest_bad_run_start", "track_km", "route_km"]].round(3).to_string(index=False))

    print("\n=== CANDIDATE MISS RATE on the NATURAL candidate set (no label injection / no training-only correction) ===")
    rows = []
    for part in PARTITIONS:
        P = A[A.partition == part]
        for r in RADII:
            for K in KS:
                hit = (P.align_dist_m <= r) & (P.ref_rank < K)
                okm = P.label_ok == 1
                rows.append(dict(partition=part, radius_m=r, k_max=("inf" if K > 10**8 else K), miss_all=1 - hit.mean(),
                                 miss_label_ok=1 - hit[okm].mean(), mean_cands=np.minimum(P[f"n_within_{r}"], K).mean()))
    M = pd.DataFrame(rows); M.to_csv(out / "s3_miss_rate.csv", index=False)
    for part in PARTITIONS:
        print(f"\n[{part}] miss rate over ALL observations  (rows = radius m, cols = k_max)")
        print(M[M.partition == part].pivot(index="radius_m", columns="k_max", values="miss_all")[[10, 20, 50, "inf"]].round(4).to_string())
    print("\nDEFAULT (radius 50 m, k_max 50) - decomposition, all observations:")
    for part in PARTITIONS:
        P = A[A.partition == part]; far = (P.align_dist_m > DEFAULT.radius_m); cap = (~far) & (P.ref_rank >= DEFAULT.k_max)
        okm = P.label_ok == 1
        print(f"  {part:13s} obs={len(P):6d} miss={ (far|cap).mean():.4f}  [ref farther than radius={far.mean():.4f}  ref beyond k_max cap={cap.mean():.4f}]"
              f"  | label_ok only: miss={(far|cap)[okm].mean():.4f} (n={int(okm.sum())})")
        W = P.assign(m=(far | cap)).groupby("trip_id").m.agg(["mean", "sum", "size"]).sort_values("sum", ascending=False)
        print("   top-3 trips by #missed obs:", [(i, int(r["sum"]), round(r["mean"], 3)) for i, r in W.head(3).iterrows()],
              "| share of misses in top-3:", round(W["sum"].head(3).sum() / max(1, W["sum"].sum()), 3))
    # cross-check against the actual stage-2 candidate files at the default config
    print("\nCross-check against stage-2 candidate files (actual table, default config):")
    for part in PARTITIONS:
        f = out / f"candidates_{part}.csv.gz"
        if not f.exists(): print("  missing", f, "- run stage 2 first"); continue
        cand = pd.read_csv(f, usecols=["trip_id", "obs_idx", "arc_id"], dtype={"trip_id": str})
        lab = A[A.partition == part][["trip_id", "obs_idx", "ref_arc_id"]]
        key = set(zip(cand.trip_id, cand.obs_idx, cand.arc_id))
        miss = np.mean([(a_, b_, c_) not in key for a_, b_, c_ in zip(lab.trip_id, lab.obs_idx, lab.ref_arc_id)])
        sweep = M[(M.partition == part) & (M.radius_m == 50) & (M.k_max == 50)].miss_all.iloc[0]
        print(f"  {part:13s} miss from candidate file = {miss:.4f} | from sweep = {sweep:.4f} | equal: {abs(miss - sweep) < 1e-9}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
