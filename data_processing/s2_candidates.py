"""Stage 2 - candidate generation + features for every permitted trip (NO .route used).

python s2_candidates.py --data-root "Student folder" --out out            # build tables + run checks
python s2_candidates.py --data-root "Student folder" --out out --self-test # checks only, 3 trips, writes nothing
python s2_candidates.py ... --radius 50 --k-max 50
"""
import argparse, ast, hashlib, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from common import PARTITIONS, list_trips, load_trip, seg_point_dist
from features import CandConfig, generate_candidates, FEATURE_COLUMNS, KEY_COLUMNS, DTYPES


def table_checks(c: pd.DataFrame, cfg: CandConfig):
    errs = []
    if list(c.columns) != KEY_COLUMNS + FEATURE_COLUMNS: errs.append("column order differs from contract")
    for k, v in DTYPES.items():
        if k != "trip_id" and str(c[k].dtype) != v: errs.append(f"dtype {k}: {c[k].dtype} != {v}")
    f = c[FEATURE_COLUMNS].to_numpy(dtype="float64")
    if not np.isfinite(f).all(): errs.append("NaN/inf in features")
    if c.duplicated(KEY_COLUMNS).any(): errs.append("duplicate (trip_id,obs_idx,arc_id)")
    if (c.dist_m > cfg.radius_m + 1e-3).any(): errs.append("dist_m > radius")
    if not c.heading_diff_deg.between(0, 180).all(): errs.append("heading_diff out of [0,180]")
    if not c.along_frac.between(0, 1).all(): errs.append("along_frac out of [0,1]")
    if (c.groupby(["trip_id", "obs_idx"]).dist_rank.min() != 0).any(): errs.append("rank does not start at 0")
    return errs


def route_independence_check(root, part, tid, cfg):
    """(1) static: features.py never touches 'route'. (2) dynamic: identical output with .route deleted."""
    tree = ast.parse(Path(__file__).with_name("features.py").read_text())
    hits = [n for n in ast.walk(tree) if (isinstance(n, ast.Attribute) and n.attr == "route")
            or (isinstance(n, ast.Name) and n.id == "route") or (isinstance(n, ast.arg) and n.arg == "route")]
    h = lambda df: hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    a, _, _ = generate_candidates(load_trip(root, part, tid), cfg)
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / part / tid; dst.mkdir(parents=True)
        for p in (Path(root) / part / tid).iterdir():
            if p.suffix != ".route": shutil.copy(p, dst / p.name)
        b, _, _ = generate_candidates(load_trip(td, part, tid), cfg)
    return len(hits) == 0, h(a) == h(b)


def brute_force_check(root, part, tid, cfg, step=11):
    c, g, o = generate_candidates(load_trip(root, part, tid), CandConfig(**{**cfg.__dict__, "k_max": 10**9}))
    bad = tot = 0
    for i in range(0, len(o), step):
        d, *_ = seg_point_dist(o.x_m[i], o.y_m[i], g.ax, g.ay, g.bx, g.by)
        tot += 1; bad += set(np.where(d <= cfg.radius_m)[0]) != set(c.arc_id[c.obs_idx == i])
    return bad, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="Student folder"); ap.add_argument("--out", default="out")
    ap.add_argument("--radius", type=float, default=CandConfig.radius_m); ap.add_argument("--k-max", type=int, default=CandConfig.k_max)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(); cfg = CandConfig(radius_m=a.radius, k_max=a.k_max)
    out = Path(a.out); out.mkdir(exist_ok=True)
    print("config:", cfg)
    first = list_trips(a.data_root, "train")[0]
    static_ok, same = route_independence_check(a.data_root, "train", first, cfg)
    print(f"[check] features.py never references 'route': {static_ok} | identical features with .route deleted: {same}")
    bad, tot = brute_force_check(a.data_root, "train", first, cfg)
    print(f"[check] KD-tree candidates vs brute force: {bad} mismatching of {tot} sampled observations")
    if a.self_test: return 0 if (static_ok and same and bad == 0) else 1
    summ, allerr = [], []
    for part in PARTITIONS:
        f = out / f"candidates_{part}.csv.gz"
        if f.exists(): f.unlink()
        first_chunk = True
        for tid in list_trips(a.data_root, part):
            t0 = time.time(); c, g, o = generate_candidates(load_trip(a.data_root, part, tid), cfg)
            allerr += [f"{part}/{tid}: {e}" for e in table_checks(c, cfg)]
            c.to_csv(f, mode="a", header=first_chunk, index=False, compression="gzip", float_format="%.4f"); first_chunk = False
            per = c.groupby("obs_idx").size()
            summ.append(dict(partition=part, trip_id=tid, n_obs=len(o), n_rows=len(c), n_arcs=g.n_arcs,
                             cand_per_obs_mean=len(c) / len(o), cand_per_obs_max=int(per.max()) if len(per) else 0,
                             obs_zero_cand=int(len(o) - per.size), heading_valid_frac_obs=float(o.heading_deg.notna().mean()),
                             seconds=time.time() - t0))
    s = pd.DataFrame(summ); s.to_csv(out / "s2_summary.csv", index=False)
    g_ = s.groupby("partition").agg(trips=("trip_id", "count"), obs=("n_obs", "sum"), rows=("n_rows", "sum"),
                                    obs_zero_cand=("obs_zero_cand", "sum"), max_cand_per_obs=("cand_per_obs_max", "max"),
                                    seconds=("seconds", "sum"))
    g_["cand_per_obs"] = g_.rows / g_.obs
    print(g_.T.to_string()); print("table-contract errors:", allerr or "none")
    return 0 if not allerr else 1

if __name__ == "__main__":
    sys.exit(main())
