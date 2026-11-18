"""viz.py - Folium map for one trip: GPS, road geometry, development reference route, alignment links.
Data preparation (map_layers) is plain numpy/pandas; folium is only imported in write_map().
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import seg_point_dist

GOOD_M, BAD_M = 15.0, 50.0          # marker colour breaks for aligned distance


def map_layers(trip, g, obs, labels=None, obs_range=None, road_radius_m=150.0, max_road_arcs=15000):
    """labels: DataFrame from align_to_route (dev only) or None for a GPS+roads-only map."""
    n = len(obs); lo, hi = obs_range or (0, n)
    lo, hi = max(0, lo), min(n, hi); sel = np.arange(lo, hi)
    x, y = obs.x_m.to_numpy()[sel], obs.y_m.to_numpy()[sel]
    L = {"window": (lo, hi)}
    # road geometry near the GPS window (undirected unique pairs; direction kept in tooltip as u->v of lowest arc_id)
    mid = np.column_stack([(g.ax + g.bx) / 2, (g.ay + g.by) / 2])
    r = road_radius_m + g.length_m.max() / 2 if g.n_arcs else 0
    near = np.zeros(g.n_arcs, bool)
    # exact per-arc test via bbox of arc vs GPS tree: use arc endpoints + midpoint distance (cheap, conservative)
    gt = cKDTree(np.column_stack([x, y]))
    for px, py in ((g.ax, g.ay), (g.bx, g.by), (mid[:, 0], mid[:, 1])):
        near |= gt.query(np.column_stack([px, py]))[0] <= road_radius_m
    ids = np.where(near)[0]
    lo_n, hi_n = np.minimum(g.u[ids], g.v[ids]), np.maximum(g.u[ids], g.v[ids])
    _, first = np.unique(lo_n * g.n_nodes + hi_n, return_index=True)
    ids = ids[np.sort(first)][:max_road_arcs]
    nl = trip.nodes
    L["roads"] = [([nl[g.u[a], 1], nl[g.u[a], 0]], [nl[g.v[a], 1], nl[g.v[a], 0]], int(a), bool(g.reverse_id[a] >= 0)) for a in ids]
    L["road_arcs_truncated"] = bool(len(ids) >= max_road_arcs)
    tr = trip.track
    L["gps"] = [(float(tr.lat.iat[i]), float(tr.lon.iat[i]), int(i), float(tr.t_s.iat[i])) for i in sel]
    # path broken at time gaps >= 5 s
    t = tr.t_s.to_numpy()[sel]; brk = np.where(np.diff(t) >= 5.0)[0] + 1
    L["gps_paths"] = [[(float(a), float(b)) for a, b in zip(tr.lat.to_numpy()[sel][s:e], tr.lon.to_numpy()[sel][s:e])]
                      for s, e in zip(np.r_[0, brk], np.r_[brk, len(sel)]) if e - s >= 2]
    if labels is not None and trip.route is not None:
        lab = labels.set_index("obs_idx").loc[sel]
        p0, p1 = int(lab.route_pos.min()), int(lab.route_pos.max())
        a_ids = trip.route[max(0, p0 - 3): p1 + 4]
        L["route"] = [([nl[g.u[a], 1], nl[g.u[a], 0]], [nl[g.v[a], 1], nl[g.v[a], 0]], int(a), int(p))
                      for p, a in zip(range(max(0, p0 - 3), p1 + 4), a_ids)]
        a = lab.ref_arc_id.to_numpy()
        d, _, fx, fy = seg_point_dist(x, y, g.ax[a], g.ay[a], g.bx[a], g.by[a])
        flon, flat = g.proj.inv(fx, fy)
        L["links"] = [((float(tr.lat.iat[i]), float(tr.lon.iat[i])), (float(la), float(lo_)), float(dd), int(i), int(aid), int(rp))
                      for i, la, lo_, dd, aid, rp in zip(sel, flat, flon, d, a, lab.route_pos.to_numpy())]
    return L


def _colour(d):
    return "#2ca02c" if d <= GOOD_M else ("#ff7f0e" if d <= BAD_M else "#d62728")


def write_map(L, path, title, show_alignment):
    import folium
    lat0 = np.mean([p[0] for p in L["gps"]]); lon0 = np.mean([p[1] for p in L["gps"]])
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    roads = folium.FeatureGroup(name="Road network (grey; arc_id in tooltip)", show=True)
    for a, b, aid, two in L["roads"]:
        folium.PolyLine([a, b], color="#888888", weight=2, opacity=0.7,
                        tooltip=f"arc {aid} ({'two-way' if two else 'one-way'})").add_to(roads)
    roads.add_to(m)
    if "route" in L:
        rt = folium.FeatureGroup(name="Reference route (green, DEV ONLY)", show=True)
        for a, b, aid, p in L["route"]:
            folium.PolyLine([a, b], color="#1a9850", weight=6, opacity=0.8, tooltip=f"route pos {p} | arc {aid}").add_to(rt)
        rt.add_to(m)
    paths = folium.FeatureGroup(name="GPS path (blue; broken at gaps >= 5 s)", show=True)
    for pth in L["gps_paths"]:
        folium.PolyLine(pth, color="#1f77b4", weight=1.5, opacity=0.6).add_to(paths)
    paths.add_to(m)
    link = {x[3]: x for x in L.get("links", [])}
    pts = folium.FeatureGroup(name="GPS fixes" + (" (colour = distance to aligned arc)" if show_alignment else ""), show=True)
    for lat, lon, i, t in L["gps"]:
        if show_alignment and i in link:
            d = link[i][2]; col = _colour(d); tip = f"obs {i} | t={t:.0f}s | arc {link[i][4]} | route pos {link[i][5]} | {d:.1f} m"
        else:
            col, tip = "#1f77b4", f"obs {i} | t={t:.0f}s"
        folium.CircleMarker([lat, lon], radius=3, color=col, fill=True, fill_opacity=0.9, weight=1, tooltip=tip).add_to(pts)
    pts.add_to(m)
    if show_alignment and link:
        lk = folium.FeatureGroup(name="Alignment links (GPS -> foot point on aligned arc)", show=True)
        for (a, b, d, i, aid, rp) in link.values():
            folium.PolyLine([a, b], color=_colour(d), weight=1, opacity=0.8).add_to(lk)
        lk.add_to(m)
    first, last = L["gps"][0], L["gps"][-1]
    folium.Marker([first[0], first[1]], tooltip="first fix", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([last[0], last[1]], tooltip="last fix", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    lats, lons = [p[0] for p in L["gps"]], [p[1] for p in L["gps"]]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    legend = (f"<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;"
              f"border:1px solid #888;font-size:12px'><b>{title}</b><br>"
              f"<span style='color:#1f77b4'>&#9679;</span> GPS fix / path<br><span style='color:#888'>&#9473;</span> road arcs<br>"
              + ("<span style='color:#1a9850'>&#9473;</span> reference route (dev label)<br>" if "route" in L else "")
              + (f"<span style='color:#2ca02c'>&#9679;</span> aligned &le;{GOOD_M:.0f} m "
                 f"<span style='color:#ff7f0e'>&#9679;</span> &le;{BAD_M:.0f} m <span style='color:#d62728'>&#9679;</span> &gt;{BAD_M:.0f} m<br>"
                 if show_alignment else "")
              + f"obs {L['window'][0]}..{L['window'][1] - 1}</div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
