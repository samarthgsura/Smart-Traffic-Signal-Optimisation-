"""Turn the raw OpenStreetMap download into a simulation-ready city network.

    python scripts/build_osm_city.py [--src data/osm/bengaluru_50km.json] [--total-vph 26000]

Steps
  1. Project lat/lon to metres around the city centre; drop points beyond the radius.
  2. Split OSM ways at shared nodes; respect one-way tags (motorways, ramps, roundabouts).
  3. Merge the several OSM nodes of one physical intersection into one junction (same
     level only, so a flyover never merges with the road it crosses).
  4. Remove pass-through points (one way in, one way out) to keep the graph small.
  5. Keep the largest strongly connected part (every junction can reach every other).
  6. Mark junctions with a real OSM traffic signal; the rest are unsignalised.
  7. Add trip zones (real hubs + highway entry points at the edge) and a gravity-model
     demand table between them. Demand figures are illustrative, not measured counts.

Writes data/cities/bengaluru_osm.json (loaded by traffic/city_graph.py).
Road data (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAT0, LON0 = 12.9716, 77.5946
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110540

CLASS_RANK = {"motorway": 5, "trunk": 4, "primary": 3, "secondary": 2, "tertiary": 1}
DEFAULT_LANES = {"motorway": 3, "trunk": 2, "primary": 2, "secondary": 1, "tertiary": 1}
DEFAULT_KMH = {"motorway": 70, "trunk": 50, "primary": 40, "secondary": 30, "tertiary": 25}

# Real trip generators: (name, lat, lon, weight, is_it_hub)
HUBS = [
    ("Majestic / CBD", 12.9767, 77.5713, 10, False), ("MG Road", 12.9756, 77.6066, 8, False),
    ("Whitefield ITPL", 12.9866, 77.7366, 9, True), ("Electronic City", 12.8452, 77.6602, 9, True),
    ("Manyata Tech Park", 13.0456, 77.6206, 8, True), ("ORR Marathahalli", 12.9404, 77.6990, 9, True),
    ("Bellandur / Ecospace", 12.9259, 77.6820, 8, True), ("Sarjapur Road", 12.9070, 77.7050, 5, True),
    ("Koramangala", 12.9352, 77.6245, 6, False), ("HSR Layout", 12.9116, 77.6474, 5, False),
    ("Jayanagar", 12.9250, 77.5938, 5, False), ("Banashankari", 12.9255, 77.5468, 5, False),
    ("Yeshwanthpur", 13.0285, 77.5401, 6, False), ("Peenya Industrial Area", 13.0329, 77.5250, 6, False),
    ("Hebbal", 13.0358, 77.5970, 5, False), ("Yelahanka", 13.1007, 77.5963, 4, False),
    ("Kempegowda Airport", 13.1986, 77.7066, 5, False), ("KR Puram", 13.0050, 77.6950, 5, False),
    ("Indiranagar", 12.9719, 77.6412, 6, False), ("Kengeri", 12.9077, 77.4826, 4, False),
    ("Hosur", 12.7409, 77.8253, 5, False), ("Bommasandra", 12.8170, 77.6960, 4, True),
    ("Nelamangala", 13.0975, 77.3910, 3, False), ("Devanahalli", 13.2437, 77.7172, 2, False),
    ("Rajajinagar", 12.9915, 77.5525, 5, False), ("Malleshwaram", 13.0035, 77.5709, 4, False),
    ("Hoskote", 13.0707, 77.7982, 3, False), ("Bidadi", 12.7970, 77.3880, 2, False),
    ("Kanakapura Road", 12.8890, 77.5560, 4, False), ("Attibele", 12.7790, 77.7700, 3, False),
    ("Doddaballapur", 13.2920, 77.5380, 2, False), ("Ramanagara", 12.7220, 77.2810, 2, False),
]

# Hospitals: ambulance destinations (no regular demand).
HOSPITALS = [
    ("Victoria Hospital", 12.9634, 77.5735), ("St. John's Hospital", 12.9293, 77.6187),
    ("Manipal Hospital, Old Airport Rd", 12.9589, 77.6490), ("Narayana Health City", 12.8113, 77.6950),
    ("Aster CMI Hospital, Hebbal", 13.0441, 77.5926), ("Sakra World Hospital", 12.9330, 77.6880),
]
AMBULANCE_TRIPS = [
    ("Electronic City", "St. John's Hospital"), ("Whitefield ITPL", "Manipal Hospital, Old Airport Rd"),
    ("Yeshwanthpur", "Victoria Hospital"), ("Kengeri", "Victoria Hospital"),
    ("Sarjapur Road", "Sakra World Hospital"), ("Manyata Tech Park", "Aster CMI Hospital, Hebbal"),
    ("Hosur", "Narayana Health City"), ("KR Puram", "Manipal Hospital, Old Airport Rd"),
]

# Well-known junctions, used to name the nearest signalised junction.
LANDMARKS = [
    ("Silk Board", 12.9177, 77.6233), ("Hebbal", 13.0358, 77.5970), ("Tin Factory", 12.9966, 77.6683),
    ("KR Puram", 13.0020, 77.6790), ("Marathahalli Bridge", 12.9569, 77.7011), ("Iblur", 12.9213, 77.6656),
    ("Agara", 12.9235, 77.6455), ("Sony World", 12.9370, 77.6270), ("Mekhri Circle", 13.0145, 77.5838),
    ("Goraguntepalya", 13.0280, 77.5300), ("Anand Rao Circle", 12.9785, 77.5760), ("Town Hall", 12.9634, 77.5855),
    ("Richmond Circle", 12.9640, 77.5970), ("Domlur", 12.9610, 77.6380), ("Trinity Circle", 12.9729, 77.6170),
    ("KR Circle", 12.9750, 77.5900), ("Madiwala", 12.9230, 77.6180), ("Bommanahalli", 12.9010, 77.6240),
    ("Jayadeva", 12.9180, 77.5990), ("Banashankari", 12.9180, 77.5730), ("Nayandahalli", 12.9420, 77.5230),
    ("Yeshwanthpur Circle", 13.0230, 77.5500), ("Nagawara", 13.0420, 77.6230), ("Hennur", 13.0270, 77.6400),
    ("Bellandur", 12.9260, 77.6760), ("Kadubeesanahalli", 12.9380, 77.6980), ("Hope Farm", 12.9840, 77.7520),
    ("HAL / Old Airport Rd", 12.9590, 77.6640), ("Electronic City Toll", 12.8500, 77.6630),
    ("Kengeri", 12.9080, 77.4850), ("Yelahanka", 13.1000, 77.5960), ("BEL Circle", 13.0440, 77.5550),
    ("Sumanahalli", 12.9850, 77.5080), ("Hosur Road / Bommasandra", 12.8180, 77.6940),
    ("Ejipura", 12.9440, 77.6260), ("Shivananda Circle", 12.9880, 77.5850), ("Cauvery Junction", 13.0010, 77.5870),
    ("Lalbagh West Gate", 12.9500, 77.5800), ("Kanakapura Road / Konanakunte", 12.8890, 77.5560),
]


def proj(lat, lon):
    return ((lon - LON0) * KX, -(lat - LAT0) * KY)


def parse_lanes(tags, cls, oneway):
    try:
        total = int(str(tags.get("lanes", "")).split(";")[0])
    except ValueError:
        total = 0
    if total <= 0:
        return DEFAULT_LANES[cls]
    return max(1, total if oneway else total // 2)


def parse_speed(tags, cls):
    try:
        v = float(str(tags.get("maxspeed", "")).split()[0])
        if 10 <= v <= 120:
            return min(v, DEFAULT_KMH[cls] + 20)  # posted limits are optimistic in city traffic
    except (ValueError, IndexError):
        pass
    return DEFAULT_KMH[cls]


def level_of(tags):
    if tags.get("bridge") in ("yes", "viaduct"):
        return 1
    if tags.get("tunnel") in ("yes", "building_passage"):
        return -1
    try:
        lay = int(str(tags.get("layer", "0")).split(";")[0])
    except ValueError:
        lay = 0
    return 1 if lay > 0 else -1 if lay < 0 else 0


def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1e-9
    best, bi = -1.0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / L
        if d > best:
            best, bi = d, i
    if best <= eps:
        return [pts[0], pts[-1]]
    return douglas_peucker(pts[:bi + 1], eps)[:-1] + douglas_peucker(pts[bi:], eps)


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "osm" / "bengaluru_50km.json"))
    ap.add_argument("--radius", type=float, default=50)
    ap.add_argument("--total-vph", type=float, default=45000)
    ap.add_argument("--merge-m", type=float, default=35)
    a = ap.parse_args()
    R = a.radius * 1000

    raw = json.loads(Path(a.src).read_text(encoding="utf-8"))
    coords, signals = {}, []
    ways = []
    for e in raw["elements"]:
        if e["type"] == "node":
            coords[e["id"]] = proj(e["lat"], e["lon"])
            if e.get("tags", {}).get("highway") == "traffic_signals":
                signals.append(coords[e["id"]])
        elif e["type"] == "way":
            ways.append(e)
    print(f"raw: {len(ways)} ways, {len(coords)} nodes, {len(signals)} signal nodes")

    # ---- 1-2. split ways into directed segments --------------------------------
    use = defaultdict(int)
    for w in ways:
        for i, n in enumerate(w["nodes"]):
            use[n] += 2 if i in (0, len(w["nodes"]) - 1) else 1
    node_level = defaultdict(set)
    segs = []  # (u, v, pts, attrs)
    for w in ways:
        t = w.get("tags", {})
        hw = t["highway"]
        cls = hw.replace("_link", "")
        if cls not in CLASS_RANK:
            continue
        ow = t.get("oneway", "")
        oneway = ow in ("yes", "true", "1") or t.get("junction") in ("roundabout", "circular") \
            or (hw in ("motorway", "motorway_link") and ow != "no")
        reverse = ow == "-1"
        lvl = level_of(t)
        attrs = {"cls": cls, "link": hw.endswith("_link"), "lanes": parse_lanes(t, cls, oneway or reverse),
                 "kmh": parse_speed(t, cls), "level": lvl,
                 "name": t.get("name") or t.get("ref") or hw.replace("_", " ").title(),
                 "ref": t.get("ref", "")}
        nodes = [n for n in w["nodes"] if n in coords]
        # clip at the radius: keep runs of inside points
        runs, cur = [], []
        for n in nodes:
            x, y = coords[n]
            if x * x + y * y <= R * R:
                cur.append(n)
            else:
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
        if len(cur) > 1:
            runs.append(cur)
        for run in runs:
            start = 0
            for i in range(1, len(run)):
                if i == len(run) - 1 or use[run[i]] > 1:
                    part = run[start:i + 1]
                    pts = [coords[n] for n in part]
                    u, v = part[0], part[-1]
                    node_level[u].add(lvl)
                    node_level[v].add(lvl)
                    if reverse:
                        segs.append((v, u, pts[::-1], attrs))
                    else:
                        segs.append((u, v, pts, attrs))
                        if not oneway:
                            segs.append((v, u, pts[::-1], attrs))
                    start = i
    print(f"segments: {len(segs)}")

    # ---- 3. merge intersection nodes (same level only) --------------------------
    def lvl_key(n):
        s = node_level[n]
        return 0 if 0 in s or len(s) != 1 else next(iter(s))
    endpoints = {s[0] for s in segs} | {s[1] for s in segs}
    grid = defaultdict(list)
    cell = a.merge_m
    for n in endpoints:
        x, y = coords[n]
        grid[(int(x // cell), int(y // cell), lvl_key(n))].append(n)
    uf = UF()
    for (cx, cy, lv), items in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in grid.get((cx + dx, cy + dy, lv), ()):
                    for n in items:
                        if n < m and math.dist(coords[n], coords[m]) <= a.merge_m:
                            uf.union(n, m)
    members = defaultdict(list)
    for n in endpoints:
        members[uf.find(n)].append(n)
    cpos = {c: (sum(coords[m][0] for m in ms) / len(ms), sum(coords[m][1] for m in ms) / len(ms))
            for c, ms in members.items()}

    edges = {}  # (cu, cv) -> edge dict (keep the bigger road if duplicated)
    for u, v, pts, at in segs:
        cu, cv = uf.find(u), uf.find(v)
        if cu == cv:
            continue
        length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        e = {"u": cu, "v": cv, "pts": [cpos[cu]] + pts[1:-1] + [cpos[cv]], "len": max(length, 5.0), **at}
        k = (cu, cv)
        if k not in edges or (CLASS_RANK[at["cls"]], at["lanes"]) > (CLASS_RANK[edges[k]["cls"]], edges[k]["lanes"]):
            edges[k] = e
    print(f"after merging intersections: {len(cpos)} nodes, {len(edges)} edges")

    # ---- mark signals --------------------------------------------------------------
    sgrid = defaultdict(list)
    for x, y in signals:
        sgrid[(int(x // 60), int(y // 60))].append((x, y))

    def has_signal(c):
        x, y = cpos[c]
        gx, gy = int(x // 60), int(y // 60)
        return any(math.hypot(sx - x, sy - y) <= 55 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                   for sx, sy in sgrid.get((gx + dx, gy + dy), ()))

    # ---- 4. contract pass-through nodes ------------------------------------------
    def rebuild_adj():
        outs, ins = defaultdict(set), defaultdict(set)
        for (u, v) in edges:
            outs[u].add(v)
            ins[v].add(u)
        return outs, ins
    sig = {c for c in cpos if has_signal(c)}
    changed = True
    while changed:
        changed = False
        outs, ins = rebuild_adj()
        for n in list(cpos):
            if n in sig or n not in outs:
                continue
            o, i = outs[n], ins[n]
            pairs = []
            if len(o) == 1 and len(i) == 1 and o != i:
                pairs = [(next(iter(i)), next(iter(o)))]
            elif len(o) == 2 and o == i:
                p, q = sorted(o)
                pairs = [(p, q), (q, p)]
            else:
                continue
            ok = True
            for p, q in pairs:
                e1, e2 = edges.get((p, n)), edges.get((n, q))
                if not e1 or not e2 or e1["level"] != e2["level"] or (p, q) in edges or p == q:
                    ok = False
            if not ok:
                continue
            for p, q in pairs:
                e1, e2 = edges.pop((p, n)), edges.pop((n, q))
                L = e1["len"] + e2["len"]
                edges[(p, q)] = {**e1, "v": q, "pts": e1["pts"] + e2["pts"][1:], "len": L,
                                 "lanes": min(e1["lanes"], e2["lanes"]),
                                 "kmh": L / (e1["len"] / e1["kmh"] + e2["len"] / e2["kmh"]),
                                 "cls": e1["cls"] if CLASS_RANK[e1["cls"]] >= CLASS_RANK[e2["cls"]] else e2["cls"],
                                 "name": e1["name"] if e1["len"] >= e2["len"] else e2["name"],
                                 "link": e1["link"] and e2["link"]}
            del cpos[n]
            changed = True
    print(f"after removing pass-through points: {len(cpos)} nodes, {len(edges)} edges")

    # ---- 5. largest strongly connected component (iterative Tarjan) -------------
    outs, _ = rebuild_adj()
    index, low, onstack, stack, comps = {}, {}, set(), [], []
    counter = [0]
    for root in cpos:
        if root in index:
            continue
        work = [(root, iter(outs[root]))]
        index[root] = low[root] = counter[0]; counter[0] += 1
        stack.append(root); onstack.add(root)
        while work:
            n, it = work[-1]
            advanced = False
            for m in it:
                if m not in index:
                    index[m] = low[m] = counter[0]; counter[0] += 1
                    stack.append(m); onstack.add(m)
                    work.append((m, iter(outs[m])))
                    advanced = True
                    break
                elif m in onstack:
                    low[n] = min(low[n], index[m])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[n])
            if low[n] == index[n]:
                comp = []
                while True:
                    m = stack.pop(); onstack.discard(m); comp.append(m)
                    if m == n:
                        break
                comps.append(comp)
    keep = set(max(comps, key=len))
    edges = {k: e for k, e in edges.items() if k[0] in keep and k[1] in keep}
    cpos = {c: p for c, p in cpos.items() if c in keep}
    print(f"largest connected network: {len(cpos)} nodes, {len(edges)} edges")

    # ---- 6. node ids, kinds, names -----------------------------------------------
    order = sorted(cpos, key=lambda c: (cpos[c][1], cpos[c][0]))
    nid = {c: f"J{i}" for i, c in enumerate(order)}
    names_at = defaultdict(lambda: defaultdict(float))
    for e in edges.values():
        for c in (e["u"], e["v"]):
            if not e["name"].lower().endswith(" link"):
                names_at[c][e["name"]] += CLASS_RANK[e["cls"]] * 10 + e["len"] / 1000
    nodes = {}
    for c in order:
        nm = sorted(names_at[c].items(), key=lambda kv: -kv[1])
        label = " × ".join(n for n, _ in nm[:2]) if nm else nid[c]
        nodes[nid[c]] = {"id": nid[c], "name": label, "x": round(cpos[c][0], 1), "y": round(cpos[c][1], 1),
                         "kind": "signal" if c in sig else "junction"}
    sig_nodes = [n for n in nodes.values() if n["kind"] == "signal"]
    used = set()
    for name, lat, lon in LANDMARKS:
        x, y = proj(lat, lon)
        best = min((n for n in sig_nodes if n["id"] not in used),
                   key=lambda n: math.hypot(n["x"] - x, n["y"] - y))
        if math.hypot(best["x"] - x, best["y"] - y) <= 600:
            best["name"] = name
            best["landmark"] = True
            used.add(best["id"])

    links = []
    for (u, v), e in edges.items():
        pts = douglas_peucker(e["pts"], 4.0)
        links.append({"id": f"{nid[u]}>{nid[v]}", "src": nid[u], "dst": nid[v], "lanes": e["lanes"],
                      "speed": round(e["kmh"] / 3.6, 2), "length": round(e["len"], 1),
                      "road": "-".join(sorted((nid[u], nid[v]))), "name": e["name"], "cls": e["cls"],
                      "level": e["level"], "ramp": e["link"],
                      "geom": [[round(px, 1), round(py, 1)] for px, py in pts]})

    # ---- 7. zones: hubs + edge entry points ---------------------------------------
    by_class = defaultdict(list)
    for l in links:
        by_class[l["src"]].append(l)
    def best_attach(x, y, max_d, allow_motorway=False):
        cands = []
        for n in nodes.values():
            d = math.hypot(n["x"] - x, n["y"] - y)
            if d <= max_d:
                cls = max((CLASS_RANK[l["cls"]] for l in by_class[n["id"]]), default=0)
                if cls == 5 and not allow_motorway:
                    continue
                cands.append((d, n["id"]))
        return [nid_ for _, nid_ in sorted(cands)[:2]]

    zones = []
    for name, lat, lon, w, it in HUBS:
        x, y = proj(lat, lon)
        if math.hypot(x, y) > R:
            continue
        att = best_attach(x, y, 2500)
        if att:
            zones.append({"name": name, "x": x, "y": y, "w": w, "it": it, "attach": att})
    for name, lat, lon in HOSPITALS:
        x, y = proj(lat, lon)
        att = best_attach(x, y, 2000)
        if att:
            zones.append({"name": name, "x": x, "y": y, "w": 0, "it": False, "attach": att[:1],
                          "hospital": True})
    # entry points: highways crossing the edge, one per 20-degree sector
    edge_cands = {}
    for l in links:
        if l["cls"] not in ("motorway", "trunk", "primary"):
            continue
        for nid_ in (l["src"], l["dst"]):
            n = nodes[nid_]
            r = math.hypot(n["x"], n["y"])
            if r > R - 4000:
                sector = int((math.degrees(math.atan2(n["y"], n["x"])) + 360) % 360 // 20)
                score = r + 3000 * CLASS_RANK[l["cls"]]
                if sector not in edge_cands or score > edge_cands[sector][0]:
                    edge_cands[sector] = (score, nid_, l["name"])
    for sector, (_, nid_, road) in sorted(edge_cands.items()):
        n = nodes[nid_]
        if any(math.hypot(z["x"] - n["x"], z["y"] - n["y"]) < 6000 for z in zones if not z.get("hospital")):
            continue
        zones.append({"name": f"{road} (city edge)", "x": n["x"], "y": n["y"], "w": 3, "it": False,
                      "attach": [nid_], "edge": True})
    print(f"zones: {sum(1 for z in zones if not z.get('edge'))} hubs + "
          f"{sum(1 for z in zones if z.get('edge'))} edge entry points")

    for i, z in enumerate(zones):
        gid = f"Z{i}"
        nodes[gid] = {"id": gid, "name": z["name"], "x": round(z["x"], 1), "y": round(z["y"], 1),
                      "kind": "gateway", "it": z["it"], "hospital": bool(z.get("hospital"))}
        z["id"] = gid
        for att in z["attach"]:
            n = nodes[att]
            d = max(150.0, math.hypot(n["x"] - z["x"], n["y"] - z["y"]))
            for s, t in ((gid, att), (att, gid)):
                links.append({"id": f"{s}>{t}", "src": s, "dst": t, "lanes": 2, "speed": 8.0, "length": round(d, 1),
                              "road": f"{gid}-{att}", "name": f"{z['name']} local roads", "cls": "connector",
                              "level": 0, "ramp": False,
                              "geom": [[nodes[s]["x"], nodes[s]["y"]], [nodes[t]["x"], nodes[t]["y"]]]})

    # ---- demand (gravity model) + routes -------------------------------------------
    adj = defaultdict(list)
    for i, l in enumerate(links):
        adj[l["src"]].append(i)

    def route(o, d):
        dist, prev, pq = {o: 0.0}, {}, [(0.0, o)]
        while pq:
            c, n = heapq.heappop(pq)
            if n == d:
                break
            if c > dist.get(n, 1e18) or (n != o and nodes[n]["kind"] == "gateway"):
                continue
            for li in adj[n]:
                l = links[li]
                pen = 20 if nodes[l["dst"]]["kind"] == "signal" else 4
                nc = c + l["length"] / l["speed"] + pen
                if nc < dist.get(l["dst"], 1e18):
                    dist[l["dst"]] = nc
                    prev[l["dst"]] = li
                    heapq.heappush(pq, (nc, l["dst"]))
        if d not in prev:
            return None
        path, n = [], d
        while n != o:
            li = prev[n]
            path.append(links[li]["id"])
            n = links[li]["src"]
        return path[::-1]

    pairs = []
    for zi in zones:
        for zj in zones:
            if zi is zj or not zi["w"] or not zj["w"]:
                continue
            dkm = math.hypot(zi["x"] - zj["x"], zi["y"] - zj["y"]) / 1000
            if dkm < 3:
                continue
            pairs.append((zi["w"] * zj["w"] / (dkm + 5) ** 1.3, zi, zj))
    pairs.sort(key=lambda p: -p[0])
    pairs = pairs[:600]
    k = a.total_vph / sum(p[0] for p in pairs)
    ods = []
    for g, zi, zj in pairs:
        r = route(zi["id"], zj["id"])
        if not r:
            continue
        tags = ["it"] if zi["it"] else []
        ods.append({"origin": zi["id"], "dest": zj["id"], "vph": round(g * k, 1), "tags": tags, "route": r})
    print(f"demand: {len(ods)} origin-destination pairs, {sum(o['vph'] for o in ods):.0f} vehicles/hour")

    amb_pairs = AMBULANCE_TRIPS
    zid = {z["name"]: z["id"] for z in zones}
    ambulance = []
    for o, d in amb_pairs:
        if o in zid and d in zid:
            r = route(zid[o], zid[d])
            if r:
                ambulance.append({"origin": zid[o], "dest": zid[d], "route": r, "label": f"{o} → {d}"})

    features = []
    seen = set()
    for l in links:
        if l["level"] != 0 and l["length"] > 120 and l["cls"] != "connector":
            nm = l["name"]
            if nm in seen:
                continue
            if l["level"] > 0 and not any(s in nm.lower() for s in ("flyover", "elevated", "bridge", "expressway")):
                continue
            seen.add(nm)
            g = l["geom"][len(l["geom"]) // 2]
            features.append({"name": nm, "x": g[0], "y": g[1], "kind": "flyover" if l["level"] > 0 else "underpass"})

    out = {"key": "bengaluru_osm", "name": "Bengaluru Metro (real map)",
           "description": f"Real road network from OpenStreetMap within {a.radius:.0f} km of the city centre: "
                          f"expressways, highways, arterial and sub-arterial roads, ramps, flyovers and underpasses.",
           "rushLabel": "IT Park 6 PM Rush", "rushTag": "it", "mPerUnit": 1.0,
           "nodes": list(nodes.values()), "links": links, "ods": ods, "ambulanceRoutes": ambulance,
           "features": features, "radius": R,
           "attribution": "Road data © OpenStreetMap contributors (ODbL)"}
    dest = ROOT / "data" / "cities"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "bengaluru_osm.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    n_sig = sum(1 for n in nodes.values() if n["kind"] == "signal")
    n_j = sum(1 for n in nodes.values() if n["kind"] == "junction")
    n_fly = sum(1 for l in links if l["level"] > 0)
    n_tun = sum(1 for l in links if l["level"] < 0)
    km = sum(l["length"] for l in links if l["cls"] != "connector") / 1000
    print(f"saved data/cities/bengaluru_osm.json: {n_sig} signalised junctions, {n_j} other junctions, "
          f"{len(links)} links ({km:.0f} km of road, {n_fly} flyover links, {n_tun} underpass links), "
          f"{len(features)} named flyovers/underpasses, "
          f"{(dest / 'bengaluru_osm.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
