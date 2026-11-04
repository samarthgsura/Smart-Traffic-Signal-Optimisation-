"""Download Bengaluru's lakes and large parks from OpenStreetMap for the 3D view.

    python scripts/fetch_osm_scenery.py      ->  data/cities/bengaluru_scenery.json

Only simple closed ways are used (most Bengaluru lakes are mapped that way); polygons
smaller than the thresholds below are skipped to keep the file small.
Data (c) OpenStreetMap contributors, ODbL.
"""
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAT0, LON0 = 12.9716, 77.5946
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110540
R = 50000
ENDPOINTS = ["https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
MIN_AREA = {"water": 40_000, "park": 80_000}


def proj(lat, lon):
    return ((lon - LON0) * KX, -(lat - LAT0) * KY)


def area(pts):
    return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2


def simplify(pts, eps):
    if len(pts) < 4:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(p, out[-1]) >= eps:
            out.append(p)
    return out


def stitch(parts):
    """Join way pieces end-to-end into closed rings."""
    parts = [p for p in parts if len(p) > 1]
    rings = []
    while parts:
        ring = parts.pop(0)
        changed = True
        while changed and math.dist(ring[0], ring[-1]) > 1:
            changed = False
            for i, p in enumerate(parts):
                if math.dist(ring[-1], p[0]) < 1:
                    ring += p[1:]
                elif math.dist(ring[-1], p[-1]) < 1:
                    ring += p[::-1][1:]
                elif math.dist(ring[0], p[-1]) < 1:
                    ring = p[:-1] + ring
                elif math.dist(ring[0], p[0]) < 1:
                    ring = p[::-1][:-1] + ring
                else:
                    continue
                parts.pop(i)
                changed = True
                break
        rings.append(ring[:-1] if math.dist(ring[0], ring[-1]) < 1 else ring)
    return rings


def main():
    q = f"""[out:json][timeout:300];
(
  way["natural"="water"](around:{R},{LAT0},{LON0});
  way["landuse"="reservoir"](around:{R},{LAT0},{LON0});
  way["leisure"="park"](around:{R},{LAT0},{LON0});
  way["landuse"="forest"](around:{R},{LAT0},{LON0});
);
(._;>;);
out body;"""
    data = urllib.parse.urlencode({"data": q}).encode()
    raw = None
    for url in ENDPOINTS:
        try:
            t0 = time.time()
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "neural-storm-ft01/1.0"})
            raw = json.loads(urllib.request.urlopen(req, timeout=600).read())
            print(f"downloaded from {url} in {time.time() - t0:.0f}s")
            break
        except Exception as e:
            print("failed", url, e)
    if raw is None:
        raise SystemExit("Overpass unavailable")
    nodes = {e["id"]: proj(e["lat"], e["lon"]) for e in raw["elements"] if e["type"] == "node"}
    out = {"water": [], "park": []}
    for e in raw["elements"]:
        if e["type"] != "way" or e["nodes"][0] != e["nodes"][-1]:
            continue
        t = e.get("tags", {})
        kind = "water" if t.get("natural") == "water" or t.get("landuse") == "reservoir" else "park"
        pts = [nodes[n] for n in e["nodes"][:-1] if n in nodes]
        if len(pts) < 3 or math.hypot(*pts[0]) > R:
            continue
        a = area(pts)
        if a < MIN_AREA[kind]:
            continue
        pts = simplify(pts, 12 if kind == "water" else 20)
        out[kind].append({"name": t.get("name", ""), "area": round(a), "pts": [[round(x), round(y)] for x, y in pts]})
    # multipolygon lakes (Bellandur, Ulsoor, Varthur... are mapped as relations)
    rq = f"""[out:json][timeout:300];
relation["natural"="water"](around:{R},{LAT0},{LON0});
out geom;"""
    rel = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": rq}).encode(),
                                         headers={"User-Agent": "neural-storm-ft01/1.0"})
            rel = json.loads(urllib.request.urlopen(req, timeout=600).read())
            break
        except Exception as e:
            print("relation query failed", url, e)
    for e in (rel or {}).get("elements", []):
        parts = [[proj(p["lat"], p["lon"]) for p in m.get("geometry", [])]
                 for m in e.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
        for ring in stitch(parts):
            if len(ring) < 3 or math.hypot(*ring[0]) > R:
                continue
            a = area(ring)
            if a < MIN_AREA["water"]:
                continue
            out["water"].append({"name": e.get("tags", {}).get("name", ""), "area": round(a),
                                 "pts": [[round(x), round(y)] for x, y in simplify(ring, 12)]})
    for k in out:
        out[k].sort(key=lambda p: -p["area"])
    out["water"] = out["water"][:400]
    out["park"] = out["park"][:300]
    dest = ROOT / "data" / "cities" / "bengaluru_scenery.json"
    dest.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    named = [w["name"] for w in out["water"] if w["name"]][:12]
    print(f"saved {dest}: {len(out['water'])} lakes, {len(out['park'])} parks/forests, "
          f"{dest.stat().st_size / 1e6:.1f} MB; biggest lakes: {', '.join(named)}")


if __name__ == "__main__":
    main()
