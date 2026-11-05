"""Download Namma Metro lines and stations and tall buildings from OpenStreetMap for the 3D view.

    python scripts/fetch_osm_urban.py   ->  data/cities/bengaluru_urban.json

Data (c) OpenStreetMap contributors, ODbL.
"""
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAT0, LON0 = 12.9716, 77.5946
KX = 111320 * math.cos(math.radians(LAT0))
KY = 110540
RADIUS = 50000
TOWER_RADIUS = 32000
MAX_TOWERS = 6000
METRES_PER_LEVEL = 3.3
ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def proj(lat, lon):
    return (lon - LON0) * KX, -(lat - LAT0) * KY


def overpass(query):
    data = urllib.parse.urlencode({"data": query}).encode()
    for url in ENDPOINTS:
        try:
            t0 = time.time()
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "neural-storm-ft01/1.0"})
            raw = json.loads(urllib.request.urlopen(req, timeout=170).read())
            print(f"  {url}: {len(raw['elements'])} elements in {time.time() - t0:.0f}s", flush=True)
            return raw["elements"]
        except Exception as e:
            print(f"  failed {url}: {e}", flush=True)
    return None


def number(text):
    match = re.match(r"\s*(\d+(?:\.\d+)?)", text or "")
    return float(match.group(1)) if match else None


def fetch_metro():
    elements = overpass(f"""[out:json][timeout:160];
relation["route"~"^(subway|light_rail)$"](around:{RADIUS},{LAT0},{LON0})->.lines;
.lines out body;
way(r.lines)->.tracks;
.tracks out tags geom;
node["railway"="station"]["station"~"^(subway|light_rail)$"](around:{RADIUS},{LAT0},{LON0});
out tags;""")
    if elements is None:
        return None
    colour, line_name = {}, {}
    for e in elements:
        if e["type"] == "relation":
            tags = e.get("tags", {})
            for m in e.get("members", []):
                if m["type"] == "way":
                    colour.setdefault(m["ref"], tags.get("colour", ""))
                    line_name.setdefault(m["ref"], tags.get("name", ""))
    lines = []
    for e in elements:
        if e["type"] != "way" or not e.get("geometry"):
            continue
        tags = e.get("tags", {})
        if tags.get("railway") not in ("subway", "light_rail", "rail"):
            continue
        pts = [[round(a) for a in proj(p["lat"], p["lon"])] for p in e["geometry"]]
        lines.append({"colour": colour.get(e["id"], tags.get("colour", "")), "name": line_name.get(e["id"], ""),
                      "tunnel": tags.get("tunnel", "no") != "no" or tags.get("layer", "0").startswith("-"),
                      "pts": pts})
    stations = []
    for e in elements:
        if e["type"] == "node":
            x, y = proj(e["lat"], e["lon"])
            tags = e.get("tags", {})
            stations.append({"name": tags.get("name:en") or tags.get("name", ""), "x": round(x), "y": round(y),
                             "under": tags.get("subway") == "yes" or tags.get("location") == "underground"})
    return {"lines": lines, "stations": stations}


def rect(points):
    """Oriented box from a footprint: direction of the longest edge, extents along and across it."""
    best, angle = -1.0, 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        d = math.dist(a, b)
        if d > best:
            best, angle = d, math.atan2(b[1] - a[1], b[0] - a[0])
    cos, sin = math.cos(angle), math.sin(angle)
    along = [p[0] * cos + p[1] * sin for p in points]
    across = [-p[0] * sin + p[1] * cos for p in points]
    mid_a, mid_c = (min(along) + max(along)) / 2, (min(across) + max(across)) / 2
    return (mid_a * cos - mid_c * sin, mid_a * sin + mid_c * cos,
            max(along) - min(along), max(across) - min(across), angle)


def fetch_towers():
    elements = overpass(f"""[out:json][timeout:160];
(
  way["building"]["building:levels"~"^([5-9]|[1-9][0-9])$"](around:{TOWER_RADIUS},{LAT0},{LON0});
  way["building"]["height"~"^([2-9][0-9]|[1-9][0-9][0-9])"](around:{TOWER_RADIUS},{LAT0},{LON0});
);
out tags geom;""")
    if elements is None:
        return None
    towers = []
    for e in elements:
        geometry = e.get("geometry")
        if e["type"] != "way" or not geometry or len(geometry) < 4:
            continue
        tags = e.get("tags", {})
        levels, height = number(tags.get("building:levels")), number(tags.get("height"))
        h = height or (levels * METRES_PER_LEVEL if levels else None)
        if not h or h > 260:   # taller than any building in the city: bad tag
            continue
        x, y, w, d, a = rect([list(proj(p["lat"], p["lon"])) for p in geometry[:-1]])
        if w < 6 or d < 4 or w > 300:
            continue
        towers.append({"x": round(x), "y": round(y), "w": round(w), "d": round(d), "a": round(a, 2), "h": round(h)})
    towers.sort(key=lambda t: -t["h"])
    return towers[:MAX_TOWERS]


# Used only when Overpass is unavailable: hand-entered, APPROXIMATE station positions (not surveyed
# or from OSM). Each station: name, latitude, longitude, underground?
SCHEMATIC_LINES = {
    "Purple Line": ("#7a4fa3", [
        ("Challaghatta", 12.8972, 77.4618, 0), ("Kengeri", 12.9080, 77.4765, 0), ("Mysuru Road", 12.9467, 77.5300, 0),
        ("Vijayanagar", 12.9709, 77.5373, 0), ("Magadi Road", 12.9757, 77.5553, 0), ("City Railway Station", 12.9757, 77.5660, 1),
        ("Majestic", 12.9757, 77.5728, 1), ("Sir M Visvesvaraya", 12.9750, 77.5835, 1), ("Vidhana Soudha", 12.9797, 77.5923, 1),
        ("Cubbon Park", 12.9810, 77.5975, 1), ("MG Road", 12.9755, 77.6067, 0), ("Trinity", 12.9731, 77.6170, 0),
        ("Halasuru", 12.9761, 77.6265, 0), ("Indiranagar", 12.9784, 77.6385, 0), ("Swami Vivekananda Road", 12.9855, 77.6450, 0),
        ("Baiyappanahalli", 12.9907, 77.6525, 0), ("KR Puram", 13.0007, 77.6750, 0), ("Mahadevapura", 12.9960, 77.6940, 0),
        ("Kundalahalli", 12.9750, 77.7150, 0), ("Whitefield (Kadugodi)", 12.9950, 77.7580, 0)]),
    "Green Line": ("#3f9b58", [
        ("Madavara", 13.0570, 77.4730, 0), ("Nagasandra", 13.0480, 77.5000, 0), ("Peenya", 13.0330, 77.5270, 0),
        ("Yeshwanthpur", 13.0230, 77.5500, 0), ("Mahalakshmi", 13.0080, 77.5490, 0), ("Rajajinagar", 13.0005, 77.5495, 0),
        ("Sampige Road", 12.9925, 77.5700, 1), ("Majestic", 12.9757, 77.5728, 1), ("Chickpete", 12.9665, 77.5745, 1),
        ("KR Market", 12.9610, 77.5745, 1), ("National College", 12.9505, 77.5735, 0), ("Jayanagar", 12.9295, 77.5800, 0),
        ("JP Nagar", 12.9075, 77.5730, 0), ("Silk Institute", 12.8620, 77.5300, 0)]),
    "Yellow Line": ("#d9b23c", [
        ("RV Road", 12.9215, 77.5800, 0), ("BTM Layout", 12.9165, 77.6100, 0), ("Central Silk Board", 12.9170, 77.6230, 0),
        ("Bommanahalli", 12.9000, 77.6200, 0), ("Electronic City", 12.8450, 77.6600, 0), ("Bommasandra", 12.8180, 77.6900, 0)]),
}


def catmull_rom(points, per_span=6):
    out = []
    for i in range(len(points) - 1):
        p0, p1, p2, p3 = points[max(i - 1, 0)], points[i], points[i + 1], points[min(i + 2, len(points) - 1)]
        for k in range(per_span):
            t = k / per_span
            out.append([round(0.5 * (2 * p1[a] + (p2[a] - p0[a]) * t + (2 * p0[a] - 5 * p1[a] + 4 * p2[a] - p3[a]) * t * t
                                     + (3 * p1[a] - p0[a] - 3 * p2[a] + p3[a]) * t ** 3)) for a in (0, 1)])
    out.append([round(points[-1][0]), round(points[-1][1])])
    return out


def schematic_metro():
    lines, stations, seen = [], [], set()
    for name, (colour, stops) in SCHEMATIC_LINES.items():
        pts = [proj(lat, lon) for _, lat, lon, _ in stops]
        path = catmull_rom(pts)
        for i in range(len(stops) - 1):   # a span is underground when both of its stations are
            span = path[i * 6:(i + 1) * 6 + 1]
            lines.append({"colour": colour, "name": name, "tunnel": bool(stops[i][3] and stops[i + 1][3]), "pts": span})
        for (station, lat, lon, under), (x, y) in zip(stops, pts):
            if station not in seen:
                seen.add(station)
                stations.append({"name": station, "x": round(x), "y": round(y), "under": bool(under)})
    return {"lines": lines, "stations": stations}


def main():
    dest = ROOT / "data" / "cities" / "bengaluru_urban.json"
    old = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    print("Namma Metro...", flush=True)
    metro = fetch_metro()
    approximate = False
    if not metro:
        if old.get("metro") and not old.get("metroApproximate"):
            metro = old["metro"]
        else:
            print("  using approximate hand-entered station positions")
            metro, approximate = schematic_metro(), True
    print("Tall buildings...", flush=True)
    towers = fetch_towers() or old.get("towers") or []
    dest.write_text(json.dumps({"metro": metro, "metroApproximate": approximate, "towers": towers},
                               separators=(",", ":")), encoding="utf-8")
    tallest = towers[0]["h"] if towers else 0
    print(f"saved {dest}: {len(metro['lines'])} metro spans, {len(metro['stations'])} stations "
          f"(approximate: {approximate}), {len(towers)} real tall buildings (tallest {tallest} m), "
          f"{dest.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
