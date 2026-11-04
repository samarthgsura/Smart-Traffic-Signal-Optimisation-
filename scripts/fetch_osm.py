"""Download Bengaluru's major road network from OpenStreetMap (Overpass API).

Road classes: motorway, trunk, primary, secondary (+ their ramps/links), plus all
traffic-signal nodes, within RADIUS_KM of the city centre. OSM data is
(c) OpenStreetMap contributors, ODbL.

Usage:  python scripts/fetch_osm.py [--radius 50] [--tertiary]
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CENTRE = (12.9716, 77.5946)  # Bengaluru (Vidhana Soudha / MG Road area)
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
OUT = Path(__file__).resolve().parents[1] / "data" / "osm"


def query(radius_m: int, tertiary: bool) -> str:
    classes = "motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link|secondary_link"
    if tertiary:
        classes += "|tertiary|tertiary_link"
    lat, lon = CENTRE
    return f"""[out:json][timeout:600];
(
  way["highway"~"^({classes})$"](around:{radius_m},{lat},{lon});
);
(._;>;);
out body;
node["highway"="traffic_signals"](around:{radius_m},{lat},{lon});
out body;
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=float, default=50)
    p.add_argument("--tertiary", action="store_true")
    a = p.parse_args()
    q = query(int(a.radius * 1000), a.tertiary)
    data = urllib.parse.urlencode({"data": q}).encode()
    OUT.mkdir(parents=True, exist_ok=True)
    for url in ENDPOINTS:
        try:
            t0 = time.time()
            print("querying", url, "...")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "neural-storm-ft01/1.0 (hackathon)"})
            raw = urllib.request.urlopen(req, timeout=900).read()
            js = json.loads(raw)
            name = f"bengaluru_{int(a.radius)}km{'_tertiary' if a.tertiary else ''}.json"
            (OUT / name).write_bytes(raw)
            ways = sum(1 for e in js["elements"] if e["type"] == "way")
            nodes = sum(1 for e in js["elements"] if e["type"] == "node")
            print(f"saved {OUT / name}: {len(raw) / 1e6:.1f} MB, {ways} ways, {nodes} nodes "
                  f"in {time.time() - t0:.0f}s")
            return
        except Exception as e:  # try the next mirror
            print("  failed:", e)
    raise SystemExit("All Overpass endpoints failed")


if __name__ == "__main__":
    main()
