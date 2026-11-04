"""City road networks for the multi-junction simulation.

Two schematic presets inspired by real corridors. Positions are in map units
(1 unit = M_PER_UNIT metres); the layout keeps each city's shape (radial vs
linear) but compresses distances so the whole network fits on one screen.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path

M_PER_UNIT = 1.5          # metres per map unit
SAT_PER_LANE = 0.45       # PCU per second of green, per lane (~1,600 PCU/h)
PCU_SPACING_M = 7.0       # road length one queued PCU occupies in a lane


@dataclass
class Node:
    id: str
    name: str
    x: float
    y: float
    kind: str  # "signal", "junction" (unsignalised) or "gateway" (trip zone)
    name_is_landmark: bool = False


@dataclass
class Link:
    id: str
    src: str
    dst: str
    lanes: int
    speed: float      # free-flow speed, m/s
    length: float     # metres
    road: str         # shared id for both directions of one road
    name: str
    geom: list | None = None   # polyline [[x, y], ...] in map units; straight line if None
    cls: str = ""              # road class (motorway, trunk, primary, ...)
    level: int = 0             # 1 = flyover / bridge, -1 = underpass / tunnel
    ramp: bool = False

    @property
    def capacity_pcu(self) -> float:
        return max(4.0, self.length * self.lanes / PCU_SPACING_M)

    @property
    def free_time(self) -> float:
        return self.length / self.speed

    @property
    def saturation(self) -> float:
        return self.lanes * SAT_PER_LANE


@dataclass
class OD:
    origin: str
    dest: str
    vph: float
    tags: tuple[str, ...] = ()


@dataclass
class CityGraph:
    key: str
    name: str
    description: str
    rush_label: str
    rush_tag: str
    nodes: dict[str, Node] = field(default_factory=dict)
    links: dict[str, Link] = field(default_factory=dict)
    ods: list[OD] = field(default_factory=list)
    ambulance_routes: list[tuple[str, str]] = field(default_factory=list)
    m_per_unit: float = M_PER_UNIT
    extra: dict = field(default_factory=dict)   # features, attribution, ... passed to the UI
    _routes: dict = field(default_factory=dict, repr=False)
    _out: dict | None = field(default=None, repr=False)
    _in: dict | None = field(default=None, repr=False)

    # ---- building ---------------------------------------------------------
    def node(self, id, name, x, y, kind="signal"):
        self.nodes[id] = Node(id, name, x, y, kind, kind == "signal")

    def road(self, a, b, lanes, kmh, name, lanes_back=None):
        na, nb = self.nodes[a], self.nodes[b]
        length = ((na.x - nb.x) ** 2 + (na.y - nb.y) ** 2) ** 0.5 * M_PER_UNIT
        road_id = f"{a}-{b}"
        for s, d, ln in ((a, b, lanes), (b, a, lanes_back or lanes)):
            lid = f"{s}>{d}"
            self.links[lid] = Link(lid, s, d, ln, kmh / 3.6, round(length, 1), road_id, name)
        self._out = self._in = None

    def od(self, o, d, vph, *tags):
        self.ods.append(OD(o, d, vph, tuple(tags)))

    # ---- queries ----------------------------------------------------------
    @property
    def signal_ids(self) -> list[str]:
        return [n.id for n in self.nodes.values() if n.kind == "signal"]

    def _adj(self):
        if self._out is None:
            self._out, self._in = {n: [] for n in self.nodes}, {n: [] for n in self.nodes}
            for l in self.links.values():
                self._out[l.src].append(l.id)
                self._in[l.dst].append(l.id)
        return self._out, self._in

    def in_links(self, node_id) -> list[str]:
        return self._adj()[1][node_id]

    def out_links(self, node_id) -> list[str]:
        return self._adj()[0][node_id]

    @property
    def junction_ids(self) -> list[str]:
        return [n.id for n in self.nodes.values() if n.kind == "junction"]

    def neighbors(self, node_id, max_hops: int = 4) -> list[str]:
        """Signal peers reachable by road without passing another signal (the P2P mesh links).
        Unsignalised junctions in between are passed through, up to max_hops links."""
        found, frontier, seen = set(), [node_id], {node_id}
        for _ in range(max_hops):
            nxt = []
            for n in frontier:
                for lid in self.out_links(n):
                    d = self.links[lid].dst
                    if d in seen:
                        continue
                    seen.add(d)
                    kind = self.nodes[d].kind
                    if kind == "signal":
                        found.add(d)
                    elif kind == "junction":
                        nxt.append(d)
            frontier = nxt
        return sorted(found)

    def route(self, origin, dest) -> list[str]:
        """Fastest free-flow path as a list of link ids (Dijkstra). Never passes through gateways."""
        key = (origin, dest)
        if key in self._routes:
            return self._routes[key]
        dist = {origin: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, origin)]
        while pq:
            d, n = heapq.heappop(pq)
            if n == dest:
                break
            if d > dist.get(n, 1e18):
                continue
            if n != origin and self.nodes[n].kind == "gateway":
                continue
            for lid in self.out_links(n):
                l = self.links[lid]
                nd = d + l.free_time + (8 if self.nodes[l.dst].kind != "junction" else 4)
                if nd < dist.get(l.dst, 1e18):
                    dist[l.dst] = nd
                    prev[l.dst] = lid
                    heapq.heappush(pq, (nd, l.dst))
        if dest not in prev:
            raise ValueError(f"No route {origin} -> {dest}")
        path, n = [], dest
        while n != origin:
            lid = prev[n]
            path.append(lid)
            n = self.links[lid].src
        path.reverse()
        self._routes[key] = path
        return path

    def to_json(self) -> dict:
        def geom(l):
            if l.geom:
                return l.geom
            a, b = self.nodes[l.src], self.nodes[l.dst]
            return [[a.x, a.y], [b.x, b.y]]
        return {
            "key": self.key, "name": self.name, "description": self.description,
            "rushLabel": self.rush_label, "rushTag": self.rush_tag, "mPerUnit": self.m_per_unit,
            "nodes": [{"id": n.id, "name": n.name, "x": n.x, "y": n.y, "kind": n.kind,
                       "landmark": n.name_is_landmark} for n in self.nodes.values()],
            "links": [{"id": l.id, "src": l.src, "dst": l.dst, "lanes": l.lanes, "speed": l.speed,
                       "length": l.length, "road": l.road, "name": l.name, "cls": l.cls,
                       "level": l.level, "ramp": l.ramp, "geom": geom(l),
                       "capacity": round(l.capacity_pcu, 1), "freeTime": round(l.free_time, 2),
                       "sat": l.saturation}
                      for l in self.links.values()],
            **self.extra,
            "ods": [{"origin": o.origin, "dest": o.dest, "vph": o.vph, "tags": list(o.tags),
                     "route": self.route(o.origin, o.dest)} for o in self.ods],
            "ambulanceRoutes": [{"origin": o, "dest": d, "route": self.route(o, d),
                                 "label": f"{self.nodes[o].name} → {self.nodes[d].name}"}
                                for o, d in self.ambulance_routes],
            "neighbors": {n: self.neighbors(n) for n in self.signal_ids},
        }


def bengaluru() -> CityGraph:
    g = CityGraph(
        "bengaluru", "Bengaluru",
        "Radial roads into the CBD crossed by the Outer Ring Road IT corridor. "
        "Silk Board, KR Puram and Marathahalli are merging chokepoints.",
        "IT Park 6 PM Rush", "it")
    # signals
    g.node("HEB", "Hebbal", 420, 130)
    g.node("MGR", "MG Road", 270, 370)
    g.node("TIN", "Tin Factory", 600, 250)
    g.node("KRP", "KR Puram", 770, 205)
    g.node("DOM", "Domlur", 460, 390)
    g.node("MTH", "Marathahalli", 770, 400)
    g.node("BEL", "Bellandur", 700, 540)
    g.node("SLK", "Silk Board", 450, 590)
    # gateways
    g.node("G_AIR", "Airport (NH44)", 420, 10, "gateway")
    g.node("G_YPR", "Yeshwanthpur", 120, 120, "gateway")
    g.node("G_CBD", "Majestic / CBD", 80, 380, "gateway")
    g.node("G_HSK", "Hoskote (NH75)", 960, 150, "gateway")
    g.node("G_WFD", "Whitefield ITPL", 960, 400, "gateway")
    g.node("G_SRJ", "Sarjapur Road", 900, 660, "gateway")
    g.node("G_ECY", "Electronic City", 450, 700, "gateway")
    # roads
    g.road("G_AIR", "HEB", 3, 50, "Bellary Road (NH44)")
    g.road("G_YPR", "HEB", 2, 40, "Outer Ring Road (west)")
    g.road("HEB", "KRP", 3, 45, "Outer Ring Road (Nagawara)")
    g.road("HEB", "MGR", 2, 35, "Bellary Road (Mekhri Circle)")
    g.road("G_CBD", "MGR", 2, 30, "MG Road / Majestic")
    g.road("MGR", "TIN", 2, 35, "Old Madras Road")
    g.road("TIN", "KRP", 2, 35, "Old Madras Road (KR Puram bridge)")
    g.road("KRP", "G_HSK", 2, 45, "NH75 to Hoskote")
    g.road("KRP", "MTH", 3, 40, "Outer Ring Road (Mahadevapura)")
    g.road("MTH", "G_WFD", 2, 35, "Whitefield Main Road")
    g.road("MGR", "DOM", 2, 35, "Old Airport Road")
    g.road("DOM", "MTH", 2, 35, "Old Airport Road (HAL)")
    g.road("MTH", "BEL", 3, 40, "Outer Ring Road (Kadubeesanahalli)")
    g.road("BEL", "G_SRJ", 2, 35, "Sarjapur Road")
    g.road("BEL", "SLK", 3, 40, "Outer Ring Road (HSR)")
    g.road("DOM", "SLK", 2, 30, "Inner Ring Road (Koramangala)")
    g.road("SLK", "G_ECY", 3, 45, "Hosur Road / E-City flyover")
    # demand (vehicles/hour); tag "it" = evening IT-park outflow
    g.od("G_ECY", "G_CBD", 420, "it")
    g.od("G_ECY", "G_AIR", 210, "it")
    g.od("G_WFD", "G_CBD", 370, "it")
    g.od("G_WFD", "G_YPR", 240, "it")
    g.od("G_SRJ", "G_CBD", 240, "it")
    g.od("G_SRJ", "G_HSK", 160, "it")
    g.od("G_WFD", "G_ECY", 210, "it")
    g.od("G_AIR", "G_ECY", 260)
    g.od("G_AIR", "G_WFD", 220)
    g.od("G_HSK", "G_CBD", 260)
    g.od("G_HSK", "G_ECY", 190)
    g.od("G_CBD", "G_WFD", 240)
    g.od("G_CBD", "G_ECY", 240)
    g.od("G_YPR", "G_WFD", 190)
    g.od("G_YPR", "G_CBD", 160)
    g.ambulance_routes = [("G_ECY", "G_AIR"), ("G_WFD", "G_CBD"), ("G_HSK", "G_ECY"),
                          ("G_SRJ", "G_YPR")]
    return g


def mumbai() -> CityGraph:
    g = CityGraph(
        "mumbai", "Mumbai",
        "One north-south spine (Western Express Highway) with the Bandra-Worli Sea Link "
        "as a fast parallel and narrow east-west connectors. Morning traffic is tidal.",
        "BKC Office 6 PM Rush", "office")
    # signals
    g.node("AND", "Andheri (WEH)", 470, 110)
    g.node("VLP", "Vile Parle / Airport", 470, 220)
    g.node("KLN", "Kalanagar, Bandra", 490, 330)
    g.node("BRC", "Bandra Reclamation (BWSL)", 300, 375)
    g.node("MAH", "Mahim", 470, 450)
    g.node("SIO", "Sion", 730, 470)
    g.node("DTT", "Dadar TT", 490, 565)
    g.node("WOR", "Worli (BWSL south)", 290, 600)
    # gateways
    g.node("G_BOR", "Borivali (WEH north)", 470, 10, "gateway")
    g.node("G_JVL", "Powai / JVLR", 900, 110, "gateway")
    g.node("G_JUH", "Juhu", 160, 220, "gateway")
    g.node("G_BKC", "BKC offices", 740, 320, "gateway")
    g.node("G_EEH", "Eastern Express Hwy", 950, 470, "gateway")
    g.node("G_SOB", "South Mumbai", 410, 700, "gateway")
    # roads
    g.road("G_BOR", "AND", 4, 50, "Western Express Highway")
    g.road("AND", "G_JVL", 1, 25, "Andheri-Kurla Road / JVLR")
    g.road("AND", "VLP", 4, 50, "Western Express Highway")
    g.road("VLP", "G_JUH", 1, 25, "Juhu Tara Road")
    g.road("VLP", "KLN", 4, 50, "Western Express Highway")
    g.road("KLN", "G_BKC", 2, 30, "BKC Connector")
    g.road("KLN", "BRC", 2, 35, "Bandra Reclamation Road")
    g.road("KLN", "MAH", 2, 30, "SV Road / Mahim Causeway")
    g.road("KLN", "SIO", 1, 25, "Sion-Bandra Link Road")
    g.road("BRC", "WOR", 4, 80, "Bandra-Worli Sea Link")
    g.road("WOR", "G_SOB", 2, 35, "Annie Besant Road")
    g.road("MAH", "DTT", 2, 30, "LJ Road / Senapati Bapat Marg")
    g.road("DTT", "G_SOB", 2, 30, "Dr Ambedkar Road")
    g.road("DTT", "SIO", 1, 25, "Dadar-Sion connector")
    g.road("WOR", "DTT", 1, 25, "Worli Naka connector")
    g.road("SIO", "G_EEH", 3, 45, "Eastern Express Highway")
    # demand; tag "am" = southbound morning tide, "office" = evening outflow from BKC / SoBo
    g.od("G_BOR", "G_SOB", 950, "am")
    g.od("G_BOR", "G_BKC", 520, "am")
    g.od("G_JVL", "G_SOB", 220, "am")
    g.od("G_JUH", "G_BKC", 220, "am")
    g.od("G_EEH", "G_SOB", 380, "am")
    g.od("G_EEH", "G_BKC", 240, "am")
    g.od("G_BKC", "G_BOR", 220, "office")
    g.od("G_BKC", "G_EEH", 200, "office")
    g.od("G_SOB", "G_BOR", 260, "office")
    g.od("G_SOB", "G_EEH", 200, "office")
    g.od("G_SOB", "G_JVL", 120, "office")
    g.ambulance_routes = [("G_BOR", "G_SOB"), ("G_JVL", "G_SOB"), ("G_EEH", "G_BKC"),
                          ("G_SOB", "G_BOR")]
    return g


CITY_DIR = Path(__file__).resolve().parents[1] / "data" / "cities"


def from_json(path: Path) -> CityGraph:
    """Load a network built by scripts/build_osm_city.py (real OpenStreetMap roads)."""
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    g = CityGraph(j["key"], j["name"], j["description"], j["rushLabel"], j["rushTag"],
                  m_per_unit=j.get("mPerUnit", 1.0),
                  extra={k: j[k] for k in ("features", "attribution", "radius") if k in j})
    for n in j["nodes"]:
        g.nodes[n["id"]] = Node(n["id"], n["name"], n["x"], n["y"], n["kind"],
                                bool(n.get("landmark")))
    for l in j["links"]:
        g.links[l["id"]] = Link(l["id"], l["src"], l["dst"], l["lanes"], l["speed"], l["length"],
                                l["road"], l["name"], l.get("geom"), l.get("cls", ""),
                                l.get("level", 0), l.get("ramp", False))
    for o in j["ods"]:
        g.ods.append(OD(o["origin"], o["dest"], o["vph"], tuple(o["tags"])))
        g._routes[(o["origin"], o["dest"])] = o["route"]
    for a in j["ambulanceRoutes"]:
        g.ambulance_routes.append((a["origin"], a["dest"]))
        g._routes[(a["origin"], a["dest"])] = a["route"]
    return g


def bengaluru_osm() -> CityGraph:
    path = CITY_DIR / "bengaluru_osm.json"
    if not path.exists():
        raise FileNotFoundError("Run: python scripts/fetch_osm.py && python scripts/build_osm_city.py")
    return from_json(path)


PRESETS = {"bengaluru": bengaluru, "mumbai": mumbai, "bengaluru_osm": bengaluru_osm}


def available() -> list[str]:
    """Presets that can be loaded right now (the real-map one needs its data file)."""
    return [k for k in PRESETS if k != "bengaluru_osm" or (CITY_DIR / "bengaluru_osm.json").exists()]


def load(key: str) -> CityGraph:
    return PRESETS[key]()
