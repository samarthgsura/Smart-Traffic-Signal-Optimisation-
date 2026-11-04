"""City-scale traffic simulation (mesoscopic, 1-second steps).

Each directed road link holds a FIFO of vehicles. A vehicle reaches the stop line
after the link's free-flow travel time, then waits for green. A green approach
discharges at its saturation flow, but only while the next link on the
vehicle's route has room: when a link fills up, traffic behind it is blocked
(spillback), which is how gridlock spreads through a real network.

Three control modes share this engine:
  fixed     30 s fixed timer at every junction
  isolated  the AdaptiveController at every junction, using only its own camera
  p2p       the same controller inside a MeshNode that also hears its neighbours
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config as C
from .city_graph import CityGraph
from .optimizer import AdaptiveController, FixedTimeController, Observation
from .p2p_mesh import (BACKPRESSURE, BACKPRESSURE_LEVEL, PLATOON, PRIORITY, STATE, MeshBus,
                       MeshNode, Packet)

MODES = ("fixed", "isolated", "p2p")
MODE_LABELS = {"fixed": "Fixed timer (30 s)", "isolated": "Isolated AI", "p2p": "P2P Mesh AI"}
JAM_LEVEL = 0.95


@dataclass
class Veh:
    vid: int
    kind: str
    pcu: float
    route: list[str]
    spawn_t: float
    free_time: float
    emergency: bool = False
    idx: int = 0
    enter_t: float = 0.0
    ready_t: float = 0.0
    done_t: float | None = None


class LinkState:
    __slots__ = ("link", "q", "occ", "speed_factor", "sat_factor", "is_exit", "credit", "n_emg")

    def __init__(self, link, is_exit):
        self.link = link
        self.q: deque[Veh] = deque()
        self.occ = 0.0
        self.speed_factor = 1.0
        self.sat_factor = 1.0
        self.is_exit = is_exit
        self.credit = 0.0  # discharge credit at an unsignalised junction
        self.n_emg = 0     # ambulances on this link

    def give_way(self, t):
        """Traffic pulls aside: an ambulance that has reached the stop line moves to the front."""
        if not self.n_emg or not self.q or self.q[0].emergency:
            return
        for i, v in enumerate(self.q):
            if v.ready_t > t:
                return
            if v.emergency:
                del self.q[i]
                self.q.appendleft(v)
                return

    def pop(self):
        v = self.q.popleft()
        self.occ -= v.pcu
        if v.emergency:
            self.n_emg -= 1
        return v

    def room(self, pcu):
        return self.occ + pcu <= self.link.capacity_pcu or not self.q


class Signal:
    def __init__(self, node_id, phases):
        self.id = node_id
        self.phases = phases
        self.current = 0
        self.elapsed = 0
        self.intergreen = 0
        self.pending = 0
        self.credit = 0.0
        self.last_green = [0.0] * len(phases)
        self.state = 0  # phase index being served, or -1 during amber/all-red


def efficiency_score(m: dict) -> int:
    """City Efficiency Score, 0-1000.

    50% travel-time index (delay relative to the free-flow trip time, so long cross-city
    trips aren't penalised for being long), 30% how much demand could enter the network
    (backlog at the city's entry points means gridlock), 20% share of roads not jammed."""
    delay = max(0.0, 1 - m["delay_ratio"] / 1.5)
    entry = min(1.0, m["entered_pcu"] / max(1.0, m["spawned_pcu"]))
    jam = max(0.0, 1 - 5 * m["jam_ratio"])
    return int(round(1000 * (0.5 * delay + 0.3 * entry + 0.2 * jam)))


RANKS = [(850, "Diamond · Gridlock Slayer"), (700, "Platinum · Corridor Commander"),
         (550, "Gold · Signal Strategist"), (400, "Silver · Junction Marshal"),
         (0, "Bronze · Traffic Rookie")]


def rank_for(score: int) -> str:
    return next(name for floor, name in RANKS if score >= floor)


class CitySim:
    def __init__(self, graph: CityGraph, mode: str = "p2p", seed: int = 0,
                 demand_scale: float = 1.0, fallback: str = "isolated"):
        assert mode in MODES
        self.g, self.mode, self.fallback = graph, mode, fallback
        self.demand_scale = demand_scale
        self.rng = np.random.default_rng(seed)        # arrivals only: identical in every mode
        self.t = 0
        self.links = {lid: LinkState(l, graph.nodes[l.dst].kind == "gateway")
                      for lid, l in graph.links.items()}
        self.entry: dict[str, deque[Veh]] = {}
        self.signals = {n: Signal(n, graph.in_links(n)) for n in graph.signal_ids}
        self.junctions = {n: list(graph.in_links(n)) for n in graph.junction_ids}
        self.exit_links = [ls for ls in self.links.values() if ls.is_exit]
        self.n_links = sum(1 for ls in self.links.values() if not ls.is_exit)
        self.controllers = {n: self._controller(n, mode) for n in self.signals}
        self.bus = MeshBus()
        self.mesh = {n: MeshNode(n, s.phases, self._next_hop) for n, s in self.signals.items()}
        self.killed: set[str] = set()
        self.rush_until = -1
        self.rush_mult = 1.0
        self._vid = 0
        self.vehicles_done: list[Veh] = []
        self.spawned_pcu = 0.0
        self.jam_link_seconds = 0
        self.blocked_seconds = 0

    # ---- setup helpers ----------------------------------------------------
    def _controller(self, node_id, mode):
        n = len(self.g.in_links(node_id))
        if mode == "fixed":
            return FixedTimeController([30] * n)
        return AdaptiveController()

    def _next_hop(self, route, max_links: int = 8):
        """Next signal along a route: (signal, its incoming link, seconds to reach it, rest of route)."""
        secs = 0.0
        for i, lid in enumerate(route[:max_links]):
            ls = self.links[lid]
            secs += ls.link.free_time / ls.speed_factor
            kind = self.g.nodes[ls.link.dst].kind
            if kind == "signal":
                return ls.link.dst, lid, secs, route[i + 1:]
            if kind != "junction":
                break
        return None, None, 0.0, []

    # ---- events -----------------------------------------------------------
    def trigger_rush(self, duration=900, mult=2.0):
        self.rush_until, self.rush_mult = self.t + duration, mult

    def monsoon(self, road_id, factor=0.5, on=True):
        for ls in self.links.values():
            if ls.link.road == road_id:
                ls.speed_factor = factor if on else 1.0
                ls.sat_factor = factor if on else 1.0

    def kill(self, node_id):
        self.killed.add(node_id)
        self.bus.dead.add(node_id)
        self.controllers[node_id] = self._controller(node_id, self.fallback)

    def revive(self, node_id):
        self.killed.discard(node_id)
        self.bus.dead.discard(node_id)
        self.controllers[node_id] = self._controller(node_id, self.mode)

    def spawn_ambulance(self, origin, dest):
        self._spawn(origin, dest, "ambulance", emergency=True)

    # ---- core -------------------------------------------------------------
    def _uses_mesh(self, node_id):
        return self.mode == "p2p" and node_id not in self.killed

    def _spawn(self, origin, dest, kind, emergency=False):
        route = self.g.route(origin, dest)
        free = sum(self.g.links[l].free_time for l in route)
        self._vid += 1
        v = Veh(self._vid, kind, C.PCU[kind], route, self.t, free, emergency)
        self.entry.setdefault(route[0], deque()).append(v)
        self.spawned_pcu += v.pcu

    def _enter(self, v: Veh, lid: str):
        ls = self.links[lid]
        v.enter_t = self.t
        tt = ls.link.free_time / ls.speed_factor
        v.ready_t = max(self.t + tt, ls.q[-1].ready_t if ls.q else 0.0)
        ls.q.append(v)
        ls.occ += v.pcu
        if v.emergency:
            ls.n_emg += 1
        if v.emergency and self.mode == "p2p":
            dst, link, secs, rest = self._next_hop(v.route[v.idx:])
            if dst is None:
                return
            src = ls.link.src
            sender = src if src in self.signals and self._uses_mesh(src) else dst
            # the peer behind the ambulance (or the camera at the next signal) announces it
            eta = (v.ready_t - self.t) + secs - tt
            p = Packet(PRIORITY, sender, dst, self.t, link_id=link, is_emergency=True,
                       eta_seconds=eta, vehicle_id=v.vid, route=rest)
            if sender == dst:
                for f in self.mesh[dst].receive(p, self.t) if self._uses_mesh(dst) else []:
                    self.bus.send(f)
            else:
                self.bus.send(p)

    def _arrival_mult(self, od):
        mult = self.demand_scale
        if self.t < self.rush_until and self.g.rush_tag in od.tags:
            mult *= self.rush_mult
        return mult

    def step(self):
        t = self.t
        # 1. mesh delivery
        if self.mode == "p2p":
            for p in self.bus.deliver():
                if p.target_id in self.killed:
                    continue
                for f in self.mesh[p.target_id].receive(p, t):
                    self.bus.send(f)
            for n, m in self.mesh.items():
                m.prune(t)
        # 2. new vehicles (same random stream in every mode)
        kinds = list(C.DEFAULT_MIX)
        shares = np.array([C.DEFAULT_MIX[k] for k in kinds])
        for od in self.g.ods:
            n = self.rng.poisson(od.vph / 3600 * self._arrival_mult(od))
            for _ in range(n):
                self._spawn(od.origin, od.dest, kinds[self.rng.choice(len(kinds), p=shares)])
        # 3. entry buffers onto the network
        for lid, buf in self.entry.items():
            ls = self.links[lid]
            while buf and ls.room(buf[0].pcu):
                self._enter(buf.popleft(), lid)
        # 4. vehicles leaving the city
        for ls in self.exit_links:
            ls.give_way(t)
            while ls.q and ls.q[0].ready_t <= t:
                v = ls.pop()
                v.done_t = t
                self.vehicles_done.append(v)
        # 5. signals, then unsignalised junctions
        for sig in self.signals.values():
            self._signal_step(sig)
        for ins in self.junctions.values():
            self._junction_step(ins)
        # 6. state gossip + backpressure warnings every 5 s
        if self.mode == "p2p" and t % 5 == 0:
            for n in self.signals:
                if not self._uses_mesh(n):
                    continue
                for lid in self.signals[n].phases:
                    ls = self.links[lid]
                    src = ls.link.src
                    fill = ls.occ / ls.link.capacity_pcu
                    if src not in self.signals:
                        continue
                    kind = BACKPRESSURE if fill > BACKPRESSURE_LEVEL else STATE
                    self.bus.send(Packet(kind, n, src, t, link_id=lid, fill=round(fill, 2)))
        # 7. jam accounting (sampled every 5 s)
        if t % 5 == 0:
            for ls in self.links.values():
                if not ls.is_exit and ls.occ >= JAM_LEVEL * ls.link.capacity_pcu:
                    self.jam_link_seconds += 5
        self.t += 1

    def _junction_step(self, ins):
        """Unsignalised junction: every approach flows at its saturation rate while the next
        road has room (rotating order so no approach always goes first)."""
        t = self.t
        k = t % len(ins)
        for lid in ins[k:] + ins[:k]:
            ls = self.links[lid]
            if not ls.q or ls.q[0].ready_t > t:
                ls.credit = 0.0
                continue
            sat = ls.link.saturation * ls.sat_factor
            ls.credit += sat
            ls.give_way(t)
            blocked = False
            while ls.q and ls.q[0].ready_t <= t and ls.q[0].pcu <= ls.credit + 1e-9:
                v = ls.q[0]
                nxt = v.route[v.idx + 1]
                if not self.links[nxt].room(v.pcu):
                    blocked = True
                    break
                ls.pop()
                ls.credit -= v.pcu
                v.idx += 1
                self._enter(v, nxt)
            if blocked or not ls.q or ls.q[0].ready_t > t:
                ls.credit = min(ls.credit, sat)

    def _observe(self, sig: Signal):
        t = self.t
        for lid in sig.phases:
            self.links[lid].give_way(t)  # ambulances pull ahead, so the cameras see them first
        q_pcu, counts, oldest, emerg, blocked = [], [], [], [], []
        mesh = self.mesh[sig.id] if self._uses_mesh(sig.id) else None
        for lid in sig.phases:
            pcu = n = blk = 0.0
            amb = False
            q = self.links[lid].q
            for v in q:
                if v.ready_t > t:
                    break
                pcu += v.pcu
                n += 1
                amb = amb or v.emergency
                if mesh is not None:
                    blk += v.pcu * mesh.downstream_block(v.route[v.idx + 1])
            q_pcu.append(pcu)
            counts.append(int(n))
            oldest.append(t - q[0].ready_t if q and q[0].ready_t <= t else 0.0)
            emerg.append(amb)
            blocked.append(blk / pcu if pcu else 0.0)
        red_for = [0.0 if i == sig.current else t - sig.last_green[i]
                   for i in range(len(sig.phases))]
        return Observation(t, q_pcu, counts, oldest, emerg, sig.current, sig.elapsed, red_for), blocked

    def _signal_step(self, sig: Signal):
        t = self.t
        if sig.intergreen > 0:
            sig.intergreen -= 1
            if sig.intergreen == 0:
                sig.current, sig.elapsed, sig.credit = sig.pending, 0, 0.0
            sig.state = -1
            return
        obs, blocked = self._observe(sig)
        if self._uses_mesh(sig.id):
            want = self.mesh[sig.id].decide(t, obs, blocked)
        else:
            want = self.controllers[sig.id].decide(obs)
        if want != sig.current:
            sig.intergreen, sig.pending, sig.state = C.INTERGREEN, want, -1
            return
        sig.state = sig.current
        sig.last_green[sig.current] = t + 1
        if sig.elapsed >= C.STARTUP_LOST:
            ls = self.links[sig.phases[sig.current]]
            ls.give_way(t)
            sat = ls.link.saturation * ls.sat_factor
            sig.credit += sat
            sent: dict[tuple, list] = {}
            blocked_now = False
            while ls.q and ls.q[0].ready_t <= t and ls.q[0].pcu <= sig.credit + 1e-9:
                v = ls.q[0]
                nxt = v.route[v.idx + 1]
                if not self.links[nxt].room(v.pcu):
                    blocked_now = True
                    break
                ls.pop()
                sig.credit -= v.pcu
                v.idx += 1
                self._enter(v, nxt)
                if self._uses_mesh(sig.id):
                    dst, link, secs, _ = self._next_hop(v.route[v.idx:], 6)
                    if dst is not None and dst != sig.id:
                        acc = sent.setdefault((dst, link), [0.0, secs])
                        acc[0] += v.pcu
                        acc[1] = min(acc[1], secs)
                if v.emergency:
                    self.mesh[sig.id].ambulance_passed(v.vid)
            if blocked_now:
                self.blocked_seconds += 1
            if blocked_now or not ls.q or ls.q[0].ready_t > t:
                sig.credit = min(sig.credit, sat)  # unused green can't be banked
            for (dst, link), (pcu, eta) in sent.items():
                self.bus.send(Packet(PLATOON, sig.id, dst, t, link_id=link,
                                     platoon_pcu=round(pcu, 1), eta_seconds=round(eta, 1)))
        sig.elapsed += 1

    def run(self, duration: int, events: list | None = None):
        events = sorted(events or [], key=lambda e: e[0])
        k = 0
        for _ in range(duration):
            while k < len(events) and events[k][0] <= self.t:
                events[k][1](self)
                k += 1
            self.step()
        return self.metrics()

    # ---- results ----------------------------------------------------------
    def metrics(self) -> dict:
        delays, amb, done_pcu, free_total, waiting_pcu = [], [], 0.0, 0.0, 0.0
        for v in self.vehicles_done:
            d = (v.done_t - v.spawn_t) - v.free_time
            delays.append(d)
            free_total += v.free_time
            done_pcu += v.pcu
            if v.emergency:
                amb.append(d)
        # vehicles still in the network count with the delay they have built up so far
        for ls in self.links.values():
            for v in ls.q:
                done_free = sum(self.g.links[l].free_time for l in v.route[:v.idx])
                cur_free = self.g.links[v.route[v.idx]].free_time
                delays.append(max(0.0, (self.t - v.spawn_t) - done_free
                                  - min(self.t - v.enter_t, cur_free)))
                free_total += done_free + min(self.t - v.enter_t, cur_free)
        for buf in self.entry.values():
            for v in buf:
                delays.append(self.t - v.spawn_t)
                waiting_pcu += v.pcu
        delay_ratio = float(sum(delays) / max(1.0, free_total))
        delays = np.array(delays) if delays else np.zeros(1)
        n_links = self.n_links
        m = {
            "trips_done": len(self.vehicles_done),
            "completed_pcu": round(done_pcu, 1),
            "spawned_pcu": round(self.spawned_pcu, 1),
            "throughput_pcu_h": round(done_pcu * 3600 / max(1, self.t), 0),
            "avg_delay_s": float(delays.mean()),
            "p95_delay_s": float(np.percentile(delays, 95)),
            "ambulance_delay_s": float(np.mean(amb)) if amb else math.nan,
            "jam_ratio": self.jam_link_seconds / max(1, n_links * self.t),
            "delay_ratio": round(delay_ratio, 3),
            "entered_pcu": round(self.spawned_pcu - waiting_pcu, 1),
            "blocked_green_s": self.blocked_seconds,
            "packets": dict(self.bus.sent),
        }
        m["score"] = efficiency_score(m)
        m["rank"] = rank_for(m["score"])
        return m
