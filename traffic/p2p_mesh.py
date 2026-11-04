"""Peer-to-peer signal mesh.

Every signal is an autonomous peer that talks only to the junctions it shares a
road with; there is no central server. Packet types:

  STATE_GOSSIP       every 5 s: "my approach from you is X% full" (state gossip).
  PLATOON            "I just released N PCU toward you; they reach your stop line in T s."
  BACKPRESSURE_WARN  "My approach from you is over 80% full; stop feeding it."
  PRIORITY_CORRIDOR  "Ambulance heading your way, ETA T s." Forwarded hop by hop
                     along the ambulance's route so signals turn green ahead of it.

A MeshNode wraps the same AdaptiveController used for isolated junctions and
adjusts its inputs with what the neighbours report.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field

from .optimizer import AdaptiveController, Observation

STATE = "STATE_GOSSIP"
PLATOON = "PLATOON"
BACKPRESSURE = "BACKPRESSURE_WARN"
PRIORITY = "PRIORITY_CORRIDOR"

BACKPRESSURE_LEVEL = 0.8   # approach fill ratio that triggers a warning
WARN_TTL = 6               # a warning stays valid this many seconds unless refreshed
STATE_TTL = 12             # gossiped fill levels expire after this many seconds
FILL_FREE = 0.5            # downstream fill below this doesn't slow us down at all
FILL_FULL = 0.95           # ...at this fill, traffic heading there counts as fully blocked
PLATOON_HORIZON = 15       # platoons arriving within this many seconds count as demand
HOLD_HORIZON = 8           # keep a green open for a platoon arriving this soon
HOLD_RATIO = 0.5           # ...if it is at least this big relative to the largest other queue
PLATOON_WEIGHT = 0.3        # how much a reported incoming platoon counts as queued demand
PREEMPT_LEAD = 20          # start the green this long before an ambulance arrives
CORRIDOR_HOPS = 3          # announce the ambulance only this many signals ahead (rolling corridor)
LOCK_STALE = 30            # drop a lock if the ambulance is this late and nobody refreshed it
EMERGENCY_MIN_GREEN = 5    # shortest green we cut to when preempting for an ambulance


@dataclass
class Packet:
    kind: str
    sender_id: str
    target_id: str
    t: int
    link_id: str = ""
    platoon_pcu: float = 0.0
    eta_seconds: float = 0.0
    is_emergency: bool = False
    vehicle_id: int = -1
    fill: float = 0.0
    route: list[str] = field(default_factory=list)  # remaining links (corridor cascade)
    hops: int = 0                                     # how far ahead of the ambulance this is

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("route")
        return d


class MeshBus:
    """Delivers packets between peers with one simulation step of latency."""

    def __init__(self, log_size: int = 500):
        self._outbox: list[Packet] = []
        self.dead: set[str] = set()
        self.log: deque[Packet] = deque(maxlen=log_size)
        self.sent = Counter()
        self.dropped = 0

    def send(self, p: Packet):
        if p.sender_id in self.dead or p.target_id in self.dead:
            self.dropped += 1
            return
        self._outbox.append(p)
        self.sent[p.kind] += 1
        self.log.append(p)

    def deliver(self) -> list[Packet]:
        out, self._outbox = self._outbox, []
        return out


class MeshNode:
    def __init__(self, node_id: str, in_links: list[str], next_hop):
        self.id = node_id
        self.in_links = in_links
        self.next_hop = next_hop  # fn(route) -> (next signal, its incoming link, seconds away, rest of route)
        self.base = AdaptiveController()
        self.platoons: dict[str, list[tuple[float, float]]] = {l: [] for l in in_links}
        self.warned: dict[str, float] = {}           # my outgoing link -> warning expiry time
        self.fill: dict[str, tuple[float, float]] = {}  # my outgoing link -> (fill, expiry)
        self.locks: dict[int, tuple[str, float]] = {}  # ambulance id -> (my incoming link, eta)
        self.received = Counter()

    # ---- inbox -------------------------------------------------------------
    def receive(self, p: Packet, t: int) -> list[Packet]:
        self.received[p.kind] += 1
        if p.kind == PLATOON and p.link_id in self.platoons:
            self.platoons[p.link_id].append((t + p.eta_seconds, p.platoon_pcu))
        elif p.kind == STATE:
            self.fill[p.link_id] = (p.fill, t + STATE_TTL)
        elif p.kind == BACKPRESSURE:
            self.warned[p.link_id] = t + WARN_TTL
            self.fill[p.link_id] = (p.fill, t + STATE_TTL)
        elif p.kind == PRIORITY and p.link_id in self.platoons:
            first = p.vehicle_id not in self.locks
            self.locks[p.vehicle_id] = (p.link_id, t + p.eta_seconds)
            if first and p.route and p.hops + 1 < CORRIDOR_HOPS:  # cascade a few signals ahead
                dst, link, secs, rest = self.next_hop(p.route)
                if dst is not None:
                    return [Packet(PRIORITY, self.id, dst, t, link_id=link, is_emergency=True,
                                   eta_seconds=p.eta_seconds + secs + 3,
                                   vehicle_id=p.vehicle_id, route=rest, hops=p.hops + 1)]
        return []

    def ambulance_passed(self, vid: int):
        self.locks.pop(vid, None)

    def prune(self, t: int):
        for vid in [v for v, (_, eta) in self.locks.items() if t - eta > LOCK_STALE]:
            del self.locks[vid]  # ambulance is stuck far behind; the next refresh re-arms it
        for l, items in self.platoons.items():
            if items:
                self.platoons[l] = [(eta, pcu) for eta, pcu in items if eta >= t - 2]
        for l in [l for l, exp in self.warned.items() if exp < t]:
            del self.warned[l]
        for l in [l for l, (_, exp) in self.fill.items() if exp < t]:
            del self.fill[l]

    def downstream_block(self, link: str) -> float:
        """0 = room downstream, 1 = downstream full (from gossip / warnings)."""
        f = self.fill.get(link)
        if f is None:
            return 0.0
        return min(1.0, max(0.0, (f[0] - FILL_FREE) / (FILL_FULL - FILL_FREE)))

    def platoon_soon(self, link: str, t: int, horizon: float) -> float:
        return sum(pcu for eta, pcu in self.platoons.get(link, ()) if eta <= t + horizon)

    # ---- decision ----------------------------------------------------------
    def decide(self, t: int, obs: Observation, blocked_share: list[float]) -> int:
        cur = obs.current
        # 1. Ambulance corridor lock: green for the approach it will arrive on.
        due = [(eta, self.in_links.index(link)) for link, eta in self.locks.values()
               if eta - t <= PREEMPT_LEAD]
        if due:
            target = min(due)[1]
            if target == cur:
                return cur
            return target if obs.elapsed >= EMERGENCY_MIN_GREEN else cur

        # 2. Effective demand: discount traffic that can't move (downstream full),
        #    add platoons that neighbours say are about to arrive.
        soon = [self.platoon_soon(l, t, PLATOON_HORIZON) for l in self.in_links]
        eff = [q * (1 - 0.9 * s) + PLATOON_WEIGHT * p
               for q, s, p in zip(obs.queues_pcu, blocked_share, soon)]
        counts = [c + (1 if p > 0 else 0) for c, p in zip(obs.counts, soon)]
        obs2 = Observation(obs.t, eff, counts, obs.oldest_wait, obs.emergency,
                           cur, obs.elapsed, obs.red_for)

        if obs.elapsed >= self.base.min_green:
            # 3. Our released traffic has nowhere to go: hand the green over.
            if blocked_share[cur] > 0.6 and self.base.best_other(obs2) is not None:
                return self.base.best_other(obs2)
            # 4. Hold the green for a platoon that is about to arrive (green wave),
            #    but only if it outweighs what is waiting on the other approaches.
            arriving = self.platoon_soon(self.in_links[cur], t, HOLD_HORIZON)
            others = max((eff[i] for i in range(len(eff)) if i != cur), default=0.0)
            if (obs.elapsed < self.base.max_green and blocked_share[cur] < 0.5
                    and arriving >= 2 and arriving >= HOLD_RATIO * others
                    and not any(obs.emergency)
                    and max(obs.red_for or [0]) < self.base.max_red):
                return cur
        return self.base.decide(obs2)
