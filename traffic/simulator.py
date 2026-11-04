"""Lightweight 4-way junction simulator (1-second steps, queue model).

Runs in a fraction of a second, so the dashboard can compare controllers live.
Arrivals are generated up front from a seed, so every controller faces exactly
the same traffic. For a full microscopic check, see scripts/sumo_compare.py.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .optimizer import Observation


@dataclass
class Vehicle:
    arrival: float
    kind: str
    pcu: float
    approach: int
    emergency: bool = False
    departure: float | None = None


@dataclass
class SimConfig:
    duration: int = 3600
    demand_vph: tuple[float, ...] = (700, 350, 600, 250)  # vehicles/hour per approach
    peak: bool = True              # add a rush-hour bump in the middle of the run
    peak_strength: float = 0.35
    mix: dict = field(default_factory=lambda: dict(C.DEFAULT_MIX))
    emergencies_per_hour: float = 3.0
    saturation: float = C.SATURATION_PCU_PER_S
    intergreen: int = C.INTERGREEN
    startup_lost: float = C.STARTUP_LOST
    seed: int = 7


def generate_arrivals(cfg: SimConfig) -> list[Vehicle]:
    rng = np.random.default_rng(cfg.seed)
    kinds = list(cfg.mix)
    shares = np.array([cfg.mix[k] for k in kinds], dtype=float)
    shares /= shares.sum()
    vehicles: list[Vehicle] = []
    for a, vph in enumerate(cfg.demand_vph):
        for t in range(cfg.duration):
            mult = 1.0
            if cfg.peak:
                mult += cfg.peak_strength * math.exp(-((t - cfg.duration / 2) / (cfg.duration / 6)) ** 2)
            n = rng.poisson(vph / 3600 * mult)
            for _ in range(n):
                k = kinds[rng.choice(len(kinds), p=shares)]
                vehicles.append(Vehicle(t + rng.random(), k, C.PCU[k], a))
    n_amb = rng.poisson(cfg.emergencies_per_hour * cfg.duration / 3600)
    for _ in range(n_amb):
        vehicles.append(Vehicle(rng.uniform(60, cfg.duration - 60), "ambulance",
                                C.PCU["ambulance"], int(rng.integers(len(cfg.demand_vph))),
                                emergency=True))
    vehicles.sort(key=lambda v: v.arrival)
    return vehicles


@dataclass
class SimResult:
    controller: str
    vehicles: list[Vehicle]
    timeline: pd.DataFrame        # per-second queues and active phase
    duration: int

    def metrics(self) -> dict:
        waits, kinds, amb = [], [], []
        served = 0
        for v in self.vehicles:
            if v.departure is not None:
                served += 1
                w = v.departure - v.arrival
            else:  # still stuck in the queue at the end: count delay so far
                w = self.duration - v.arrival
            waits.append(w)
            kinds.append(v.kind)
            if v.emergency:
                amb.append(w)
        waits = np.array(waits)
        delay_h = {}
        for w, k in zip(waits, kinds):
            delay_h[k] = delay_h.get(k, 0.0) + w / 3600
        fuel = sum(h * C.IDLE_FUEL_L_PER_H.get(k, 0.5) for k, h in delay_h.items())
        q = self.timeline[[c for c in self.timeline if c.startswith("q_")]]
        return {
            "avg_wait_s": float(waits.mean()) if len(waits) else 0.0,
            "p95_wait_s": float(np.percentile(waits, 95)) if len(waits) else 0.0,
            "max_wait_s": float(waits.max()) if len(waits) else 0.0,
            "avg_queue_pcu": float(q.sum(axis=1).mean()),
            "max_queue_pcu": float(q.max().max()),
            "throughput": served,
            "vehicles": len(self.vehicles),
            "ambulance_avg_wait_s": float(np.mean(amb)) if amb else float("nan"),
            "phase_changes": int(((self.timeline["phase"] == -1)
                                  & (self.timeline["phase"].shift(1) != -1)).sum()),
            "idle_fuel_l": fuel,
            "co2_kg": fuel * C.CO2_KG_PER_L,
        }


def run(controller, cfg: SimConfig, arrivals: list[Vehicle] | None = None) -> SimResult:
    """Simulate one controller. Pass the same `arrivals` to compare controllers fairly."""
    src = arrivals if arrivals is not None else generate_arrivals(cfg)
    vehicles = [Vehicle(v.arrival, v.kind, v.pcu, v.approach, v.emergency) for v in src]
    n = len(cfg.demand_vph)
    queues: list[deque[Vehicle]] = [deque() for _ in range(n)]
    nxt = 0

    current, elapsed, credit = 0, 0, 0.0
    last_green = [0.0] * n
    intergreen_left, pending = 0, 0
    rows = []

    for t in range(cfg.duration):
        while nxt < len(vehicles) and vehicles[nxt].arrival < t + 1:
            queues[vehicles[nxt].approach].append(vehicles[nxt])
            nxt += 1

        if intergreen_left > 0:
            intergreen_left -= 1
            if intergreen_left == 0:
                current, elapsed, credit = pending, 0, 0.0
            phase = -1
        else:
            obs = Observation(
                t=t,
                queues_pcu=[sum(v.pcu for v in q) for q in queues],
                counts=[len(q) for q in queues],
                oldest_wait=[(t - q[0].arrival) if q else 0.0 for q in queues],
                emergency=[any(v.emergency for v in q) for q in queues],
                current=current,
                elapsed=elapsed,
                red_for=[0.0 if i == current else t - last_green[i] for i in range(n)],
            )
            want = controller.decide(obs)
            if want != current:
                intergreen_left, pending = cfg.intergreen, want
                phase = -1
            else:
                phase = current
                last_green[current] = t + 1
                if elapsed >= cfg.startup_lost:
                    credit += cfg.saturation
                    q = queues[current]
                    while q and q[0].pcu <= credit + 1e-9:
                        v = q.popleft()
                        v.departure = t + 1
                        credit -= v.pcu
                    if not q:
                        credit = min(credit, cfg.saturation)  # unused green can't be banked
                elapsed += 1

        row = {"t": t, "phase": phase}
        for i, q in enumerate(queues):
            row[f"q_{C.APPROACHES[i]}"] = sum(v.pcu for v in q)
        rows.append(row)

    return SimResult(controller.name, vehicles, pd.DataFrame(rows), cfg.duration)
