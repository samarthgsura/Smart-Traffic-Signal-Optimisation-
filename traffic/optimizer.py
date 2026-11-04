"""Signal control policies.

Every controller sees the same observation each second and returns the phase
(approach index) that should be green. The simulator (or a real junction)
handles amber / all-red whenever the requested phase changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C


@dataclass
class Observation:
    t: int                       # simulation clock (s)
    queues_pcu: list[float]      # queued demand per approach, in PCU
    counts: list[int]            # queued vehicles per approach
    oldest_wait: list[float]     # how long the first queued vehicle has waited (s)
    emergency: list[bool]        # ambulance waiting on this approach?
    current: int                 # approach that is green now
    elapsed: int                 # seconds since this green started
    red_for: list[float] | None = None  # seconds since each approach last had green


class FixedTimeController:
    """Classic fixed timer: every approach gets the same green, in rotation."""

    name = "Fixed timer"

    def __init__(self, greens: list[int]):
        self.greens = greens

    def decide(self, obs: Observation) -> int:
        if obs.elapsed >= self.greens[obs.current]:
            return (obs.current + 1) % len(self.greens)
        return obs.current


@dataclass
class AdaptiveController:
    """Queue-aware actuated controller with safety constraints.

    Rules, in priority order:
      1. Never end a green before MIN_GREEN (pedestrian / safety minimum).
      2. Ambulance on a red approach: switch to it as soon as allowed.
      3. Any approach waiting more than MAX_RED: serve it (no starvation).
      4. MAX_GREEN reached: hand over to the approach under most pressure.
      5. Gap-out: current queue has cleared -> hand over to a waiting approach.
      6. Another queue is much bigger than ours -> hand over early.
    "Pressure" = queued PCU plus an ageing term, so long waits also count.
    """

    min_green: int = C.MIN_GREEN
    max_green: int = C.MAX_GREEN
    max_red: int = C.MAX_RED
    gap_pcu: float = 1.0          # queue below this counts as "cleared"
    imbalance_ratio: float = 3.0  # switch early if another queue is this many times bigger
    ageing_s: float = 15.0        # 15 s of waiting counts like 1 extra PCU
    name: str = field(default="AI adaptive", init=False)

    def pressure(self, obs: Observation, i: int) -> float:
        return obs.queues_pcu[i] + obs.oldest_wait[i] / self.ageing_s

    def best_other(self, obs: Observation) -> int | None:
        others = [i for i in range(len(obs.queues_pcu))
                  if i != obs.current and obs.counts[i] > 0]
        if not others:
            return None
        return max(others, key=lambda i: self.pressure(obs, i))

    def decide(self, obs: Observation) -> int:
        cur = obs.current
        if obs.elapsed < self.min_green:
            return cur

        # 2. Emergency vehicle priority.
        if obs.emergency[cur] and obs.elapsed < self.max_green + 30:
            return cur  # let the ambulance through first
        waiting_amb = [i for i, e in enumerate(obs.emergency) if e and i != cur]
        if waiting_amb:
            return max(waiting_amb, key=lambda i: obs.oldest_wait[i])

        best = self.best_other(obs)
        if best is None:
            return cur  # nobody else is waiting: keep the green

        # 3. Starvation guard: based on how long the approach has been on red
        # (not on the oldest vehicle's wait, which is always long in heavy traffic).
        red = obs.red_for if obs.red_for is not None else obs.oldest_wait
        starving = [i for i in range(len(obs.counts))
                    if i != cur and obs.counts[i] > 0 and red[i] >= self.max_red]
        if starving:
            return max(starving, key=lambda i: red[i])

        # 4. Max green.
        if obs.elapsed >= self.max_green:
            return best
        # 5. Gap-out.
        if obs.queues_pcu[cur] < self.gap_pcu:
            return best
        # 6. Strong imbalance.
        if obs.queues_pcu[best] > self.imbalance_ratio * obs.queues_pcu[cur] + 5:
            return best
        return cur


def webster_plan(flows_pcu_s: list[float],
                 saturation: float = C.SATURATION_PCU_PER_S,
                 intergreen: int = C.INTERGREEN,
                 startup_lost: float = C.STARTUP_LOST,
                 min_green: int = C.MIN_GREEN,
                 max_green: int = C.MAX_GREEN) -> tuple[int, list[int]]:
    """Webster's optimal cycle length and green split (the textbook pre-timed plan).

    Returns (cycle_seconds, green_seconds_per_approach).
    """
    n = len(flows_pcu_s)
    lost = n * (intergreen + startup_lost)
    ratios = [max(f, 1e-6) / saturation for f in flows_pcu_s]
    y = min(sum(ratios), 0.9)  # cap so the formula stays finite when oversaturated
    cycle = (1.5 * lost + 5) / (1 - y)
    cycle = max(40.0, min(cycle, 180.0))
    effective = cycle - lost
    greens = [int(round(min(max_green, max(min_green, effective * r / sum(ratios)))))
              for r in ratios]
    return int(round(sum(greens) + n * intergreen)), greens


def recommend_greens(queue_pcu: list[float],
                     saturation: float = C.SATURATION_PCU_PER_S,
                     min_green: int = C.MIN_GREEN,
                     max_green: int = C.MAX_GREEN) -> list[int]:
    """Green time each approach needs to clear what is queued right now (+10% margin)."""
    return [int(min(max_green, max(min_green, C.STARTUP_LOST + 1.1 * q / saturation)))
            for q in queue_pcu]
