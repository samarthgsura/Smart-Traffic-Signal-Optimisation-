"""Benchmark signal control modes.

City mode (default): Fixed timer vs Isolated AI vs P2P Mesh AI on the Bengaluru
and Mumbai networks, in three scenarios (normal day, 6 PM rush, a mesh node
failure), each with ambulances. Seeds 10+ are used so the results are not the
seeds the mesh parameters were tuned on (0-2).

    python scripts/compare.py                 # city benchmark -> data/city_results.csv
    python scripts/compare.py --seeds 3       # quicker
    python scripts/compare.py --junction      # original single-junction benchmark
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic import config as C  # noqa: E402
from traffic.city_graph import available, load  # noqa: E402
from traffic.city_sim import MODE_LABELS, MODES, CitySim, rank_for  # noqa: E402
from traffic.optimizer import AdaptiveController, FixedTimeController, webster_plan  # noqa: E402
from traffic.simulator import SimConfig, generate_arrivals, run  # noqa: E402

DURATION = 2400
FAIL_NODE = {"bengaluru": "SLK", "mumbai": "KLN", "bengaluru_osm": "Silk Board"}  # busy junction that dies


def fail_node_id(city, graph):
    key = FAIL_NODE[city]
    if key in graph.nodes:
        return key
    return next(n.id for n in graph.nodes.values() if n.name == key and n.kind == "signal")


def scenario_events(city, scenario, graph):
    ev = []
    for k in range(8):  # one ambulance every ~4 minutes on a fixed set of routes
        o, d = graph.ambulance_routes[k % len(graph.ambulance_routes)]
        ev.append((200 + 250 * k, lambda s, o=o, d=d: s.spawn_ambulance(o, d)))
    if scenario == "rush":
        ev.append((600, lambda s: s.trigger_rush(1200, 1.8)))
    if scenario == "node_failure":
        dead = fail_node_id(city, graph)
        ev.append((600, lambda s: s.kill(dead) if s.mode == "p2p" else None))
    return ev


def city_benchmark(seeds: int, real_seeds: int, cities: list[str]):
    rows = []
    for city in cities:
        g = load(city)
        n_seeds = real_seeds if city.endswith("_osm") else seeds
        for scenario in ("normal", "rush", "node_failure"):
            for seed in range(10, 10 + n_seeds):
                for mode in MODES:
                    sim = CitySim(load(city), mode, seed=seed)
                    m = sim.run(DURATION, scenario_events(city, scenario, g))
                    rows.append({"city": g.name, "scenario": scenario, "seed": seed,
                                 "mode": MODE_LABELS[mode], **{k: v for k, v in m.items()
                                                               if k not in ("packets", "rank")},
                                 "packets": sum(m["packets"].values())})
            print(f"  done: {g.name} / {scenario}")
    df = pd.DataFrame(rows)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "city_results.csv", index=False)

    cols = ["avg_delay_s", "p95_delay_s", "throughput_pcu_h", "ambulance_delay_s",
            "jam_ratio", "score"]
    summary = (df.groupby(["city", "scenario", "mode"], sort=False)[cols].mean()
               .round({"avg_delay_s": 1, "p95_delay_s": 0, "throughput_pcu_h": 0,
                       "ambulance_delay_s": 1, "jam_ratio": 3, "score": 0}))
    summary["rank"] = summary["score"].astype(int).map(rank_for)
    pd.set_option("display.width", 200)
    print(summary.to_string())
    print()
    for (city, scenario), g in df.groupby(["city", "scenario"], sort=False):
        d = g.groupby("mode")["avg_delay_s"].mean()
        a = g.groupby("mode")["ambulance_delay_s"].mean()
        fx, iso, p2p = (d[MODE_LABELS[m]] for m in MODES)
        print(f"{city:9} {scenario:12} P2P vs fixed {100 * (p2p - fx) / fx:+5.0f}% delay | "
              f"P2P vs isolated {100 * (p2p - iso) / iso:+5.0f}% delay | ambulance delay "
              f"{a[MODE_LABELS['fixed']]:.0f}s -> {a[MODE_LABELS['isolated']]:.0f}s -> "
              f"{a[MODE_LABELS['p2p']]:.0f}s")


def junction_benchmark(seeds: int, duration: int):
    rows = []
    for seed in range(seeds):
        cfg = SimConfig(duration=duration, seed=seed)
        arrivals = generate_arrivals(cfg)
        avg_pcu = sum(cfg.mix[k] * C.PCU[k] for k in cfg.mix) / sum(cfg.mix.values())
        _, greens = webster_plan([v / 3600 * avg_pcu for v in cfg.demand_vph],
                                 cfg.saturation, cfg.intergreen, cfg.startup_lost)
        web = FixedTimeController(greens)
        web.name = "Webster pre-timed"
        for ctrl in (FixedTimeController([30] * 4), web, AdaptiveController()):
            rows.append({"seed": seed, "controller": ctrl.name,
                         **run(ctrl, cfg, arrivals).metrics()})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "sim_results.csv", index=False)
    cols = ["avg_wait_s", "p95_wait_s", "avg_queue_pcu", "throughput",
            "ambulance_avg_wait_s", "idle_fuel_l", "co2_kg"]
    summary = df.groupby("controller", sort=False)[cols].mean().round(1)
    print(summary.to_string())
    base, ai = summary.loc["Fixed timer", "avg_wait_s"], summary.loc["AI adaptive", "avg_wait_s"]
    print(f"\nAI adaptive cuts average waiting time by {100 * (base - ai) / base:.0f}% "
          f"vs a 30 s fixed timer ({seeds} seeds, {duration} s each).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--junction", action="store_true", help="single-junction benchmark")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--real-seeds", type=int, default=2, help="seeds for the real OSM map (slower)")
    p.add_argument("--cities", default=",".join(available()))
    p.add_argument("--duration", type=int, default=3600, help="junction mode only")
    args = p.parse_args()
    if args.junction:
        junction_benchmark(args.seeds, args.duration)
    else:
        cities = args.cities.split(",")
        print(f"City benchmark: {', '.join(cities)} x 3 scenarios x 3 modes, {DURATION // 60} min "
              f"each ({args.seeds} seeds; {args.real_seeds} for the real map)")
        city_benchmark(args.seeds, args.real_seeds, cities)


if __name__ == "__main__":
    main()
