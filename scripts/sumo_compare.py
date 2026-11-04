"""Validate the controllers in SUMO (microscopic traffic simulator).

Builds a 4-way junction (2 lanes per approach), sends Indian-style mixed traffic,
and drives the signal through TraCI with the same controller classes the
dashboard uses. Results go to data/sumo_results.csv.

Usage:  python scripts/sumo_compare.py [--duration 3600] [--gui]
Needs:  pip install eclipse-sumo traci
"""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sumo  # noqa: E402  (eclipse-sumo package: provides the binaries)
import traci  # noqa: E402

from traffic import config as C  # noqa: E402
from traffic.optimizer import (AdaptiveController, FixedTimeController,  # noqa: E402
                               Observation, webster_plan)

OUT = ROOT / "sumo"
BIN = Path(sumo.SUMO_HOME) / "bin"
os.environ.setdefault("SUMO_HOME", str(sumo.SUMO_HOME))
APPROACH_EDGES = ["N2C", "E2C", "S2C", "W2C"]       # index matches C.APPROACHES
DEMAND_VPH = [700, 350, 600, 250]
VTYPES = {  # id: (vClass, length m, maxSpeed m/s, share, pcu)
    "motorcycle": ("motorcycle", 2.0, 16, 0.50, 0.5),
    "car": ("passenger", 4.5, 14, 0.35, 1.0),
    "bus": ("bus", 11.0, 12, 0.08, 3.0),
    "truck": ("truck", 9.0, 11, 0.07, 3.0),
}
PCU_BY_TYPE = {k: v[4] for k, v in VTYPES.items()} | {"ambulance": 1.0}


def build_network():
    OUT.mkdir(exist_ok=True)
    (OUT / "nodes.nod.xml").write_text("""<nodes>
  <node id="C" x="0" y="0" type="traffic_light"/>
  <node id="N" x="0" y="300"/><node id="E" x="300" y="0"/>
  <node id="S" x="0" y="-300"/><node id="W" x="-300" y="0"/>
</nodes>""")
    edges = []
    for d in "NESW":
        edges.append(f'<edge id="{d}2C" from="{d}" to="C" numLanes="2" speed="13.9"/>')
        edges.append(f'<edge id="C2{d}" from="C" to="{d}" numLanes="2" speed="13.9"/>')
    (OUT / "edges.edg.xml").write_text("<edges>\n  " + "\n  ".join(edges) + "\n</edges>")
    subprocess.run([str(BIN / "netconvert"), "-n", str(OUT / "nodes.nod.xml"),
                    "-e", str(OUT / "edges.edg.xml"), "-o", str(OUT / "junction.net.xml"),
                    "--no-turnarounds", "true", "--no-warnings", "true",
                    "--lefthand", "true"], check=True)  # India drives on the left


def build_routes(duration: int, seed: int):
    lines = ['<routes>', '  <vTypeDistribution id="mix">']
    for vid, (vclass, length, vmax, share, _) in VTYPES.items():
        lines.append(f'    <vType id="{vid}" vClass="{vclass}" length="{length}" '
                     f'maxSpeed="{vmax}" probability="{share}" sigma="0.5"/>')
    lines.append('  </vTypeDistribution>')
    lines.append('  <vType id="ambulance" vClass="emergency" length="5" maxSpeed="20" '
                 'guiShape="emergency" color="1,0,0"/>')
    dirs = "NESW"
    turn = {0: 0.2, 1: 0.6, 2: 0.2}  # right, straight, left (as offsets 1,2,3)
    for i, d in enumerate(dirs):
        for off, share in zip((1, 2, 3), turn.values()):
            to = dirs[(i + off) % 4]
            lines.append(f'  <flow id="f_{d}{to}" type="mix" begin="0" end="{duration}" '
                         f'from="{d}2C" to="C2{to}" vehsPerHour="{DEMAND_VPH[i] * share:.0f}" '
                         f'departLane="best" departSpeed="max"/>')
    for k, t in enumerate(range(400, duration - 100, 700)):
        d = dirs[(k + seed) % 4]
        to = dirs[(k + seed + 2) % 4]
        lines.append(f'  <trip id="amb{k}" type="ambulance" depart="{t}" '
                     f'from="{d}2C" to="C2{to}"/>')
    lines.append('</routes>')
    (OUT / "routes.rou.xml").write_text("\n".join(lines))


def link_groups(tls):
    """For each approach, which signal link indices it controls."""
    groups = [[] for _ in APPROACH_EDGES]
    for idx, links in enumerate(traci.trafficlight.getControlledLinks(tls)):
        for in_lane, _, _ in links:
            edge = in_lane.rsplit("_", 1)[0]
            if edge in APPROACH_EDGES:
                groups[APPROACH_EDGES.index(edge)].append(idx)
    return groups


def state_for(groups, n_links, green=None, amber=None):
    s = ["r"] * n_links
    for a, ch in ((green, "G"), (amber, "y")):
        if a is not None:
            for i in groups[a]:
                s[i] = ch
    return "".join(s)


def observe(t, current, elapsed, last_green):
    q, counts, oldest, emerg = [], [], [], []
    for e in APPROACH_EDGES:
        pcu, n, w, amb = 0.0, 0, 0.0, False
        for vid in traci.edge.getLastStepVehicleIDs(e):
            vt = traci.vehicle.getTypeID(vid).split("@")[0]
            if vt == "ambulance":
                amb = True
            if traci.vehicle.getSpeed(vid) < 1.0:
                pcu += PCU_BY_TYPE.get(vt, 1.0)
                n += 1
                w = max(w, traci.vehicle.getWaitingTime(vid))
        q.append(pcu)
        counts.append(n)
        oldest.append(w)
        emerg.append(amb)
    red_for = [0.0 if i == current else t - last_green[i] for i in range(len(APPROACH_EDGES))]
    return Observation(t, q, counts, oldest, emerg, current, elapsed, red_for)


def simulate(controller, duration, seed, gui=False):
    trip = OUT / f"tripinfo_{controller.name.replace(' ', '_')}.xml"
    exe = BIN / ("sumo-gui" if gui else "sumo")
    traci.start([str(exe), "-n", str(OUT / "junction.net.xml"), "-r", str(OUT / "routes.rou.xml"),
                 "--seed", str(seed), "--tripinfo-output", str(trip), "--no-step-log", "true",
                 "--no-warnings", "true", "--time-to-teleport", "-1", "--end", str(duration + 900)])
    tls = traci.trafficlight.getIDList()[0]
    groups = link_groups(tls)
    n_links = len(traci.trafficlight.getControlledLinks(tls))
    current, elapsed, transition, pending = 0, 0, 0, 0
    last_green = [0.0] * len(APPROACH_EDGES)
    traci.trafficlight.setRedYellowGreenState(tls, state_for(groups, n_links, green=0))
    t = 0
    while t < duration or traci.simulation.getMinExpectedNumber() > 0:
        if transition > 0:
            transition -= 1
            if transition == 2:
                traci.trafficlight.setRedYellowGreenState(tls, state_for(groups, n_links))
            if transition == 0:
                current, elapsed = pending, 0
                traci.trafficlight.setRedYellowGreenState(
                    tls, state_for(groups, n_links, green=current))
        else:
            want = controller.decide(observe(t, current, elapsed, last_green))
            if want != current:
                pending, transition = want, C.INTERGREEN
                traci.trafficlight.setRedYellowGreenState(
                    tls, state_for(groups, n_links, amber=current))
            else:
                elapsed += 1
                last_green[current] = t + 1
        traci.simulationStep()
        t += 1
        if t > duration + 900:
            break
    traci.close()

    rows = []
    for ti in ET.parse(trip).getroot().iter("tripinfo"):
        rows.append({"type": ti.get("vType").split("@")[0],
                     "wait": float(ti.get("waitingTime")),
                     "time_loss": float(ti.get("timeLoss"))})
    df = pd.DataFrame(rows)
    amb = df[df.type == "ambulance"]
    return {"controller": controller.name, "vehicles": len(df),
            "avg_wait_s": round(df.wait.mean(), 1),
            "avg_time_loss_s": round(df.time_loss.mean(), 1),
            "p95_wait_s": round(df.wait.quantile(0.95), 1),
            "ambulance_avg_wait_s": round(amb.wait.mean(), 1) if len(amb) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gui", action="store_true")
    args = p.parse_args()

    build_network()
    build_routes(args.duration, args.seed)
    avg_pcu = sum(v[3] * v[4] for v in VTYPES.values())
    _, greens = webster_plan([v / 3600 * avg_pcu for v in DEMAND_VPH])
    web = FixedTimeController(greens)
    web.name = "Webster pre-timed"
    results = [simulate(c, args.duration, args.seed, args.gui)
               for c in (FixedTimeController([30] * 4), web, AdaptiveController())]
    df = pd.DataFrame(results)
    df.to_csv(ROOT / "data" / "sumo_results.csv", index=False)
    print(df.to_string(index=False))
    base, ai = df.avg_wait_s.iloc[0], df.avg_wait_s.iloc[-1]
    print(f"\nSUMO: AI adaptive cuts average waiting time by {100 * (base - ai) / base:.0f}% "
          f"vs a 30 s fixed timer.")


if __name__ == "__main__":
    main()
