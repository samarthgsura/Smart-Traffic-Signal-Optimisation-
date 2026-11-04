"""Smart Traffic Signal Optimisation - Streamlit dashboard (Neural Storm 2026, FT-01).

Run:  streamlit run app.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from traffic import config as C  # noqa: E402
from traffic.optimizer import (AdaptiveController, FixedTimeController,  # noqa: E402
                               recommend_greens, webster_plan)
from traffic.simulator import SimConfig, generate_arrivals, run  # noqa: E402

COLOURS = {"Fixed timer": "#9aa0a6", "Webster pre-timed": "#f5a623", "AI adaptive": "#1a9e5c"}

st.set_page_config(page_title="Smart Traffic Signal AI", page_icon="🚦", layout="wide")
st.title("🚦 Smart Traffic Signal Optimisation")
st.caption("Neural Storm 2026 · FT-01 · Camera-based vehicle counting + adaptive signal "
           "control, from one junction up to a city-wide peer-to-peer signal mesh")

tab_city, tab_clips, tab_live, tab_sim, tab_edge, tab_how = st.tabs(
    ["🏙️ City Command Center", "🛺 Bengaluru clips", "🎥 Live junction",
     "📊 Single junction: adaptive vs fixed", "⚡ Edge performance", "ℹ️ How it works"])


# --------------------------------------------------------------------------- city command center
CITY_SHORT = {"bengaluru_osm": "Bengaluru · real map", "bengaluru": "Bengaluru · schematic",
              "mumbai": "Mumbai · schematic"}


def _source_stamp() -> tuple:
    """Changes whenever the map data or the page template changes (cache key)."""
    files = [ROOT / "traffic" / "web" / "command_center.html", DATA / "cities" / "bengaluru_osm.json",
             DATA / "cities" / "bengaluru_scenery.json", ROOT / "traffic" / "web" / "vendor" / "three.min.js",
             ROOT / "traffic" / "city_graph.py", ROOT / "traffic" / "p2p_mesh.py"]
    return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)


@st.cache_data(show_spinner="Loading the city road network…")
def command_center_html(_stamp: tuple) -> str:
    from traffic import city_graph, p2p_mesh
    from traffic.city_sim import JAM_LEVEL

    ctrl = AdaptiveController()
    constants = {
        "MIN_GREEN": ctrl.min_green, "MAX_GREEN": ctrl.max_green, "MAX_RED": ctrl.max_red,
        "GAP_PCU": ctrl.gap_pcu, "IMBALANCE_RATIO": ctrl.imbalance_ratio, "AGEING_S": ctrl.ageing_s,
        "INTERGREEN": C.INTERGREEN, "STARTUP_LOST": C.STARTUP_LOST,
        "BACKPRESSURE_LEVEL": p2p_mesh.BACKPRESSURE_LEVEL, "WARN_TTL": p2p_mesh.WARN_TTL,
        "STATE_TTL": p2p_mesh.STATE_TTL, "FILL_FREE": p2p_mesh.FILL_FREE,
        "FILL_FULL": p2p_mesh.FILL_FULL, "PLATOON_HORIZON": p2p_mesh.PLATOON_HORIZON,
        "HOLD_HORIZON": p2p_mesh.HOLD_HORIZON, "HOLD_RATIO": p2p_mesh.HOLD_RATIO,
        "PLATOON_WEIGHT": p2p_mesh.PLATOON_WEIGHT, "PREEMPT_LEAD": p2p_mesh.PREEMPT_LEAD,
        "EMERGENCY_MIN_GREEN": p2p_mesh.EMERGENCY_MIN_GREEN, "JAM_LEVEL": JAM_LEVEL,
        "CORRIDOR_HOPS": p2p_mesh.CORRIDOR_HOPS, "LOCK_STALE": p2p_mesh.LOCK_STALE,
        "M_PER_UNIT": city_graph.M_PER_UNIT, "RUSH_MULT": 1.8, "RUSH_DURATION": 1200,
    }
    edge = None
    bench_file = DATA / "edge_benchmark.json"
    if bench_file.exists():
        res = {r["imgsz"]: r for r in json.loads(bench_file.read_text())["results"]}
        best = res.get(640) or next(iter(res.values()))
        fast = res.get(320)
        edge = {"fps": round(best["onnx_fps"]),
                "detail": f"640 px · {round(fast['onnx_fps'])} FPS at 320 px" if fast else "CPU only"}
    order = [k for k in CITY_SHORT if k in city_graph.available()]
    scenery_file = DATA / "cities" / "bengaluru_scenery.json"   # lakes + parks for the 3D view
    scenery = json.loads(scenery_file.read_text(encoding="utf-8")) if scenery_file.exists() else None
    cities = {}
    for k in order:
        cities[k] = city_graph.load(k).to_json()
        cities[k]["short"] = CITY_SHORT[k]
    payload = {
        "constants": constants,
        "pcu": C.PCU,
        "mix": [[k, v] for k, v in C.DEFAULT_MIX.items()],
        "cities": cities,
        "cityOrder": order,
        "defaultCity": order[0],
        "edge": edge,
        "scenery": scenery,
    }
    web = ROOT / "traffic" / "web"
    html = (web / "command_center.html").read_text(encoding="utf-8")
    three = web / "vendor" / "three.min.js"  # bundled so the 3D view also works offline
    if three.exists():
        html = html.replace("<!--THREE_JS-->", "<script>" + three.read_text(encoding="utf-8") + "</script>")
    return html.replace("/*__DATA__*/null", json.dumps(payload).replace("</", "<\\/"))


with tab_city:
    st.markdown("Every signal is a **peer** that talks only to its neighbouring junctions. "
                "All three control modes run side by side on identical traffic, so switching "
                "mode shows the same city under a different brain. **Drag to pan, scroll to "
                "zoom** (vehicles appear when you zoom in), click any signal to inspect it, and "
                "use **God mode** to cause trouble.")
    st.iframe(command_center_html(_source_stamp()), height=920)  # our own generated HTML
    city_csv = DATA / "city_results.csv"
    if city_csv.exists():
        st.subheader("Offline benchmark (`python scripts/compare.py`)")
        cr = pd.read_csv(city_csv)
        summ = (cr.groupby(["city", "scenario", "mode"], sort=False)
                [["avg_delay_s", "p95_delay_s", "throughput_pcu_h", "ambulance_delay_s", "score"]]
                .mean().round(0).reset_index())
        summ.columns = ["City", "Scenario", "Mode", "Avg delay (s)", "95th pct delay (s)",
                        "Throughput (PCU/h)", "Ambulance delay (s)", "Score (XP)"]
        st.dataframe(summ, hide_index=True, width="stretch")
        seeds = ", ".join(f"{c}: {n}" for c, n in cr.groupby("city", sort=False)["seed"].nunique().items())
        st.caption(f"Mean of unseen random seeds per scenario ({seeds}), 40 simulated minutes each, "
                   "with 8 ambulances per run. Python engine (traffic/city_sim.py); the live map runs "
                   "a JavaScript port of the same rules.")


# --------------------------------------------------------------------------- live video
@st.cache_resource
def get_detector(model_path: str, zones_path: str, imgsz: int, conf: float):
    from traffic.detector import TrafficDetector, load_zones
    return TrafficDetector(load_zones(zones_path), model_path=model_path, imgsz=imgsz, conf=conf)


with tab_live:
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Input")
        demo = DATA / "vehicles.mp4"
        options = (["Demo video"] if demo.exists() else []) + ["Upload a video"]
        source = st.radio("Video source", options, horizontal=True)
        video_path = None
        if source == "Upload a video":
            up = st.file_uploader("Traffic video (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"])
            if up:
                tmp = Path(tempfile.gettempdir()) / f"upload_{up.name}"
                tmp.write_bytes(up.getbuffer())
                video_path = str(tmp)
        else:
            video_path = str(demo)

        zones_file = st.text_input("Approach zones (JSON)", str(DATA / "zones.json"))
        c1, c2 = st.columns(2)
        n_frames = c1.slider("Frames to process", 30, 600, 180, 30)
        stride = c2.slider("Process every Nth frame", 1, 5, 2)
        c3, c4 = st.columns(2)
        imgsz = c3.select_slider("Model input size", [320, 416, 640], 640)
        conf = c4.slider("Confidence", 0.1, 0.8, 0.3, 0.05)
        onnx = DATA / f"yolov8n_{imgsz}.onnx"
        use_onnx = st.checkbox(f"Use ONNX edge model ({onnx.name})", value=onnx.exists(),
                               disabled=not onnx.exists())
        start = st.button("▶ Run detection", type="primary", disabled=video_path is None)
        stats_box = st.empty()
        rec_box = st.empty()

    with left:
        frame_box = st.empty()
        counts_box = st.empty()
        if not start:
            frame_box.info("Pick a video and press **Run detection**. Vehicles are detected "
                           "with YOLOv8, tracked with ByteTrack, and counted inside each "
                           "approach zone.")

    if start and video_path:
        model_path = str(onnx if use_onnx else DATA / "yolov8n.pt")
        if not Path(model_path).exists():
            model_path = "yolov8n.pt"
        det = get_detector(model_path, zones_file, imgsz, conf)
        det.reset()
        cap = cv2.VideoCapture(video_path)
        lat, i, res = [], 0, None
        t_start = time.perf_counter()
        while i < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride:
                continue
            res = det.process(frame)
            lat.append(res.latency_ms)
            shown = res.annotated
            if shown.shape[1] > 1280:  # 4K frames are slow to send to the browser
                shown = cv2.resize(shown, (1280, int(shown.shape[0] * 1280 / shown.shape[1])))
            frame_box.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), channels="RGB",
                            width="stretch")
            names = list(res.pcu)
            table = pd.DataFrame(
                [{"Approach": z,
                  **{k: res.counts[z].get(k, 0) for k in ("motorcycle", "car", "bus", "truck",
                                                           "bicycle")},
                  "Queue (PCU)": round(res.pcu[z], 1),
                  "Unique vehicles seen": res.unique_seen[z]} for z in names])
            counts_box.dataframe(table, hide_index=True, width="stretch")
            greens = recommend_greens([res.pcu[z] for z in names])
            fig = go.Figure(go.Bar(x=names, y=greens, text=[f"{g}s" for g in greens],
                                   marker_color="#1a9e5c"))
            fig.update_layout(title="Recommended green time (to clear current queue)",
                              yaxis_title="seconds", height=280,
                              margin=dict(l=10, r=10, t=40, b=10))
            rec_box.plotly_chart(fig, width="stretch", key=f"rec_{i}")
            stats_box.markdown(
                f"**Inference:** {np.median(lat):.0f} ms/frame (median) · "
                f"**Frames:** {len(lat)} · "
                f"**Night mode:** {'on' if res.enhanced else 'off'}")
        cap.release()
        if res is not None:
            names = list(res.pcu)
            order = sorted(names, key=lambda z: -res.pcu[z])
            st.success(f"Processed {len(lat)} frames in {time.perf_counter() - t_start:.1f}s. "
                       f"Next green should go to **{order[0]}** "
                       f"({res.pcu[order[0]]:.1f} PCU queued).")


# --------------------------------------------------------------------------- simulation
@st.cache_data(show_spinner="Simulating the junction…")
def simulate(demand, duration_min, peak, amb_per_h, seed):
    cfg = SimConfig(duration=duration_min * 60, demand_vph=tuple(demand), peak=peak,
                    emergencies_per_hour=amb_per_h, seed=seed)
    arrivals = generate_arrivals(cfg)
    avg_pcu = sum(cfg.mix[k] * C.PCU[k] for k in cfg.mix)
    _, wg = webster_plan([v / 3600 * avg_pcu for v in demand])
    web = FixedTimeController(wg)
    web.name = "Webster pre-timed"
    out = {}
    for ctrl in (FixedTimeController([30] * 4), web, AdaptiveController()):
        r = run(ctrl, cfg, arrivals)
        per = pd.DataFrame([{"approach": C.APPROACHES[v.approach],
                             "wait": (v.departure if v.departure is not None else r.duration)
                             - v.arrival} for v in r.vehicles])
        out[ctrl.name] = {"metrics": r.metrics(), "timeline": r.timeline,
                          "per_approach": per.groupby("approach")["wait"].mean()}
    return out, wg


with tab_sim:
    st.markdown("Same random traffic, three signal policies. Adjust demand to see where "
                "adaptive control helps most.")
    with st.expander("Traffic scenario", expanded=True):
        cols = st.columns(4)
        defaults = SimConfig().demand_vph
        demand = [cols[i].slider(f"{a} (veh/h)", 0, 1200, int(defaults[i]), 50)
                  for i, a in enumerate(C.APPROACHES)]
        c1, c2, c3, c4 = st.columns(4)
        duration_min = c1.slider("Duration (min)", 15, 120, 60, 15)
        peak = c2.toggle("Rush-hour peak", True)
        amb = c3.slider("Ambulances per hour", 0, 10, 3)
        seed = c4.number_input("Random seed", 0, 999, 7)

    res, wgreens = simulate(demand, duration_min, peak, amb, int(seed))
    fixed, ai = res["Fixed timer"]["metrics"], res["AI adaptive"]["metrics"]

    def pct(a, b):
        return f"{100 * (a - b) / b:+.0f}%" if b else "–"  # negative = less waiting

    m = st.columns(5)
    m[0].metric("Avg wait · AI", f"{ai['avg_wait_s']:.0f} s",
                pct(ai["avg_wait_s"], fixed["avg_wait_s"]), delta_color="inverse")
    m[1].metric("95th pct wait · AI", f"{ai['p95_wait_s']:.0f} s",
                pct(ai["p95_wait_s"], fixed["p95_wait_s"]), delta_color="inverse")
    amb_ai, amb_fx = ai["ambulance_avg_wait_s"], fixed["ambulance_avg_wait_s"]
    m[2].metric("Ambulance wait · AI", "–" if np.isnan(amb_ai) else f"{amb_ai:.0f} s",
                None if np.isnan(amb_ai) else pct(amb_ai, amb_fx), delta_color="inverse")
    m[3].metric("Idle fuel saved", f"{fixed['idle_fuel_l'] - ai['idle_fuel_l']:.1f} L")
    m[4].metric("CO₂ avoided", f"{fixed['co2_kg'] - ai['co2_kg']:.1f} kg")
    st.caption("Deltas compare AI adaptive with a 30 s fixed timer. Fuel/CO₂ is a rough "
               "idling estimate for this one junction over the simulated period.")

    summary = pd.DataFrame({name: v["metrics"] for name, v in res.items()}).T
    show = summary[["avg_wait_s", "p95_wait_s", "avg_queue_pcu", "max_queue_pcu", "throughput",
                    "ambulance_avg_wait_s", "phase_changes", "idle_fuel_l", "co2_kg"]]
    show.columns = ["Avg wait (s)", "95th pct wait (s)", "Avg queue (PCU)", "Max queue (PCU)",
                    "Vehicles served", "Ambulance wait (s)", "Phase changes", "Idle fuel (L)",
                    "CO₂ (kg)"]
    st.dataframe(show.round(1), width="stretch")
    st.caption(f"Webster pre-timed greens (N/E/S/W): {wgreens} s — the best a fixed plan can "
               "do when it knows the average demand in advance.")

    g1, g2 = st.columns([3, 2])
    with g1:
        fig = go.Figure()
        for name, v in res.items():
            tl = v["timeline"]
            total = tl[[c for c in tl if c.startswith("q_")]].sum(axis=1).rolling(60).mean()
            fig.add_trace(go.Scatter(x=tl["t"] / 60, y=total, name=name,
                                     line=dict(color=COLOURS[name], width=2)))
        fig.update_layout(title="Vehicles queued at the junction (1-min average)",
                          xaxis_title="minutes", yaxis_title="PCU queued", height=360,
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with g2:
        per = pd.DataFrame({n: v["per_approach"] for n, v in res.items()}).reindex(C.APPROACHES)
        long = per.reset_index(names="approach").melt("approach", var_name="controller",
                                                      value_name="wait")
        fig = px.bar(long, x="approach", y="wait", color="controller", barmode="group",
                     color_discrete_map=COLOURS, title="Average wait by approach")
        fig.update_layout(yaxis_title="seconds", xaxis_title=None, height=360,
                          legend=dict(orientation="h", y=-0.2, title=None),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Signal timeline (first 10 minutes)")
    rows = []
    for name in ("Fixed timer", "AI adaptive"):
        tl = res[name]["timeline"].head(600)
        seg = (tl["phase"] != tl["phase"].shift()).cumsum()
        for _, g in tl.groupby(seg):
            p = int(g["phase"].iloc[0])
            rows.append({"controller": name, "start": g["t"].iloc[0],
                         "len": len(g), "phase": "amber / all-red" if p < 0 else C.APPROACHES[p]})
    tdf = pd.DataFrame(rows)
    pal = {"North": "#4287f5", "East": "#f5a623", "South": "#50c878", "West": "#dc50c8",
           "amber / all-red": "#d0d0d0"}
    fig = go.Figure()
    for ph, g in tdf.groupby("phase"):
        fig.add_trace(go.Bar(y=g["controller"], x=g["len"], base=g["start"], orientation="h",
                             name=ph, marker_color=pal[ph],
                             hovertemplate="%{base}s +%{x}s"))
    fig.update_layout(barmode="overlay", height=230, xaxis_title="seconds",
                      legend=dict(orientation="h", y=-0.4), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    sumo_csv = DATA / "sumo_results.csv"
    if sumo_csv.exists():
        st.subheader("Validated in SUMO (microscopic simulator)")
        st.dataframe(pd.read_csv(sumo_csv), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- edge
with tab_edge:
    bench = DATA / "edge_benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        st.markdown(f"YOLOv8n on **CPU only** ({b['cpu']}), median over {b['frames']} frames. "
                    "A junction controller needs about 1–2 decisions per second, so anything "
                    "above ~5 FPS per camera is enough.")
        df = pd.DataFrame(b["results"])
        long = df.melt("imgsz", ["pytorch_cpu_ms", "onnx_cpu_ms"], var_name="runtime",
                       value_name="ms")
        long["runtime"] = long["runtime"].map({"pytorch_cpu_ms": "PyTorch",
                                               "onnx_cpu_ms": "ONNX Runtime"})
        fig = px.bar(long, x=long["imgsz"].astype(str), y="ms", color="runtime",
                     barmode="group", color_discrete_sequence=["#9aa0a6", "#1a9e5c"],
                     title="Inference latency per frame (lower is better)")
        fig.update_layout(xaxis_title="model input size (px)", yaxis_title="ms", height=360)
        st.plotly_chart(fig, width="stretch")
        df.columns = ["Input size", "PyTorch CPU (ms)", "ONNX CPU (ms)", "ONNX FPS",
                      "Model size (MB)"]
        st.dataframe(df, hide_index=True, width="stretch")
    else:
        st.info("Run `python scripts/export_edge.py` to export the ONNX edge model and "
                "benchmark it on this machine.")


# --------------------------------------------------------------------------- bengaluru clips
with tab_clips:
    st.subheader("YOLOv8n vs UVH-26 on real Bengaluru footage")
    st.caption("Banashankari traffic circle, Aug 2024. UVH-26 is trained on Bengaluru CCTV and has "
               "a three-wheeler class. Pre-rendered at 10 frames per second, not a live feed.")
    clip_names = {"banashankari_02": "Clip 2: signal queue with many autos (29 s)",
                  "banashankari_01": "Clip 1: under the metro line (22 s)"}
    clip = st.selectbox("Clip", list(clip_names), format_func=clip_names.get)
    clip_video = DATA / "clip_videos" / f"{clip}.webm"
    if clip_video.exists():
        st.video(str(clip_video), loop=True, muted=True)
    else:
        st.info("Run `python scripts/download_bengaluru.py`, then `python scripts/render_clips.py`.")

    clip_results = DATA / "bengaluru_clips.json"
    if clip_results.exists():
        rows = [{"Detector": name, "Avg detected per frame": r["avg_visible"],
                 "Max in a frame": r["max_visible"], "Time per frame (ms, CPU)": r["ms_per_frame"],
                 "Autos (% of detections)": r["share_of_detections_pct"].get("auto", "no auto class")}
                for name, r in json.loads(clip_results.read_text())[clip].items()]
        st.dataframe(pd.DataFrame(rows).astype({"Autos (% of detections)": str}),
                     hide_index=True, width="stretch")
    st.warning("More detections are not automatically better: UVH-26 also makes false detections, "
               "and these clips have no labelled ground truth, so these are not accuracy scores. "
               "The camera is handheld and pans, so these clips can't give queue lengths or green "
               "times. The Live junction tab still uses YOLOv8n.")
    st.caption('Clips: "Banashankari traffic circle, Banashankari, Bangalore (2024)" 01 and 02 by '
               "Gpkp, Wikimedia Commons, CC BY-SA 4.0, with detection boxes added. "
               "UVH-26 model by AIM @ IISc, Apache 2.0.")


# --------------------------------------------------------------------------- about
with tab_how:
    st.markdown("""
**Pipeline:** camera feed → YOLOv8n vehicle detection → ByteTrack tracking → per-approach
zone counting (in PCU) → adaptive signal controller → phase timing output.

**Why PCU?** A bus needs as much green time as about three cars and a two-wheeler about
half a car (IRC:106), so queues are measured in Passenger Car Units instead of raw counts.

**Controller rules (in priority order)**
1. Minimum green of 10 s (safety and pedestrian start-up). Amber + all-red of 5 s between phases.
2. Ambulance waiting on a red approach → switch to it as soon as the minimum green ends.
3. No approach waits more than 120 s (starvation guard).
4. Maximum green of 60 s.
5. Gap-out: the current queue has cleared → give green to the approach under most pressure.
6. Another queue is far larger → hand over early.

**Baselines:** a 30 s fixed timer (what most junctions run) and a Webster pre-timed plan
(the textbook optimal fixed plan, given average demand in advance).

**Edge:** the model is exported to ONNX and benchmarked on CPU; the low-light (CLAHE)
step improves detection at night.

**City scale (P2P mesh):** each junction runs the controller above as an autonomous peer
and swaps four packet types with the junctions it shares a road with:
`STATE_GOSSIP` (how full each road is), `PLATOON` (vehicles released toward you and when they
arrive), `BACKPRESSURE_WARN` (stop feeding a road that is over 80% full) and
`PRIORITY_CORRIDOR` (ambulance coming, forwarded hop by hop so signals turn green ahead of
it). There is no central server; a killed peer falls back to isolated AI or a fixed timer.
""")
