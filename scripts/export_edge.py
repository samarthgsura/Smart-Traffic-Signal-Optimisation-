"""Export YOLOv8n to ONNX and benchmark CPU latency (edge-readiness check).

Usage:  python scripts/export_edge.py [--video data/vehicles.mp4] [--frames 60]
Writes data/edge_benchmark.json, which the dashboard displays.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DATA = Path(__file__).resolve().parents[1] / "data"


def bench(model, frames, imgsz):
    model.predict(frames[0], imgsz=imgsz, device="cpu", verbose=False)  # warm-up
    times = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=str(DATA / "vehicles.mp4"))
    p.add_argument("--frames", type=int, default=60)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    if not frames:
        raise SystemExit(f"Could not read frames from {args.video}")

    pt = DATA / "yolov8n.pt"
    results = []
    for imgsz in (640, 416, 320):
        onnx_path = Path(YOLO(str(pt)).export(format="onnx", imgsz=imgsz, simplify=True,
                                              dynamic=False))
        target = DATA / f"yolov8n_{imgsz}.onnx"
        onnx_path.replace(target)
        pt_ms = bench(YOLO(str(pt)), frames, imgsz)
        onnx_ms = bench(YOLO(str(target), task="detect"), frames, imgsz)
        results.append({"imgsz": imgsz, "pytorch_cpu_ms": round(pt_ms, 1),
                        "onnx_cpu_ms": round(onnx_ms, 1),
                        "onnx_fps": round(1000 / onnx_ms, 1),
                        "onnx_mb": round(target.stat().st_size / 1e6, 1)})
        print(results[-1])

    out = {"cpu": platform.processor() or platform.machine(), "frames": len(frames),
           "results": results}
    (DATA / "edge_benchmark.json").write_text(json.dumps(out, indent=2))
    print("saved", DATA / "edge_benchmark.json")


if __name__ == "__main__":
    main()
