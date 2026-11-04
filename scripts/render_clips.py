"""Render side-by-side videos of YOLOv8n and UVH-26 on the Bengaluru clips.

Usage:  python scripts/render_clips.py
Writes: data/clip_videos/<clip>.webm (VP8, plays in any browser)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyse_clips import CONF, IMGSZ, STRIDE, UVH_KINDS  # noqa: E402
from traffic import config as C  # noqa: E402

PANEL = (960, 540)
DETECTORS = {
    "YOLOv8n": (ROOT / "data/yolov8n.pt", lambda i, name: i in C.COCO_VEHICLES),
    "UVH-26": (ROOT / "data/models/UVH-26-MV-YOLOv11-S.pt", lambda i, name: name in UVH_KINDS),
}


def caption(picture, text):
    cv2.rectangle(picture, (0, 0), (picture.shape[1], 44), (35, 35, 35), -1)
    cv2.putText(picture, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return picture


def render(clip: Path, out: Path):
    models = {}
    for label, (path, wanted) in DETECTORS.items():
        model = YOLO(str(path), task="detect")
        models[label] = (model, [i for i, n in model.names.items() if wanted(i, n)])
    video = cv2.VideoCapture(str(clip))
    fps = video.get(cv2.CAP_PROP_FPS) / STRIDE
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"VP80"), fps, (PANEL[0] * 2, PANEL[1]))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open a VP8 video writer")
    index, written = -1, 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        index += 1
        if index % STRIDE:
            continue
        panels = []
        for label, (model, classes) in models.items():
            res = model.track(frame, conf=CONF, imgsz=IMGSZ, classes=classes, persist=True,
                              tracker="bytetrack.yaml", verbose=False)[0]
            picture = cv2.resize(res.plot(line_width=3), PANEL)
            panels.append(caption(picture, f"{label}: {len(res.boxes)} vehicles detected"))
        writer.write(np.hstack(panels))
        written += 1
    video.release()
    writer.release()
    print(f"{out.name}: {written} frames, {out.stat().st_size / 1e6:.1f} MB", flush=True)


def main():
    out = ROOT / "data/clip_videos"
    out.mkdir(exist_ok=True)
    for clip in sorted((ROOT / "data/clips").glob("*.ogg")):
        render(clip, out / f"{clip.stem}.webm")


if __name__ == "__main__":
    main()
