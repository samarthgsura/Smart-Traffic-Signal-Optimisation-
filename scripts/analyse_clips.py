"""Analyse real Bengaluru clips with two detectors: YOLOv8n (COCO) and UVH-26 YOLOv11-S.

The Banashankari clips are handheld and pan around the circle, so fixed approach zones,
queue lengths and green-time recommendations would not be valid. This script measures what
a moving camera does support: vehicles visible per frame, vehicle mix and detector speed.

Usage:  python scripts/download_bengaluru.py
        python scripts/analyse_clips.py
Writes: data/bengaluru_clips.json and side-by-side frames in data/clip_frames/
"""
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic import config as C  # noqa: E402

STRIDE = 3          # analyse 10 of the 30 frames per second
IMGSZ = 640         # same input size for both detectors
CONF = 0.3
MIN_TRACK_HITS = 3  # ignore tracks seen in fewer analysed frames than this
SAMPLE_EVERY_S = 5  # save a comparison image this often

# Passenger Car Units. Auto-rickshaw and LCV are not in the COCO classes, so they are added here.
PCU = {**C.PCU, "auto": 1.2, "lcv": 1.4, "other": 1.0}

UVH_KINDS = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car", "Van": "car",
    "Tempo-traveller": "lcv", "LCV": "lcv", "Bus": "bus", "Mini-bus": "bus", "Truck": "truck",
    "Three-wheeler": "auto", "Two-wheeler": "motorcycle", "Bicycle": "bicycle", "Others": "other",
}
MODELS = {
    "YOLOv8n (COCO)": (ROOT / "data/yolov8n.pt", lambda name: C.COCO_VEHICLES.get(_coco_id[name])),
    "UVH-26 YOLOv11-S": (ROOT / "data/models/UVH-26-MV-YOLOv11-S.pt", lambda name: UVH_KINDS.get(name)),
}
_coco_id = {}


def analyse(clip: Path, label: str, path: Path, to_kind, samples: dict):
    model = YOLO(str(path), task="detect")
    if not _coco_id and "COCO" in label:
        _coco_id.update({n: i for i, n in model.names.items()})
    keep = [i for i, n in model.names.items() if to_kind(n)]
    video = cv2.VideoCapture(str(clip))
    fps = video.get(cv2.CAP_PROP_FPS)
    visible, hits, ms = [], collections.defaultdict(collections.Counter), []
    kind_frames = collections.Counter()
    index = -1
    while True:
        ok, frame = video.read()
        if not ok:
            break
        index += 1
        if index % STRIDE:
            continue
        start = time.perf_counter()
        res = model.track(frame, conf=CONF, imgsz=IMGSZ, classes=keep, persist=True,
                          tracker="bytetrack.yaml", verbose=False)[0]
        ms.append((time.perf_counter() - start) * 1000)
        kinds = [to_kind(model.names[int(c)]) for c in res.boxes.cls]
        visible.append(len(kinds))
        kind_frames.update(kinds)
        if res.boxes.id is not None:
            for tid, kind in zip(res.boxes.id.int().tolist(), kinds):
                hits[tid][kind] += 1
        if index % int(fps * SAMPLE_EVERY_S) < STRIDE:
            picture = cv2.resize(res.plot(line_width=3), (1280, 720))
            cv2.putText(picture, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
            cv2.putText(picture, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            samples.setdefault(index, []).append(picture)
    video.release()

    # Each track takes its most frequent label, so a flickering class counts only once.
    tracks = collections.Counter(
        h.most_common(1)[0][0] for h in hits.values() if sum(h.values()) >= MIN_TRACK_HITS)
    total_seen = sum(kind_frames.values()) or 1
    return {
        "frames_analysed": len(visible),
        "avg_visible": round(float(np.mean(visible)), 1),
        "max_visible": max(visible),
        "ms_per_frame": round(float(np.median(ms)), 1),
        "tracks_by_kind": dict(tracks.most_common()),
        "tracks_total": sum(tracks.values()),
        "share_of_detections_pct": {k: round(100 * n / total_seen, 1)
                                    for k, n in kind_frames.most_common()},
        "avg_visible_pcu": round(sum(PCU[k] * n for k, n in kind_frames.items()) / len(visible), 1),
    }


def main():
    out = ROOT / "data/clip_frames"
    out.mkdir(exist_ok=True)
    models = {k: v for k, v in MODELS.items() if "--coco-only" not in sys.argv or "COCO" in k}
    results = {}
    for clip in sorted((ROOT / "data/clips").glob("*.ogg")):
        samples = {}
        results[clip.stem] = {}
        for label, (path, to_kind) in models.items():
            r = analyse(clip, label, path, to_kind, samples)
            results[clip.stem][label] = r
            print(clip.stem, label, json.dumps(r), flush=True)
        for index, pictures in samples.items():
            cv2.imwrite(str(out / f"{clip.stem}_{index:04d}.jpg"), np.vstack(pictures),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
    (ROOT / "data/bengaluru_clips.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
