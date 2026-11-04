"""Vehicle detection, tracking and per-approach counting from a camera / video feed.

Pipeline per frame:
  (optional low-light enhancement) -> YOLO detection -> ByteTrack tracking
  -> which approach zone each vehicle is in -> counts, PCU and unique vehicles seen.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from . import config as C

ZONE_COLOURS = [(66, 135, 245), (245, 166, 35), (80, 200, 120), (220, 80, 200)]  # BGR


@dataclass
class Zone:
    name: str
    polygon: list[list[float]]   # normalised (0-1) x,y points


def load_zones(path: str | Path) -> list[Zone]:
    data = json.loads(Path(path).read_text())
    return [Zone(z["name"], z["polygon"]) for z in data["zones"]]


def enhance_low_light(frame: np.ndarray, threshold: float = 70) -> tuple[np.ndarray, bool]:
    """Brighten dark (night) frames with CLAHE on the luminance channel."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() >= threshold:
        return frame, False
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), True


@dataclass
class FrameResult:
    annotated: np.ndarray
    counts: dict[str, dict[str, int]]     # zone -> kind -> vehicles currently in zone
    pcu: dict[str, float]                 # zone -> PCU currently in zone
    unique_seen: dict[str, int]           # zone -> distinct tracked vehicles so far
    latency_ms: float
    enhanced: bool


@dataclass
class TrafficDetector:
    zones: list[Zone]
    model_path: str = "yolov8n.pt"
    conf: float = 0.3
    imgsz: int = 640
    device: str | None = None
    low_light: bool = True
    _seen: dict[str, set] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.model = YOLO(self.model_path, task="detect")
        self._seen = {z.name: set() for z in self.zones}
        self._wh = None
        self._pz: list[sv.PolygonZone] = []

    def _build_zones(self, w: int, h: int):
        self._wh = (w, h)
        sc = w / 1280
        self.box = sv.BoxAnnotator(thickness=max(1, int(2 * sc)))
        self.label = sv.LabelAnnotator(text_scale=0.45 * sc, text_thickness=max(1, int(sc)),
                                       text_padding=int(4 * sc), smart_position=True)
        self._pz = [
            sv.PolygonZone(
                polygon=(np.array(z.polygon) * [w, h]).astype(int),
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )
            for z in self.zones
        ]

    def reset(self):
        self.model = YOLO(self.model_path, task="detect")  # fresh ByteTrack state
        self._seen = {z.name: set() for z in self.zones}

    def process(self, frame: np.ndarray) -> FrameResult:
        h, w = frame.shape[:2]
        if self._wh != (w, h):
            self._build_zones(w, h)
        enhanced = False
        if self.low_light:
            frame, enhanced = enhance_low_light(frame)

        t0 = time.perf_counter()
        # Ultralytics' built-in ByteTrack keeps vehicle IDs across frames.
        res = self.model.track(frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                               persist=True, tracker="bytetrack.yaml", verbose=False,
                               classes=list(C.COCO_VEHICLES))[0]
        latency = (time.perf_counter() - t0) * 1000

        det = sv.Detections.from_ultralytics(res)
        det = det[np.isin(det.class_id, list(C.COCO_VEHICLES))]
        if det.tracker_id is None:  # no tracked vehicles in this frame
            det.tracker_id = np.full(len(det), -1, dtype=int)

        counts, pcu = {}, {}
        out = frame.copy()
        for i, (z, pz) in enumerate(zip(self.zones, self._pz)):
            mask = pz.trigger(det)
            kinds: dict[str, int] = {}
            for cid, tid in zip(det.class_id[mask], det.tracker_id[mask]):
                k = C.COCO_VEHICLES[int(cid)]
                kinds[k] = kinds.get(k, 0) + 1
                if tid >= 0:
                    self._seen[z.name].add(int(tid))
            counts[z.name] = kinds
            pcu[z.name] = sum(C.PCU[k] * n for k, n in kinds.items())
        s = w / 1280  # scale drawing to the frame size (4K video -> 3x)
        th = max(1, int(round(2 * s)))
        overlay = out.copy()
        polys = [(np.array(z.polygon) * [w, h]).astype(np.int32) for z in self.zones]
        for i, poly in enumerate(polys):
            cv2.fillPoly(overlay, [poly], ZONE_COLOURS[i % len(ZONE_COLOURS)])
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        for i, poly in enumerate(polys):
            cv2.polylines(out, [poly], True, ZONE_COLOURS[i % len(ZONE_COLOURS)], th)

        labels = [f"#{tid} {C.COCO_VEHICLES[int(c)]}"
                  for c, tid in zip(det.class_id, det.tracker_id)]
        out = self.box.annotate(out, det)
        out = self.label.annotate(out, det, labels=labels)

        # Zone labels on a dark tag at each zone's centre, drawn last so they stay readable.
        for i, (z, poly) in enumerate(zip(self.zones, polys)):
            text = f"{z.name}: {sum(counts[z.name].values())} veh / {pcu[z.name]:.1f} PCU"
            (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, th)
            cx, cy = poly.mean(axis=0).astype(int)
            x0 = int(min(max(cx - tw // 2, 4), w - tw - 8))
            y0 = int(cy + i % 2 * (tht + 14 * s))  # stagger neighbours so tags don't touch
            cv2.rectangle(out, (x0 - 6, y0 - tht - 6), (x0 + tw + 6, y0 + base + 4), (20, 20, 20), -1)
            cv2.rectangle(out, (x0 - 6, y0 - tht - 6), (x0 + tw + 6, y0 + base + 4),
                          ZONE_COLOURS[i % len(ZONE_COLOURS)], th)
            cv2.putText(out, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s,
                        (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(out, f"{latency:.0f} ms/frame{'  [night mode]' if enhanced else ''}",
                    (int(10 * s), h - int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                    (0, 255, 255), th, cv2.LINE_AA)

        return FrameResult(out, counts, pcu,
                           {k: len(v) for k, v in self._seen.items()}, latency, enhanced)
