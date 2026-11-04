"""Download a sample traffic video and the YOLOv8n weights into data/.

Usage:  python scripts/download_demo.py
"""
import os
import shutil
from pathlib import Path

from supervision.assets import VideoAssets, download_assets
from ultralytics import YOLO

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

os.chdir(DATA)
name = download_assets(VideoAssets.VEHICLES)
print("video:", DATA / name)

YOLO("yolov8n.pt")  # downloads weights to the current folder if missing
if Path("yolov8n.pt").exists():
    print("weights:", DATA / "yolov8n.pt")
else:
    src = Path.home() / "yolov8n.pt"
    if src.exists():
        shutil.copy(src, DATA / "yolov8n.pt")
