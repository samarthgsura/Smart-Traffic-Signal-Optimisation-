"""Download real Bengaluru footage and a detector trained on Bengaluru cameras.

Clips:   Banashankari traffic circle, Bengaluru, by Gpkp (31 Aug 2024),
         Wikimedia Commons, CC BY-SA 4.0.
Model:   UVH-26 YOLOv11-S (IISc AIM), trained on Bengaluru Safe-City CCTV, Apache-2.0.

Usage:  python scripts/download_bengaluru.py
"""
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMONS = "https://upload.wikimedia.org/wikipedia/commons"
HF = "https://huggingface.co/iisc-aim/UVH-26/resolve/main"
FILES = {
    ROOT / "data/clips/banashankari_01.ogg":
        f"{COMMONS}/1/1e/Banashankari_traffic_circle%2C_Banashankari%2C_Bangalore_%282024%29_01.ogg",
    ROOT / "data/clips/banashankari_02.ogg":
        f"{COMMONS}/7/7f/Banashankari_traffic_circle%2C_Banashankari%2C_Bangalore_%282024%29_02.ogg",
    ROOT / "data/models/UVH-26-MV-YOLOv11-S.pt":
        f"{HF}/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt",
    ROOT / "data/models/uvh_classes.txt":
        f"{HF}/uvh_classes.txt",
}


def main():
    for path, url in FILES.items():
        if path.exists() and path.stat().st_size:
            print("have", path.name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        for attempt in range(8):  # resume after dropped connections
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"User-Agent": "neural-storm-ft01/1.0", "Range": f"bytes={have}-"}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    mode = "ab" if r.status == 206 else "wb"
                    with open(tmp, mode) as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                break
            except urllib.error.HTTPError as e:
                if e.code == 416:  # already complete
                    break
                print(f"  HTTP {e.code}, retrying")
            except OSError as e:
                print(f"  {e}, retrying from {have / 1e6:.1f} MB")
            time.sleep(3 + attempt * 3)
        else:
            raise SystemExit(f"could not download {url}")
        tmp.replace(path)
        print(f"got  {path.name}  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
