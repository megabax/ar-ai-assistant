"""Скачивание файлов YOLOv4-tiny в папку models/."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

FILES = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dst = MODELS_DIR / filename
        if dst.exists():
            print(f"skip: {dst}")
            continue
        print(f"download: {filename}")
        urlretrieve(url, dst)
    print("done")


if __name__ == "__main__":
    main()
