"""Детекторы объектов и разметки (YOLO, QR, ArUco, лица, руки)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_CFG_PATH = MODELS_DIR / "yolov4-tiny.cfg"
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolov4-tiny.weights"
YOLO_NAMES_PATH = MODELS_DIR / "coco.names"


@dataclass
class Detection:
    kind: str
    label: str
    bbox: tuple[int, int, int, int] | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameContext:
    """Результат анализа одного кадра."""

    frame: Any
    detections: list[Detection] = field(default_factory=list)
    summary: str = ""


class VisionService:
    """Агрегирует детекторы. Каждый детектор — отдельный модуль в будущем."""

    def __init__(self, enabled: list[str]) -> None:
        self.enabled = set(enabled)
        self._yolo_model: cv2.dnn_DetectionModel | None = None
        self._yolo_classes: list[str] = []
        self._yolo_initialized = False

    def analyze(self, frame: Any) -> FrameContext:
        detections: list[Detection] = []

        if "yolo" in self.enabled:
            detections.extend(self._detect_yolo(frame))
        if "qr" in self.enabled:
            detections.extend(self._detect_qr(frame))
        if "aruco" in self.enabled:
            detections.extend(self._detect_aruco(frame))
        if "face" in self.enabled:
            detections.extend(self._detect_faces(frame))
        if "hand" in self.enabled:
            detections.extend(self._detect_hands(frame))

        return FrameContext(
            frame=frame,
            detections=detections,
            summary=self._build_summary(detections, frame is not None),
        )

    def _detect_qr(self, frame: Any) -> list[Detection]:
        # TODO: pyzbar / opencv QRCodeDetector
        return []

    def _detect_aruco(self, frame: Any) -> list[Detection]:
        # TODO: cv2.aruco
        return []

    def _detect_faces(self, frame: Any) -> list[Detection]:
        # TODO: mediapipe / haar cascades
        return []

    def _detect_hands(self, frame: Any) -> list[Detection]:
        # TODO: mediapipe hands
        return []

    def _detect_yolo(self, frame: Any) -> list[Detection]:
        if frame is None:
            return []
        if not self._yolo_initialized:
            self._init_yolo()
        if self._yolo_model is None:
            return []

        class_ids, confidences, boxes = self._yolo_model.detect(
            frame,
            confThreshold=0.45,
            nmsThreshold=0.4,
        )

        detections: list[Detection] = []
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            class_idx = int(class_id[0]) if hasattr(class_id, "__len__") else int(class_id)
            score = float(confidence[0]) if hasattr(confidence, "__len__") else float(confidence)
            if 0 <= class_idx < len(self._yolo_classes):
                label = self._yolo_classes[class_idx]
            else:
                label = f"class_{class_idx}"

            x, y, w, h = [int(v) for v in box]
            detections.append(
                Detection(
                    kind="yolo",
                    label=label,
                    bbox=(x, y, w, h),
                    payload={"confidence": score},
                )
            )
        return detections

    def _init_yolo(self) -> None:
        self._yolo_initialized = True
        if not (YOLO_CFG_PATH.exists() and YOLO_WEIGHTS_PATH.exists()):
            return

        net = cv2.dnn.readNetFromDarknet(str(YOLO_CFG_PATH), str(YOLO_WEIGHTS_PATH))
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1 / 255.0, swapRB=True)
        self._yolo_model = model
        self._yolo_classes = self._load_classes()

    def _load_classes(self) -> list[str]:
        if not YOLO_NAMES_PATH.exists():
            return []
        with YOLO_NAMES_PATH.open("r", encoding="utf-8") as names_file:
            return [line.strip() for line in names_file if line.strip()]

    @staticmethod
    def _build_summary(detections: list[Detection], has_frame: bool = False) -> str:
        if not detections:
            if has_frame:
                return "Камера активна. Объекты не обнаружены."
            return "Камера не активна."
        parts = [f"{d.kind}: {d.label}" for d in detections]
        return "; ".join(parts)
