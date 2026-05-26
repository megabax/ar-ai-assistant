"""Детекторы объектов и разметки (QR, ArUco, лица, руки)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    def analyze(self, frame: Any) -> FrameContext:
        detections: list[Detection] = []

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
            summary=self._build_summary(detections),
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

    @staticmethod
    def _build_summary(detections: list[Detection]) -> str:
        if not detections:
            return "Объекты не обнаружены (шаблон)."
        parts = [f"{d.kind}: {d.label}" for d in detections]
        return "; ".join(parts)
