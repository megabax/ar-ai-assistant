"""Наложение подсказок и псевдо-3D поверх кадра."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from src.vision.detectors import Detection, FrameContext


class OverlayRenderer:
    """Рисует AI-подсказки и анимацию поверх видео."""

    def render(self, context: FrameContext) -> None:
        for detection in context.detections:
            self._draw_hint(context.frame, detection)

    def _draw_hint(self, frame, detection: Detection) -> None:
        if frame is None or detection.bbox is None:
            return
        x, y, w, h = detection.bbox
        color = (0, 220, 0) if detection.kind == "yolo" else (230, 140, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        confidence = detection.payload.get("confidence")
        if isinstance(confidence, float):
            title = f"{detection.label} {confidence:.2f}"
        else:
            title = detection.label
        label_y = y - 10 if y > 20 else y + 20
        cv2.putText(
            frame,
            title,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
