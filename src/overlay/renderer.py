"""Наложение подсказок и псевдо-3D поверх кадра."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vision.detectors import FrameContext


class OverlayRenderer:
    """Рисует AI-подсказки и анимацию поверх видео."""

    def render(self, context: FrameContext) -> None:
        # TODO: OpenGL / Qt QPainter / blend с frame
        for detection in context.detections:
            self._draw_hint(detection)

    def _draw_hint(self, detection) -> None:
        # TODO: текст, стрелки, 3D-примитивы по bbox
        pass
