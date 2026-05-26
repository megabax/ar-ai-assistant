"""Оркестрация кадра: камера → детекция → оверлей → UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ai.assistant import AIAssistant
from src.camera.capture import CameraSource
from src.config import AppConfig
from src.overlay.renderer import OverlayRenderer
from src.vision.detectors import VisionService

if TYPE_CHECKING:
    from src.vision.detectors import FrameContext


class FramePipeline:
    """Связывает подсистемы в единый цикл обработки (пока — заглушки)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = CameraSource(config.camera_index)
        self.vision = VisionService(config.enabled_detectors)
        self.overlay = OverlayRenderer()
        self.assistant = AIAssistant()
        self._running = False

    def start(self) -> None:
        self._running = True
        self.camera.open()

    def stop(self) -> None:
        self._running = False
        self.camera.close()

    @property
    def is_running(self) -> bool:
        return self._running

    def process_frame(self) -> FrameContext | None:
        """Один шаг цикла: захват → детекция → отрисовка оверлея."""
        if not self._running:
            return None

        frame = self.camera.read()
        if frame is None:
            return None

        context = self.vision.analyze(frame)
        self.overlay.render(context)
        return context

    def ask(self, question: str, context: FrameContext | None = None) -> str:
        """Текстовый или голосовой вопрос к ИИ о текущей сцене."""
        return self.assistant.answer(question, context)
