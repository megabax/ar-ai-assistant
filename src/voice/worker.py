"""Фоновое распознавание речи, чтобы не блокировать UI."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal

from src.config import AppConfig
from src.voice.speech import VoiceInput, VoiceListenResult


class VoiceListenWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self._voice = VoiceInput(config)

    def run(self) -> None:
        result = self._voice.listen()
        self.finished.emit(result)
