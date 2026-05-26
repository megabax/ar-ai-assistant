"""Захват видео с камеры (заглушка под OpenCV)."""

from __future__ import annotations

from typing import Any


class CameraSource:
    """Источник кадров. В будущем — cv2.VideoCapture."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._opened = False

    def open(self) -> None:
        self._opened = True
        # TODO: self._capture = cv2.VideoCapture(self.device_index)

    def close(self) -> None:
        self._opened = False
        # TODO: release capture

    def read(self) -> Any | None:
        if not self._opened:
            return None
        # TODO: ret, frame = self._capture.read()
        return None
