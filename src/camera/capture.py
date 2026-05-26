"""Захват видео с камеры через OpenCV."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np


class CameraSource:
    """Источник кадров (BGR, numpy.ndarray)."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        if self.is_opened:
            return True

        if sys.platform == "win32":
            capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.device_index)

        if not capture.isOpened():
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture = capture
        return True

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> np.ndarray | None:
        if not self.is_opened:
            return None

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame
