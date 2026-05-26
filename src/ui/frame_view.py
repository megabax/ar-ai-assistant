"""Преобразование кадра OpenCV в QPixmap для Qt."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


def frame_to_pixmap(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, channels = rgb.shape
    bytes_per_line = channels * w
    image = QImage(
        rgb.data,
        w,
        h,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()
    pixmap = QPixmap.fromImage(image)
    return pixmap.scaled(
        max_width,
        max_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
