import sys
import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from src.ai.assistant import AIAssistant
from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.ui.frame_view import frame_to_pixmap
from src.vision.detectors import FrameContext, VisionService


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_app = QApplication.instance() or QApplication(sys.argv)

    def test_config_defaults(self) -> None:
        cfg = AppConfig()
        self.assertIn("qr", cfg.enabled_detectors)

    def test_vision_empty_frame(self) -> None:
        ctx = VisionService(["qr"]).analyze(frame=None)
        self.assertEqual(ctx.detections, [])
        self.assertIn("не активна", ctx.summary)

    def test_frame_to_pixmap(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 2] = 200
        pixmap = frame_to_pixmap(frame, 160, 120)
        self.assertFalse(pixmap.isNull())

    def test_assistant_stub(self) -> None:
        reply = AIAssistant().answer("Что это?", FrameContext(frame=None))
        self.assertIn("Шаблон", reply)

    def test_pipeline_ask_without_camera(self) -> None:
        pipe = FramePipeline(AppConfig())
        answer = pipe.ask("Привет")
        self.assertIn("Камера не активна", answer)


if __name__ == "__main__":
    unittest.main()
