import os
import sys
import unittest

import numpy as np
from PyQt6.QtWidgets import QApplication

from src.ai.assistant import AIAssistant
from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.ui.frame_view import frame_to_pixmap
from src.vision.detectors import FrameContext, VisionService
from src.voice.speech import VoiceInput


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Не дергаем Ollama в unit-тестах
        os.environ["AI_PET_ENABLE_LLM"] = "0"
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
        self.assertIn("Шаблон", reply.content)
        self.assertEqual(reply.model, "заглушка")

    def test_pipeline_ask_without_camera(self) -> None:
        pipe = FramePipeline(AppConfig())
        answer = pipe.ask("Привет")
        self.assertIn("Камера не активна", answer.content)

    def test_voice_disabled(self) -> None:
        os.environ["AI_PET_ENABLE_VOICE"] = "0"
        result = VoiceInput(AppConfig()).listen()
        self.assertIsNone(result.text)
        self.assertIn("отключён", result.error or "")

    def test_voice_model_missing(self) -> None:
        os.environ["AI_PET_ENABLE_VOICE"] = "1"
        cfg = AppConfig()
        cfg.vosk_model_dir = cfg.vosk_model_dir.parent / "missing-vosk-model"
        result = VoiceInput(cfg).listen()
        self.assertIsNone(result.text)
        self.assertIn("Vosk", result.error or "")


if __name__ == "__main__":
    unittest.main()
