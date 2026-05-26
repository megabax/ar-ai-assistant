import unittest

from src.ai.assistant import AIAssistant
from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.vision.detectors import FrameContext, VisionService


class SmokeTests(unittest.TestCase):
    def test_config_defaults(self) -> None:
        cfg = AppConfig()
        self.assertIn("qr", cfg.enabled_detectors)

    def test_vision_empty_frame(self) -> None:
        ctx = VisionService(["qr"]).analyze(frame=None)
        self.assertEqual(ctx.detections, [])

    def test_assistant_stub(self) -> None:
        reply = AIAssistant().answer("Что это?", FrameContext(frame=None))
        self.assertIn("Шаблон", reply)

    def test_pipeline_ask_without_camera(self) -> None:
        pipe = FramePipeline(AppConfig())
        answer = pipe.ask("Привет")
        self.assertIn("камера не активна", answer)


if __name__ == "__main__":
    unittest.main()
