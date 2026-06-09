from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOSK_MODEL_DIR = PROJECT_ROOT / "models" / "vosk-model-small-ru-0.22"


@dataclass
class AppConfig:
    """Настройки приложения (расширять по мере разработки)."""

    window_title: str = "AI Pet — псевдо-AR ассистент"
    window_width: int = 1280
    window_height: int = 720
    camera_index: int = 0
    target_fps: int = 30
    enabled_detectors: list[str] = field(
        default_factory=lambda: ["yolo", "qr", "aruco", "face", "hand"]
    )
    vosk_model_dir: Path = field(default_factory=lambda: DEFAULT_VOSK_MODEL_DIR)
    voice_listen_seconds: float = 5.0
    voice_sample_rate: int = 16000
