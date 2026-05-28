from dataclasses import dataclass, field


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
