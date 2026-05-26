import sys

from PyQt6.QtWidgets import QApplication

from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.ui.main_window import MainWindow


def run() -> None:
    config = AppConfig()
    app = QApplication(sys.argv)
    pipeline = FramePipeline(config)
    window = MainWindow(config, pipeline)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
