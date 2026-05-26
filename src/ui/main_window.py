"""Главное окно: превью камеры, чат с ИИ, управление."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.voice.speech import VoiceInput


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, pipeline: FramePipeline) -> None:
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self.voice = VoiceInput()
        self._last_context = None

        self.setWindowTitle(config.window_title)
        self.resize(config.window_width, config.window_height)

        self._video_label = QLabel("Превью камеры\n(запустите камеру)")
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #aaa; border: 1px solid #444;"
        )
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._chat_log = QTextEdit()
        self._chat_log.setReadOnly(True)
        self._chat_log.setPlaceholderText("Ответы ИИ появятся здесь…")

        self._question_input = QLineEdit()
        self._question_input.setPlaceholderText("Спросите ИИ о том, что видит камера…")
        self._question_input.returnPressed.connect(self._on_send_text)

        self._btn_camera = QPushButton("Старт камеры")
        self._btn_camera.clicked.connect(self._toggle_camera)

        self._btn_send = QPushButton("Отправить")
        self._btn_send.clicked.connect(self._on_send_text)

        self._btn_voice = QPushButton("Голос")
        self._btn_voice.clicked.connect(self._on_voice)

        input_row = QHBoxLayout()
        input_row.addWidget(self._question_input, stretch=1)
        input_row.addWidget(self._btn_voice)
        input_row.addWidget(self._btn_send)

        right = QVBoxLayout()
        right.addWidget(QLabel("Диалог с ИИ"))
        right.addWidget(self._chat_log, stretch=1)
        right.addLayout(input_row)

        root = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self._video_label, stretch=1)
        left.addWidget(self._btn_camera)
        root.addLayout(left, stretch=2)
        root.addLayout(right, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        mono = QFont("Consolas", 10)
        self._chat_log.setFont(mono)

        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(1000 // max(config.target_fps, 1))
        self._frame_timer.timeout.connect(self._on_frame_tick)

        self._append_system(
            "Шаблон AI Pet. Включите камеру и задайте вопрос текстом или голосом."
        )

    def _toggle_camera(self) -> None:
        if self.pipeline.is_running:
            self.pipeline.stop()
            self._frame_timer.stop()
            self._btn_camera.setText("Старт камеры")
            self._video_label.setText("Превью камеры\n(камера остановлена)")
            self._append_system("Камера остановлена.")
        else:
            self.pipeline.start()
            self._frame_timer.start()
            self._btn_camera.setText("Стоп камеры")
            self._video_label.setText("Камера активна\n(OpenCV — в разработке)")
            self._append_system("Камера запущена (захват кадров — заглушка).")

    def _on_frame_tick(self) -> None:
        self._last_context = self.pipeline.process_frame()

    def _on_send_text(self) -> None:
        question = self._question_input.text().strip()
        if not question:
            return
        self._question_input.clear()
        self._chat_log.append(f"<b>Вы:</b> {question}")
        answer = self.pipeline.ask(question, self._last_context)
        self._chat_log.append(f"<b>ИИ:</b> {answer.replace(chr(10), '<br>')}")

    def _on_voice(self) -> None:
        self._append_system("Голосовой ввод пока не подключён.")
        text = self.voice.listen()
        if text:
            self._question_input.setText(text)
            self._on_send_text()

    def _append_system(self, message: str) -> None:
        self._chat_log.append(f"<i style='color:#888;'>{message}</i>")

    def closeEvent(self, event) -> None:
        self.pipeline.stop()
        super().closeEvent(event)
