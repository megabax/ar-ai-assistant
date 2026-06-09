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

from src.ai.assistant import AIResponse
from src.config import AppConfig
from src.core.pipeline import FramePipeline
from src.ui.frame_view import frame_to_pixmap
from src.voice.speech import VoiceListenResult
from src.voice.worker import VoiceListenWorker


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, pipeline: FramePipeline) -> None:
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self._voice_worker: VoiceListenWorker | None = None
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
            self._video_label.clear()
            self._video_label.setText("Превью камеры\n(камера остановлена)")
            self._append_system("Камера остановлена.")
        else:
            if not self.pipeline.start():
                self._append_system(
                    "Не удалось открыть камеру. Проверьте подключение и индекс в config."
                )
                return
            self._frame_timer.start()
            self._btn_camera.setText("Стоп камеры")
            self._video_label.clear()
            self._append_system("Камера запущена.")

    def _on_frame_tick(self) -> None:
        context = self.pipeline.process_frame()
        if context is None or context.frame is None:
            return
        self._last_context = context
        pixmap = frame_to_pixmap(
            context.frame,
            self._video_label.width(),
            self._video_label.height(),
        )
        self._video_label.setPixmap(pixmap)

    def _on_send_text(self) -> None:
        question = self._question_input.text().strip()
        if not question:
            return
        self._question_input.clear()
        self._chat_log.append(f"<b>Вы:</b> {question}")
        response = self.pipeline.ask(question, self._last_context)
        self._append_ai_response(response)

    def _on_voice(self) -> None:
        if self._voice_worker is not None and self._voice_worker.isRunning():
            return

        self._btn_voice.setEnabled(False)
        seconds = self.config.voice_listen_seconds
        self._append_system(f"Слушаю микрофон ({seconds:.0f} с)…")
        self._voice_worker = VoiceListenWorker(self.config, self)
        self._voice_worker.finished.connect(self._on_voice_finished)
        self._voice_worker.start()

    def _on_voice_finished(self, result: VoiceListenResult) -> None:
        self._btn_voice.setEnabled(True)
        if result.error:
            self._append_system(result.error)
            return
        if not result.text:
            self._append_system("Речь не распознана.")
            return

        elapsed = self._format_elapsed(result.elapsed_s)
        self._chat_log.append(
            f"<span style='color:#6a9955;'>Распознано ({elapsed}): {result.text}</span>"
        )
        self._question_input.setText(result.text)
        self._on_send_text()

    def _append_ai_response(self, response: AIResponse) -> None:
        text = response.content.replace("\n", "<br>")
        elapsed = self._format_elapsed(response.elapsed_s)
        self._chat_log.append(f"<b>ИИ:</b> {text}")
        meta = f"Модель: {response.model} · {elapsed}"
        if response.is_stub:
            meta += " · заглушка"
        if response.error:
            meta += f" · {response.error}"
        self._chat_log.append(f"<span style='color:#6a9955;'>{meta}</span>")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        if seconds < 1.0:
            return f"{seconds * 1000:.0f} мс"
        return f"{seconds:.2f} с"

    def _append_system(self, message: str) -> None:
        self._chat_log.append(f"<i style='color:#888;'>{message}</i>")

    def closeEvent(self, event) -> None:
        if self._voice_worker is not None and self._voice_worker.isRunning():
            self._voice_worker.wait(3000)
        self.pipeline.stop()
        super().closeEvent(event)
