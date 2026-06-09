"""Голосовой ввод (Vosk, офлайн, CPU) и озвучка ответов."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import AppConfig


@dataclass
class VoiceListenResult:
    text: str | None
    elapsed_s: float = 0.0
    error: str | None = None


class VoiceInput:
    """Распознавание речи через Vosk — лёгкая модель для слабого CPU."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self._model = None
        self._model_dir = Path(
            os.getenv("AI_PET_VOSK_MODEL_DIR", str(self.config.vosk_model_dir))
        )

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def is_available(self) -> bool:
        return (self._model_dir / "am" / "final.mdl").exists()

    def listen(self) -> VoiceListenResult:
        if os.getenv("AI_PET_ENABLE_VOICE", "1") != "1":
            return VoiceListenResult(text=None, error="Голосовой ввод отключён.")

        if not self.is_available():
            return VoiceListenResult(
                text=None,
                error=(
                    "Модель Vosk не найдена. Скачайте: "
                    "python tools/download_vosk_model.py"
                ),
            )

        started = time.perf_counter()
        try:
            text = self._listen_vosk()
            elapsed = time.perf_counter() - started
            if not text:
                return VoiceListenResult(
                    text=None,
                    elapsed_s=elapsed,
                    error="Речь не распознана. Попробуйте говорить ближе к микрофону.",
                )
            return VoiceListenResult(text=text, elapsed_s=elapsed)
        except Exception as exc:
            return VoiceListenResult(
                text=None,
                elapsed_s=time.perf_counter() - started,
                error=str(exc),
            )

    def _get_model(self):
        if self._model is None:
            from vosk import Model

            self._model = Model(str(self._model_dir))
        return self._model

    def _listen_vosk(self) -> str:
        import numpy as np
        import sounddevice as sd
        from vosk import KaldiRecognizer

        sample_rate = self.config.voice_sample_rate
        seconds = float(
            os.getenv("AI_PET_VOICE_LISTEN_SECONDS", str(self.config.voice_listen_seconds))
        )

        model = self._get_model()
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.SetWords(False)

        frames = int(seconds * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()

        chunk_samples = 4000
        audio = np.asarray(audio).reshape(-1)
        audio_bytes = audio.tobytes()
        chunk_bytes = chunk_samples * 2

        for offset in range(0, len(audio_bytes), chunk_bytes):
            recognizer.AcceptWaveform(audio_bytes[offset : offset + chunk_bytes])

        result = json.loads(recognizer.FinalResult())
        return str(result.get("text", "")).strip()


class VoiceOutput:
    """Синтез речи (заглушка)."""

    def speak(self, text: str) -> None:
        pass
