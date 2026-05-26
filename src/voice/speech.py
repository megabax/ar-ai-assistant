"""Голосовой ввод и озвучка ответов."""

from __future__ import annotations


class VoiceInput:
    """Распознавание речи (заглушка)."""

    def listen(self) -> str | None:
        # TODO: SpeechRecognition + микрофон
        return None


class VoiceOutput:
    """Синтез речи (заглушка)."""

    def speak(self, text: str) -> None:
        # TODO: pyttsx3 / edge-tts
        pass
