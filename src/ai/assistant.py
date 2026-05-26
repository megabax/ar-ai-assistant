"""ИИ-ассистент: ответы о том, что видит камера."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vision.detectors import FrameContext


class AIAssistant:
    """Обёртка над LLM / vision API (пока заглушка)."""

    def answer(self, question: str, context: FrameContext | None = None) -> str:
        scene = context.summary if context else "камера не активна"
        # TODO: multimodal prompt (frame + detections + question)
        return (
            f"[Шаблон] Вопрос: «{question.strip()}»\n"
            f"Контекст сцены: {scene}\n"
            "Здесь будет ответ ИИ с подсказками и объяснениями."
        )
