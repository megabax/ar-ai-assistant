"""ИИ-ассистент: ответы о том, что видит камера (Ollama через OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vision.detectors import FrameContext

_DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
_cached_base_url: str | None = None


@dataclass
class AIResponse:
    content: str
    model: str
    elapsed_s: float
    is_stub: bool = False
    error: str | None = None


class AIAssistant:
    """Обёртка над LLM (Ollama) + подсказки по тому, что видит камера."""

    def answer(self, question: str, context: FrameContext | None = None) -> AIResponse:
        scene = context.summary if context else "Камера не активна."

        if os.getenv("AI_PET_ENABLE_LLM", "1") != "1":
            return self._template_answer(question, scene)

        api_key = os.getenv("AI_PET_OLLAMA_API_KEY", "ollama")
        model = os.getenv("AI_PET_OLLAMA_MODEL", "phi3:3.8b")
        temperature = float(os.getenv("AI_PET_TEMPERATURE", "0.0"))
        top_p = float(os.getenv("AI_PET_TOP_P", "0.1"))
        timeout_s = float(os.getenv("AI_PET_LLM_TIMEOUT", "120"))

        base_url = self._resolve_base_url(os.getenv("AI_PET_OLLAMA_BASE_URL"))
        if base_url is None:
            return self._template_answer(
                question,
                scene,
                model=model,
                error=self._connection_help(),
            )

        prompt = self._build_prompt(question, scene)

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_s,
            )
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.perf_counter() - started
            return AIResponse(
                content=response.choices[0].message.content or "",
                model=model,
                elapsed_s=elapsed,
            )
        except Exception as openai_error:
            try:
                started = time.perf_counter()
                content = self._call_ollama_http(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    timeout_s=timeout_s,
                )
                elapsed = time.perf_counter() - started
                return AIResponse(content=content, model=model, elapsed_s=elapsed)
            except Exception as http_error:
                global _cached_base_url
                _cached_base_url = None
                return self._template_answer(
                    question,
                    scene,
                    model=model,
                    error=self._format_error(openai_error, http_error, base_url),
                )

    @classmethod
    def _resolve_base_url(cls, configured: str | None) -> str | None:
        global _cached_base_url
        if _cached_base_url and cls._probe_ollama(_cached_base_url):
            return _cached_base_url

        candidates: list[str] = []
        if configured:
            candidates.append(cls._normalize_base_url(configured))
        candidates.append(_DEFAULT_BASE_URL)

        wsl_ip = cls._get_wsl_ip()
        if wsl_ip:
            candidates.append(f"http://{wsl_ip}:11434/v1")

        seen: set[str] = set()
        for base_url in candidates:
            if base_url in seen:
                continue
            seen.add(base_url)
            if cls._probe_ollama(base_url):
                _cached_base_url = base_url
                return base_url

        return None

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        return (
            base_url.replace("http://localhost:", "http://127.0.0.1:")
            .replace("https://localhost:", "https://127.0.0.1:")
            .rstrip("/")
        )

    @staticmethod
    def _ollama_root(base_url: str) -> str:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root

    @classmethod
    def _probe_ollama(cls, base_url: str, timeout_s: float = 3.0) -> bool:
        url = cls._ollama_root(base_url) + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _get_wsl_ip() -> str | None:
        try:
            result = subprocess.run(
                ["wsl", "hostname", "-I"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        ip = result.stdout.strip().split()
        return ip[0] if ip else None

    @staticmethod
    def _connection_help() -> str:
        return (
            "Ollama недоступна. Запустите WSL и Ollama: "
            "wsl → ollama serve (или ollama list). "
            "Проверка: http://127.0.0.1:11434/ в браузере."
        )

    @staticmethod
    def _format_error(openai_error: Exception, http_error: Exception, base_url: str) -> str:
        parts = [str(openai_error).strip(), str(http_error).strip()]
        details = " | ".join(part for part in parts if part)
        if "10061" in details or "Connection error" in details or "refused" in details.lower():
            return f"{AIAssistant._connection_help()} ({base_url})"
        return f"{details} ({base_url})"

    @staticmethod
    def _template_answer(
        question: str,
        scene: str,
        *,
        model: str = "заглушка",
        error: str | None = None,
    ) -> AIResponse:
        return AIResponse(
            content=(
                f"[Шаблон] Вопрос: «{question.strip()}»\n"
                f"Контекст сцены: {scene}\n"
                "Здесь будет ответ ИИ с подсказками и объяснениями."
            ),
            model=model,
            elapsed_s=0.0,
            is_stub=True,
            error=error,
        )

    @staticmethod
    def _build_prompt(question: str, scene: str) -> list[dict[str, str]]:
        system_msg = (
            "Ты ассистент для псевдо-AR интерфейса. "
            "Отвечай по контексту камеры: давай краткие подсказки, "
            "объясняй наблюдаемое и предлагай следующий шаг пользователю."
        )
        user_msg = (
            f"Вопрос пользователя: {question.strip()}\n\n"
            f"Что видит камера (детекции): {scene}\n\n"
            "Ответь по делу и, если уместно, предложи подсказки "
            "к тем объектам, что обнаружены."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _call_ollama_http(
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        timeout_s: float,
    ) -> str:
        url = base_url.rstrip("/") + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"] or ""
