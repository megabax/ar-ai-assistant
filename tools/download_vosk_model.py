"""Скачивание лёгкой русской модели Vosk для слабого CPU."""

from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "vosk-model-small-ru-0.22"
MODEL_ZIP = MODELS_DIR / f"{MODEL_NAME}.zip"
MODEL_DIR = MODELS_DIR / MODEL_NAME
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
MARKER_FILE = MODEL_DIR / "am" / "final.mdl"


def _log(message: str) -> None:
    print(message, flush=True)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _download(url: str, destination: Path) -> None:
    _log(f"[1/3] Скачивание {MODEL_NAME}")
    _log(f"      URL: {url}")
    _log(f"      Файл: {destination}")
    _log("      Это может занять несколько минут — ждите обновления прогресса…")

    started = time.perf_counter()
    last_report = -1

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        nonlocal last_report
        downloaded = block_num * block_size
        elapsed = time.perf_counter() - started

        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            if percent >= last_report + 5 or percent == 100:
                _log(
                    f"      … {percent:3d}% "
                    f"({_format_size(downloaded)} / {_format_size(total_size)}) "
                    f"· {elapsed:.1f} с"
                )
                last_report = percent
        elif block_num % 50 == 0:
            _log(f"      … {_format_size(downloaded)} скачано · {elapsed:.1f} с")

    urlretrieve(url, destination, reporthook=report_progress)
    elapsed = time.perf_counter() - started
    _log(
        f"      Скачивание завершено за {elapsed:.1f} с, "
        f"размер: {_format_size(destination.stat().st_size)}"
    )


def _extract(archive_path: Path, destination_dir: Path) -> None:
    _log(f"[2/3] Распаковка {archive_path.name}")
    started = time.perf_counter()

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        total = len(names)
        _log(f"      Файлов в архиве: {total}")

        step = max(1, total // 20)
        for index, name in enumerate(names, start=1):
            archive.extract(name, destination_dir)
            if index == 1 or index == total or index % step == 0:
                percent = index * 100 // total
                elapsed = time.perf_counter() - started
                _log(f"      … {percent:3d}% ({index}/{total}) · {elapsed:.1f} с")

    elapsed = time.perf_counter() - started
    _log(f"      Распаковка завершена за {elapsed:.1f} с")


def _ensure_model_dir() -> None:
    if MODEL_DIR.exists():
        return

    nested = next(MODELS_DIR.glob(f"{MODEL_NAME}*/am/final.mdl"), None)
    if nested is None:
        return

    found_dir = nested.parent.parent
    _log(f"[2/3] Перенос {found_dir.name} -> {MODEL_DIR.name}")
    shutil.move(str(found_dir), str(MODEL_DIR))


def main() -> None:
    _log("=== Загрузка модели Vosk ===")
    _log(f"Папка models: {MODELS_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MARKER_FILE.exists():
        _log(f"Модель уже установлена: {MODEL_DIR}")
        _log(f"Проверка: {MARKER_FILE} OK")
        return

    if MODEL_ZIP.exists():
        _log(f"Найден архив: {MODEL_ZIP} ({_format_size(MODEL_ZIP.stat().st_size)})")
    else:
        _download(MODEL_URL, MODEL_ZIP)

    _extract(MODEL_ZIP, MODELS_DIR)
    _ensure_model_dir()

    _log("[3/3] Проверка установки")
    if not MARKER_FILE.exists():
        _log(f"ОШИБКА: не найден файл модели: {MARKER_FILE}")
        sys.exit(1)

    _log(f"Готово: {MODEL_DIR}")
    _log(f"Маркер: {MARKER_FILE} OK")


if __name__ == "__main__":
    main()
