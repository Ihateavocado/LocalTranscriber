r"""Локальный тест качества faster-whisper Small на одном 5-минутном фрагменте.

Запуск из папки LocalTranscriber:
    .\.venv\Scripts\python.exe quality_test.py

Использует уже созданный output/benchmark_clip.wav, не изменяет transcribe.py.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Ошибка: faster-whisper не установлен в этом окружении.")
    print(r"Запустите: .\.venv\Scripts\python.exe -m pip install faster-whisper")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
CLIP = OUTPUT / "benchmark_clip.wav"
MODEL_NAME = "small"
EXCERPT_START = 2 * 60 + 30  # 02:30 относительно benchmark_clip.wav
EXCERPT_END = 4 * 60 + 10    # 04:10 относительно benchmark_clip.wav

# Конфигурации отличаются только ОДНИМ параметром относительно базового теста.
EXPERIMENTS = [
    {
        "name": "terms",
        "description": "Подсказка с профессиональными терминами (initial_prompt)",
        "changes": {
            "initial_prompt": (
                "Термины разговора: CRM-система, YouGile, Моё дело, "
                "1С:УНФ, коммерческое предложение, КП, управленческий учёт, "
                "маржинальность, рентабельность, дебиторская задолженность, MVP."
            )
        },
    },
    {
        "name": "vad500",
        "description": "Более чувствительный VAD: min_silence_duration_ms=500",
        "changes": {"vad_parameters": {"min_silence_duration_ms": 500}},
    },
    {
        "name": "no_context",
        "description": "Без передачи распознанного текста в следующее окно",
        "changes": {"condition_on_previous_text": False},
    },
]


def stamp(seconds: float) -> str:
    """Время в формате ЧЧ:ММ:СС,ммм."""
    ms_total = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"


def format_segment(segment) -> str:
    return f"[{stamp(segment.start)} --> {stamp(segment.end)}] {segment.text.strip()}"


def report_section(name: str, description: str, segments: list, seconds: float) -> list[str]:
    overlapping = [
        s for s in segments
        if s.start < EXCERPT_END and s.end > EXCERPT_START
    ]
    longest = max((s.end - s.start for s in segments), default=0)
    lines = [
        f"=== {name}: {description} ===",
        f"Время обработки: {seconds:.2f} с",
        f"Число сегментов: {len(segments)}",
        f"Максимальная длина одного сегмента: {longest:.1f} с",
        f"Сегменты, пересекающие {stamp(EXCERPT_START)}–{stamp(EXCERPT_END)}:",
    ]
    if overlapping:
        lines.extend("  " + format_segment(s) for s in overlapping)
    else:
        lines.append("  (Не найдено сегментов речи на этом интервале.)")
    lines.append("")
    return lines


def main() -> int:
    print("LocalTranscriber — сравнительный тест качества модели Small")
    print(f"Исходный фрагмент: {CLIP}")
    if not CLIP.is_file():
        print("\nОШИБКА: Не найден benchmark_clip.wav")
        print(r"Скопируйте пятиминутный фрагмент в LocalTranscriber\output\benchmark_clip.wav")
        print("Ничего переустанавливать не нужно.")
        return 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Загружаю уже скачанную модель Small (CPU, int8)...")
    load_started = time.perf_counter()
    try:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    except Exception as exc:
        print(f"Не удалось загрузить модель: {exc}")
        return 1
    load_seconds = time.perf_counter() - load_started
    print(f"Модель готова за {load_seconds:.1f} с.")

    all_report = [
        "LocalTranscriber — эксперимент с настройками faster-whisper",
        f"Создан: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Аудио: {CLIP}",
        "Модель: small | CPU | int8 | язык ru | beam_size=5 | vad_filter=True",
        "Базовый текст для сравнения: output/benchmark_small.txt",
        "В каждом тесте меняется только один параметр относительно базы.",
        "ВНИМАНИЕ: без сверки с аудио качество нельзя считать измеренной точностью.",
        f"Время загрузки модели (отдельно): {load_seconds:.2f} с",
        "",
    ]

    for idx, experiment in enumerate(EXPERIMENTS, 1):
        name = experiment["name"]
        path = OUTPUT / f"quality_small_{name}.txt"
        print(f"\n[{idx}/{len(EXPERIMENTS)}] {experiment['description']}")
        kwargs = dict(
            language="ru",
            beam_size=5,
            vad_filter=True,
            **experiment["changes"],
        )
        try:
            t0 = time.perf_counter()
            generator, info = model.transcribe(str(CLIP), **kwargs)
            segments = list(generator)  # ВАЖНО: обработка идёт при чтении генератора
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            print(f"Ошибка в эксперименте '{name}': {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            return 1

        path.write_text(
            "\n".join(format_segment(s) for s in segments) + "\n",
            encoding="utf-8-sig",  # удобно открывать в Windows/Блокноте
        )
        duration = float(getattr(info, "duration", 0) or 0)
        speed = duration / elapsed if elapsed > 0 else 0
        print(f"Готово за {elapsed:.1f} с; аудио {duration:.1f} с; быстрее реального времени {speed:.1f}x")
        print(f"Файл: {path}")
        all_report.extend(report_section(name, experiment["description"], segments, elapsed))

    report_path = OUTPUT / "quality_report.txt"
    report_path.write_text("\n".join(all_report) + "\n", encoding="utf-8-sig")
    print("\nГотово. Отчёт о проблемном интервале:")
    print(report_path)
    print("\nДля сравнения пришлите 4 файла:")
    print("  output/benchmark_small.txt (исходный Small)")
    for experiment in EXPERIMENTS:
        print(f"  output/quality_small_{experiment['name']}.txt")
    print("При необходимости — output/quality_report.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
