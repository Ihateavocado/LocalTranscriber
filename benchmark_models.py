"""
Сравнение моделей base и small на одном пятиминутном фрагменте.
Не изменяет transcribe.py. Записи никуда не отправляются.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
import wave
from pathlib import Path

import av
from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_SOURCE = Path(r"C:\Users\lenovo\Desktop\проект трансформации интервью 2.mkv")
DEFAULT_START_SEC = 10 * 60
DEFAULT_DURATION_SEC = 5 * 60
CLIP_PATH = OUTPUT_DIR / "benchmark_clip.wav"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "ru"
BEAM_SIZE = 5
SAMPLE_RATE = 16000


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def parse_clock(value: str) -> float:
    text = value.strip()
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Не удалось разобрать время: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сравнить base и small на одном локальном 5-минутном фрагменте."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_SOURCE),
        help="Путь к исходной записи",
    )
    parser.add_argument(
        "--start",
        default="10:00",
        help="Начало фрагмента, секунды или мм:сс",
    )
    parser.add_argument(
        "--duration",
        default="300",
        help="Длительность фрагмента в секундах",
    )
    return parser.parse_args()


def model_is_cached(name: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    folder = hub / f"models--Systran--faster-whisper-{name}"
    if not folder.exists():
        return False
    return any(folder.rglob("model.bin"))


def extract_clip(source: Path, dest: Path, start_sec: float, duration_sec: float) -> float:
    dest.parent.mkdir(parents=True, exist_ok=True)
    end_sec = start_sec + duration_sec
    raw = bytearray()

    container = av.open(str(source), mode="r", metadata_errors="ignore")
    try:
        if not container.streams.audio:
            raise RuntimeError("В файле нет звуковой дорожки.")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
        )
        seek_ts = int(start_sec * av.time_base)
        try:
            container.seek(seek_ts)
        except av.AVError:
            container.seek(int(start_sec / float(stream.time_base)), stream=stream)

        for frame in container.decode(audio=0):
            frame_time = float(frame.time) if frame.time is not None else None
            if frame_time is not None and frame_time + 0.05 < start_sec:
                continue
            if frame_time is not None and frame_time >= end_sec:
                break

            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, (list, tuple)):
                resampled = [resampled]
            for item in resampled:
                raw.extend(item.to_ndarray().tobytes())
    finally:
        container.close()

    target_bytes = int(duration_sec * SAMPLE_RATE) * 2
    if len(raw) > target_bytes:
        raw = raw[:target_bytes]
    if len(raw) < SAMPLE_RATE * 2:
        raise RuntimeError("Не удалось извлечь достаточно аудио. Проверьте начало и длительность фрагмента.")

    with wave.open(str(dest), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(bytes(raw))

    return len(raw) / 2 / SAMPLE_RATE


def write_timed_transcript(path: Path, segments: list[tuple[float, float, str]]) -> None:
    lines = [
        f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
        for start, end, text in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_model(name: str, clip_path: Path) -> dict:
    cached_before = model_is_cached(name)
    print(f"Модель {name}: {'уже в локальном кэше' if cached_before else 'будет скачана при загрузке'}")

    load_started = time.perf_counter()
    model = WhisperModel(name, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_seconds = time.perf_counter() - load_started

    print(f"  Загрузка модели: {load_seconds:.2f} с")
    print("  Распознавание (включая полный обход segments)...")

    transcribe_started = time.perf_counter()
    segment_iter, info = model.transcribe(
        str(clip_path),
        language=LANGUAGE,
        vad_filter=True,
        beam_size=BEAM_SIZE,
    )
    collected: list[tuple[float, float, str]] = []
    for segment in segment_iter:
        text = segment.text.strip()
        if not text:
            continue
        collected.append((segment.start, segment.end, text))
    transcribe_seconds = time.perf_counter() - transcribe_started

    audio_duration = float(info.duration)
    speedup = audio_duration / transcribe_seconds if transcribe_seconds > 0 else 0.0
    print(f"  Время распознавания: {transcribe_seconds:.2f} с")
    print(f"  Фрагментов с речью: {len(collected)}")

    del model
    gc.collect()

    return {
        "name": name,
        "audio_duration": audio_duration,
        "transcribe_seconds": transcribe_seconds,
        "load_seconds": load_seconds,
        "cached_before": cached_before,
        "speedup": speedup,
        "segments": collected,
    }


def print_table(results: list[dict]) -> None:
    print()
    print("Таблица сравнения")
    print("-" * 78)
    print(f"{'Модель':<10} {'Длительность аудио':>20} {'Время обработки':>18} {'Ускорение к realtime':>22}")
    print("-" * 78)
    for item in results:
        print(
            f"{item['name']:<10} "
            f"{item['audio_duration']:>17.1f} с "
            f"{item['transcribe_seconds']:>15.2f} с "
            f"{item['speedup']:>19.1f}x"
        )
    print("-" * 78)
    print("Ускорение = длительность аудио / время распознавания. Больше — быстрее.")
    print("Победителя по точности нет: эталонной расшифровки нет.")
    print()
    print("Время загрузки модели (скачивание, если его ещё не было, входит сюда):")
    for item in results:
        cache_note = "кэш был" if item["cached_before"] else "первое скачивание/загрузка"
        print(f"  {item['name']}: {item['load_seconds']:.2f} с ({cache_note})")


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser()
    start_sec = parse_clock(args.start)
    duration_sec = parse_clock(args.duration)

    print("Выбранный исходный файл:")
    print(f"  {source}")
    print(f"Интервал: {format_timestamp(start_sec)} → {format_timestamp(start_sec + duration_sec)}")
    print("Настройки обеих моделей: CPU, int8, ru, VAD=True, beam_size=5")
    print("Модели запускаются по очереди, чтобы не перегружать RAM.")
    print()

    if not source.exists():
        print(f"Ошибка: исходный файл не найден: {source}", file=sys.stderr)
        return 1

    print("Извлекаю один и тот же аудиофрагмент для обоих тестов...")
    clip_duration = extract_clip(source, CLIP_PATH, start_sec, duration_sec)
    print(f"  Клип: {CLIP_PATH}")
    print(f"  Фактически извлечено: {clip_duration:.1f} с")
    print()

    results: list[dict] = []
    for name in ("base", "small"):
        result = run_model(name, CLIP_PATH)
        out_path = OUTPUT_DIR / f"benchmark_{name}.txt"
        write_timed_transcript(out_path, result["segments"])
        result["transcript_path"] = out_path
        print(f"  Транскрипция: {out_path}")
        print()
        results.append(result)

    print_table(results)
    print("Файлы:")
    print(f"  Клип: {CLIP_PATH}")
    for item in results:
        print(f"  {item['name']}: {item['transcript_path']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        print("Непредвиденная ошибка. Полный traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
