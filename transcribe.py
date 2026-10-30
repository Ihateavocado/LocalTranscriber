"""
Локальная транскрибация аудио и видео на CPU.
Файлы никуда не отправляются: обработка идёт только на этом компьютере.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv"}
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_NAME = "small"


def format_timestamp(seconds: float) -> str:
    """Превращает секунды в таймкод вида 00:01:23,456 для SRT."""
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Локально распознать речь в аудио- или видеофайле (русский язык)."
    )
    parser.add_argument(
        "input_file",
        help="Путь к файлу: MP3, WAV, M4A, MP4 или MKV",
    )
    return parser.parse_args()


def validate_input(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            "Проверьте путь. Если в нём есть пробелы, заключите путь в кавычки."
        )
    if not path.is_file():
        raise ValueError(f"Это не файл: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Файл пустой: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Формат {extension or '(без расширения)'} не поддерживается.\n"
            f"Допустимые форматы: {supported}"
        )
    return path.resolve()


def write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_srt(path: Path, segments: list[tuple[int, float, float, str]]) -> None:
    lines: list[str] = []
    for index, start, end, text in segments:
        lines.append(str(index))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def transcribe_file(input_path: Path) -> None:
    print("Этап 1/4: проверяю входной файл...")
    print(f"  Файл: {input_path}")
    print(f"  Размер: {input_path.stat().st_size / (1024 * 1024):.1f} МБ")

    print("Этап 2/4: загружаю модель Whisper (base, CPU, int8)...")
    print("  Первый запуск может занять время: модель скачивается один раз и сохраняется локально.")
    try:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
    except Exception as exc:
        raise RuntimeError(
            "Не удалось загрузить модель распознавания.\n"
            "Проверьте интернет при первом запуске и свободное место на диске."
        ) from exc

    print("Этап 3/4: распознаю речь (язык: русский, включён VAD)...")
    print("  На CPU это может занять несколько минут. Запись никуда не отправляется.")
    try:
        segment_iter, info = model.transcribe(
            str(input_path),
            language="ru",
            vad_filter=True,
        )
        print(f"  Определённая длительность аудио: {info.duration:.1f} сек.")
        collected: list[tuple[int, float, float, str]] = []
        for index, segment in enumerate(segment_iter, start=1):
            text = segment.text.strip()
            if not text:
                continue
            collected.append((index, segment.start, segment.end, text))
            print(f"  Фрагмент {index}: {format_timestamp(segment.start)} → {format_timestamp(segment.end)}")
    except Exception as exc:
        raise RuntimeError(
            "Не удалось обработать файл.\n"
            "Убедитесь, что файл не повреждён и в нём есть звуковая дорожка."
        ) from exc

    if not collected:
        raise RuntimeError(
            "Речь не найдена. Проверьте, что в файле есть русская речь и звук не слишком тихий."
        )

    print("Этап 4/4: сохраняю TXT и SRT...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    txt_path = OUTPUT_DIR / f"{stem}.txt"
    srt_path = OUTPUT_DIR / f"{stem}.srt"

    full_text = "\n".join(item[3] for item in collected)
    write_txt(txt_path, full_text + "\n")
    write_srt(srt_path, collected)

    print("Готово.")
    print(f"  TXT: {txt_path}")
    print(f"  SRT: {srt_path}")
    print(f"  Фрагментов с речью: {len(collected)}")


def main() -> int:
    args = parse_args()
    try:
        input_path = validate_input(args.input_file)
        transcribe_file(input_path)
        return 0
    except FileNotFoundError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        print("Полный traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        return 130
    except Exception:
        print("Непредвиденная ошибка. Полный traceback:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
