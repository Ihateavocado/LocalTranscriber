r"""Combine local Whisper SRT subtitles with local pyannote diarization JSON.

No model loading, network access or third-party Python dependencies.
Use the same audio timeline for both inputs (including any trim offset).

Example (PowerShell, from the LocalTranscriber folder):
    .\.venv\Scripts\python.exe .\merge_transcript.py --srt .\output\benchmark_clip.srt

The pyannote file defaults to output/diarization_test.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys


TIMECODE = r"\d{1,3}:\d{2}:\d{2}[,.]\d{1,3}"
RANGE_RE = re.compile(rf"^\s*({TIMECODE})\s*-->\s*({TIMECODE})(?:\s+.*)?$")


@dataclass(frozen=True)
class Subtitle:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


def parse_time(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def format_time(value: float) -> str:
    milliseconds = round(max(0, value) * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def read_srt(path: Path) -> list[Subtitle]:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Some older Windows subtitle exports use the local Cyrillic code page.
        content = path.read_text(encoding="cp1251")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    subtitles: list[Subtitle] = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            match = RANGE_RE.match(line)
            if not match:
                continue
            start, end = parse_time(match[1]), parse_time(match[2])
            text = " ".join(part.strip() for part in lines[i + 1 :] if part.strip())
            if end > start and text:
                subtitles.append(Subtitle(start, end, text))
            break
    subtitles.sort(key=lambda s: (s.start, s.end))
    if not subtitles:
        raise ValueError(f"No valid timed subtitles found in {path.name}")
    return subtitles


def read_diarization(path: Path) -> tuple[list[Turn], dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Diarization JSON must be an object")
    source_key = "exclusive_diarization" if payload.get("exclusive_diarization") else "diarization"
    raw = payload.get(source_key)
    if not isinstance(raw, list) or not raw:
        raise ValueError("No diarization intervals found in JSON")
    turns: list[Turn] = []
    for i, item in enumerate(raw):
        try:
            start, end, speaker = float(item["start"]), float(item["end"]), str(item["speaker"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid diarization interval at index {i}") from exc
        if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end) or not speaker.strip():
            raise ValueError(f"Invalid diarization interval at index {i}")
        turns.append(Turn(start, end, speaker))
    turns.sort(key=lambda t: (t.start, t.end))
    return turns, payload, source_key


def speaker_for(subtitle: Subtitle, turns: list[Turn], dominance_threshold: float) -> tuple[str | None, str, float, float]:
    """Pick the speaker with most time overlap; don't invent word-level alignment."""
    weights: dict[str, float] = defaultdict(float)
    for turn in turns:
        if turn.start >= subtitle.end:
            break
        overlap = min(subtitle.end, turn.end) - max(subtitle.start, turn.start)
        if overlap > 0:
            weights[turn.speaker] += overlap

    total = sum(weights.values())
    duration = subtitle.end - subtitle.start
    if total < 0.12 or total / duration < 0.15:
        return None, "unassigned", min(1.0, total / duration), 0.0
    winner = max(weights, key=weights.get)
    dominance = weights[winner] / total
    if dominance < dominance_threshold:
        return None, "mixed", min(1.0, total / duration), dominance
    return winner, "assigned", min(1.0, total / duration), dominance


def merge(subtitles: list[Subtitle], turns: list[Turn], threshold: float) -> list[dict]:
    ids = sorted({t.speaker for t in turns})
    display = {speaker: f"Спикер {i + 1}" for i, speaker in enumerate(ids)}
    merged: list[dict] = []
    for subtitle in subtitles:
        speaker_id, status, coverage, dominance = speaker_for(subtitle, turns, threshold)
        label = display[speaker_id] if speaker_id else (
            "Спикер неоднозначен" if status == "mixed" else "Спикер не определён"
        )
        merged.append({
            "start": round(subtitle.start, 3),
            "end": round(subtitle.end, 3),
            "speaker_id": speaker_id,
            "speaker": label,
            "status": status,
            "coverage_ratio": round(coverage, 3),
            "dominance_ratio": round(dominance, 3),
            "text": subtitle.text,
        })
    return merged


def save_files(prefix: Path, merged: list[dict], metadata: dict) -> list[Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    txt = Path(str(prefix) + ".txt")
    srt = Path(str(prefix) + ".srt")
    js = Path(str(prefix) + ".json")
    txt.write_text(
        "\n".join(
            f"[{format_time(s['start'])} - {format_time(s['end'])}] {s['speaker']}: {s['text']}"
            for s in merged
        ) + "\n", encoding="utf-8-sig"
    )
    srt.write_text(
        "\n".join(
            f"{i}\n{format_time(s['start'])} --> {format_time(s['end'])}\n{s['speaker']}: {s['text']}\n"
            for i, s in enumerate(merged, 1)
        ) + "\n", encoding="utf-8-sig"
    )
    js.write_text(json.dumps({**metadata, "segments": merged}, ensure_ascii=False, indent=2), encoding="utf-8")
    return [txt, srt, js]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Whisper SRT with pyannote speaker intervals (offline)")
    parser.add_argument("--srt", required=True, type=Path, help="Whisper SRT from the SAME audio/timeline")
    parser.add_argument("--diar", type=Path, default=Path("output/diarization_test.json"))
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--dominance", type=float, default=0.60, help="Min fraction of voiced overlap for one speaker (default 0.60)")
    args = parser.parse_args()
    if not 0.5 <= args.dominance <= 1.0:
        parser.error("--dominance must be between 0.5 and 1.0")
    for path in (args.srt, args.diar):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    try:
        subtitles = read_srt(args.srt)
        turns, diar_meta, source_key = read_diarization(args.diar)
        merged = merge(subtitles, turns, args.dominance)
        prefix = args.output_prefix or Path("output") / f"{args.srt.stem}_speakers"
        metadata = {
            "source_srt": str(args.srt),
            "source_diarization": str(args.diar),
            "diarization_audio": diar_meta.get("audio"),
            "diarization_field": source_key,
            "method": "speaker assigned by maximum time overlap with each SRT cue",
            "warning": "SRT cues crossing speaker changes cannot be split reliably without word-level timestamps",
        }
        paths = save_files(prefix, merged, metadata)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    counts = Counter(m["status"] for m in merged)
    print(f"SRT subtitles: {len(subtitles)}; diarization turns: {len(turns)} ({source_key})")
    print(f"Assigned: {counts['assigned']}; mixed: {counts['mixed']}; unassigned: {counts['unassigned']}")
    print("Saved:", ", ".join(map(str, paths)))
    diar_end = max(t.end for t in turns)
    srt_end = max(s.end for s in subtitles)
    if srt_end > diar_end + 60:
        print(
            "WARNING: SRT continues far beyond the diarization intervals. "
            "Use subtitles from the same 5-minute audio, or expect unlabeled later sections."
        )
    print("Review speaker labels manually: diarization does not establish people's identities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
