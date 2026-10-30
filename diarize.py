"""Local speaker diarization test using pyannote Community-1 (CPU).

Run with the project's .venv-diar Python environment.
Audio stays on the local computer; Hugging Face is used to download the model.
"""

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

# Set these BEFORE importing pyannote / huggingface_hub.
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments(annotation):
    return [
        {"start": round(turn.start, 3), "end": round(turn.end, 3), "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def main():
    parser = argparse.ArgumentParser(description="Local CPU diarization test")
    parser.add_argument("--input", default="output/benchmark_clip.wav")
    parser.add_argument("--num-speakers", type=int, default=2, help="0 = detect automatically")
    parser.add_argument("--output-prefix", default="output/diarization_test")
    parser.add_argument("--non-interactive", action="store_true", help="Read HF_TOKEN from environment or local HF login; do not prompt")
    args = parser.parse_args()

    audio_path = Path(args.input).resolve()
    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")
    if args.num_speakers < 0:
        parser.error("--num-speakers must be 0 or greater")

    # GUI supplies the token via the child process environment, never on its command line.
    # Cached HF login can also be used when no explicit token is supplied.
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and not args.non_interactive and sys.stdin.isatty():
        token = getpass.getpass("Hugging Face read token (hidden input; Enter to use cached login): ").strip()

    from pyannote.audio import Pipeline
    import torch

    print("Loading Community-1 model (first run downloads model files)...", flush=True)
    started = time.perf_counter()
    try:
        if token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", token=token
            )
        else:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    except Exception as error:
        print(f"Model loading failed: {type(error).__name__}")
        print("Check Hugging Face model access, token permissions and internet connection.")
        raise SystemExit(1) from None
    del token
    pipeline.to(torch.device("cpu"))
    load_seconds = time.perf_counter() - started
    print(f"Model ready: {load_seconds:.1f} s", flush=True)

    started = time.perf_counter()
    print(f"Analyzing: {audio_path.name}", flush=True)
    pipeline_args = {"num_speakers": args.num_speakers} if args.num_speakers else {}
    result = pipeline(str(audio_path), **pipeline_args)
    run_seconds = time.perf_counter() - started

    all_turns = segments(result.speaker_diarization)
    exclusive_turns = segments(result.exclusive_speaker_diarization)
    speaker_ids = sorted({item["speaker"] for item in all_turns})
    display_names = {speaker: f"Спикер {i + 1}" for i, speaker in enumerate(speaker_ids)}

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio": audio_path.name,
        "expected_speakers": args.num_speakers,
        "detected_speakers": len(speaker_ids),
        "model_load_seconds": round(load_seconds, 2),
        "processing_seconds": round(run_seconds, 2),
        "diarization": all_turns,
        "exclusive_diarization": exclusive_turns,
    }
    json_path = Path(str(prefix) + ".json")
    txt_path = Path(str(prefix) + ".txt")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Результат определения спикеров (без текста речи)\n")
        f.write(f"Аудио: {audio_path.name}\n")
        f.write(f"Время обработки: {run_seconds:.1f} с\n\n")
        for item in exclusive_turns:
            name = display_names.get(item["speaker"], item["speaker"])
            f.write(f"[{timestamp(item['start'])} - {timestamp(item['end'])}] {name}\n")

    print(f"Detected speakers: {len(speaker_ids)}")
    print(f"Intervals: {len(exclusive_turns)}")
    print(f"Model load: {load_seconds:.1f} s; diarization: {run_seconds:.1f} s")
    print(f"Saved: {txt_path} and {json_path}")


if __name__ == "__main__":
    main()
