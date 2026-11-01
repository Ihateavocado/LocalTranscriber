"""Run faster-whisper in a separate Python process (CUDA resources exit with process).

Whisper Small on CUDA uses CTranslate2 and DLLs from the validated GPU
PyTorch environment in .venv-diar. CPU mode does not need these DLLs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def configure_cuda_dlls(root: Path):
    """Add Windows cuBLAS/cuDNN directory before importing faster-whisper.

    Do not close the returned os.add_dll_directory handle until process exit.
    """
    torch_lib = root / ".venv-diar" / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.is_dir():
        raise RuntimeError(f"CUDA DLL directory not found: {torch_lib}")
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    return os.add_dll_directory(str(torch_lib)) if os.name == "nt" else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Whisper worker (CPU / CUDA)")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", choices=("base", "small", "medium"), default="small")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glossary-file", type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input audio not found: {args.input}")
    glossary = args.glossary_file.read_text(encoding="utf-8").strip() if args.glossary_file else ""
    prompt = ("В записи могут встречаться следующие названия и термины: " + glossary + ".") if glossary else None

    dll_handle = configure_cuda_dlls(Path(__file__).resolve().parent) if args.device == "cuda" else None
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        args.model, device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
    )
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    parts, info = model.transcribe(
        str(args.input), language="ru", vad_filter=True,
        beam_size=5, initial_prompt=prompt, condition_on_previous_text=True,
    )
    # Consume the generator: transcribe() itself performs most inference lazily.
    segments = [
        {"start": float(part.start), "end": float(part.end), "text": part.text.strip()}
        for part in parts if part.text.strip()
    ]
    duration = float(info.duration or 0)
    processing_seconds = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "audio": args.input.name,
        "model": args.model,
        "device": args.device,
        "duration": duration,
        "model_load_seconds": round(load_seconds, 2),
        "processing_seconds": round(processing_seconds, 2),
        "segments": segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Whisper {args.model} {args.device}; load {load_seconds:.1f}s; "
          f"transcription {processing_seconds:.1f}s; segments {len(segments)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
