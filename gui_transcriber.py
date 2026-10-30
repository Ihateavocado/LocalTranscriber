"""LocalTranscriber: простое локальное Windows-приложение на faster-whisper.

Запуск из папки проекта: .venv\\Scripts\\python.exe gui_transcriber.py
Не использует облачные API для передачи аудио.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "output"
MEDIA_TYPES = (
    ("Аудио и видео", "*.mkv *.mp4 *.mp3 *.wav *.m4a *.mov *.aac *.flac *.ogg *.webm *.wma"),
    ("Все файлы", "*.*"),
)
DEFAULT_TERMS = (
    "YouGile, CRM-система, 1С:УНФ, Моё дело, управленческий учёт, "
    "номенклатура, коммерческое предложение, КП, "
    "рентабельность, маржинальность, дебиторская задолженность"
)


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    h, rem = divmod(milliseconds, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_unique_stem(source: Path, model: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / f"{source.stem}_{model}_{stamp}"
    result = base
    suffix = 2
    while any(Path(str(result) + ext).exists() for ext in (".txt", ".srt", ".docx")):
        result = OUT_DIR / f"{base.name}_{suffix}"
        suffix += 1
    return result


def export_files(stem: Path, segments: list[tuple[float, float, str]], source: Path) -> list[Path]:
    txt = Path(str(stem) + ".txt")
    srt = Path(str(stem) + ".srt")
    txt_lines = [f"[{clock(start)} – {clock(end)}] {text}" for start, end, text in segments]
    txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8-sig")
    srt_lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        srt_lines.append(f"{i}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    srt.write_text("\n".join(srt_lines), encoding="utf-8-sig")

    paths = [txt, srt]
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "Для экспорта Word нужен python-docx. "
            "В терминале проекта выполните: .\\.venv\\Scripts\\python.exe -m pip install python-docx. "
            f"TXT и SRT уже сохранены: {txt}, {srt}"
        ) from exc

    doc = Document()
    doc.add_heading("Транскрипция встречи", level=1)
    doc.add_paragraph(f"Исходный файл: {source.name}")
    for start, end, text in segments:
        doc.add_paragraph(f"[{clock(start)} – {clock(end)}] {text}")
    docx = Path(str(stem) + ".docx")
    doc.save(docx)
    paths.append(docx)
    return paths


class TranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LocalTranscriber — локальная транскрибация")
        self.root.geometry("900x690")
        self.root.minsize(680, 560)
        self.jobs: queue.Queue = queue.Queue()
        self.busy = False
        self.model_name = None
        self.model_instance = None
        self.latest_outputs: list[Path] = []

        self.file_path = tk.StringVar()
        self.model_value = tk.StringVar(value="small")
        self.use_terms = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Готово к работе. Укажи файл и нажми «Распознать». ")
        self.progress_text = tk.StringVar(value="")
        self._build()
        self.root.after(120, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self):
        body = ttk.Frame(self.root, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="LocalTranscriber", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(body, text="Видео и аудио обрабатываются на этом компьютере, без платных API.").pack(anchor="w", pady=(1, 14))

        box = ttk.LabelFrame(body, text="Исходная запись", padding=10)
        box.pack(fill="x", pady=(0, 12))
        pathrow = ttk.Frame(box)
        pathrow.pack(fill="x")
        ttk.Entry(pathrow, textvariable=self.file_path).pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(pathrow, text="Выбрать файл…", command=self._browse)
        self.browse_button.pack(side="left", padx=(8, 0))
        ttk.Label(box, text="Можно выбрать большой MKV/MP4: файл читается с диска, без загрузки через браузер.", foreground="#60666d").pack(anchor="w", pady=(6, 0))

        settings = ttk.LabelFrame(body, text="Параметры распознавания", padding=10)
        settings.pack(fill="x", pady=(0, 12))
        row = ttk.Frame(settings)
        row.pack(fill="x")
        ttk.Label(row, text="Модель:").pack(side="left")
        self.model_combo = ttk.Combobox(row, textvariable=self.model_value, values=("base", "small", "medium"), state="readonly", width=12)
        self.model_combo.pack(side="left", padx=(8, 18))
        ttk.Label(row, text="Язык: русский  •  CPU/int8  •  Beam size: 5  •  VAD: вкл.").pack(side="left")
        self.terms_checkbox = ttk.Checkbutton(settings, text="Подсказывать специальные термины модели", variable=self.use_terms)
        self.terms_checkbox.pack(anchor="w", pady=(10, 4))
        self.terms = scrolledtext.ScrolledText(settings, height=3, wrap="word", font=("Segoe UI", 10))
        self.terms.insert("1.0", DEFAULT_TERMS)
        self.terms.pack(fill="x")
        ttk.Label(settings, text="Подсказка повышает шанс распознать термины, но может и ошибочно навязывать слова.", foreground="#60666d").pack(anchor="w", pady=(4, 0))

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=(0, 8))
        self.start_button = ttk.Button(actions, text="▶  Распознать", command=self._start)
        self.start_button.pack(side="left")
        self.folder_button = ttk.Button(actions, text="Открыть результаты", command=self._open_folder)
        self.folder_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, text="Сохранение: TXT + SRT + Word (DOCX)").pack(side="right")

        ttk.Label(body, textvariable=self.status, wraplength=820).pack(anchor="w", pady=(3, 3))
        self.progress = ttk.Progressbar(body, mode="indeterminate", length=450)
        self.progress.pack(fill="x", pady=(4, 4))
        ttk.Label(body, textvariable=self.progress_text, foreground="#60666d").pack(anchor="w", pady=(0, 10))

        result_frame = ttk.LabelFrame(body, text="Распознанный текст (предпросмотр)", padding=8)
        result_frame.pack(fill="both", expand=True)
        self.preview = scrolledtext.ScrolledText(result_frame, wrap="word", font=("Segoe UI", 10), state="disabled")
        self.preview.pack(fill="both", expand=True)

    def _browse(self):
        selected = filedialog.askopenfilename(title="Выберите видео или аудиофайл", filetypes=MEDIA_TYPES)
        if selected:
            self.file_path.set(selected)

    def _set_preview(self, text: str):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def _start(self):
        if self.busy:
            return
        path = Path(self.file_path.get().strip().strip('"'))
        if not path.is_file():
            messagebox.showwarning("Не выбран файл", "Выбери существующий файл с аудио или видео.")
            return
        model_name = self.model_value.get()
        use_terms = self.use_terms.get()
        glossary = self.terms.get("1.0", "end").strip() if use_terms else ""
        self.busy = True
        self.start_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.model_combo.configure(state="disabled")
        self.terms_checkbox.configure(state="disabled")
        self.terms.configure(state="disabled")
        self.status.set("Запуск распознавания…")
        self.progress_text.set("Первый запуск новой модели может потребовать загрузки её файлов из интернета.")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._set_preview("")
        worker = threading.Thread(target=self._transcribe, args=(path, model_name, glossary), daemon=True)
        worker.start()

    def _transcribe(self, path: Path, model_name: str, glossary: str):
        try:
            from faster_whisper import WhisperModel
            start_wall = time.perf_counter()
            self.jobs.put(("status", f"Загружаю модель {model_name} (локально)…"))
            if self.model_instance is None or self.model_name != model_name:
                self.model_instance = WhisperModel(model_name, device="cpu", compute_type="int8")
                self.model_name = model_name
            self.jobs.put(("status", "Распознаю речь. Это может занять несколько минут…"))
            prompt = ("В записи могут встречаться следующие названия и термины: " + glossary + ".") if glossary else None
            parts, info = self.model_instance.transcribe(
                str(path), language="ru", vad_filter=True,
                beam_size=5, initial_prompt=prompt,
                condition_on_previous_text=True,
            )
            duration = float(info.duration or 0)
            gathered = []
            for n, seg in enumerate(parts, 1):
                content = seg.text.strip()
                if content:
                    gathered.append((float(seg.start), float(seg.end), content))
                if n % 4 == 0:
                    at = clock(float(seg.end))
                    total = clock(duration) if duration > 0 else "неизвестна"
                    self.jobs.put(("position", f"Обработана речь примерно до {at} / {total} записи"))
            if not gathered:
                raise RuntimeError("Речь не обнаружена. Проверьте наличие звуковой дорожки и уровень громкости.")
            stem = get_unique_stem(path, model_name)
            saved = export_files(stem, gathered, path)
            elapsed = time.perf_counter() - start_wall
            lines = [f"[{clock(a)} – {clock(b)}] {c}" for a, b, c in gathered]
            self.jobs.put(("done", saved, "\n".join(lines), len(gathered), elapsed))
        except Exception:
            self.jobs.put(("error", traceback.format_exc()))

    def _poll(self):
        try:
            while True:
                message = self.jobs.get_nowait()
                kind = message[0]
                if kind == "status":
                    self.status.set(message[1])
                elif kind == "position":
                    self.progress_text.set(message[1])
                elif kind == "error":
                    self._finish()
                    self.status.set("Ошибка обработки. Подробности показаны ниже.")
                    self.progress_text.set("Проверь, что установлены faster-whisper, av==18.1.0 и python-docx.")
                    self._set_preview(message[1])
                    messagebox.showerror("Ошибка", "Не удалось обработать файл. Подробности показаны в окне программы.")
                elif kind == "done":
                    _, paths, transcript, count, elapsed = message
                    self.latest_outputs = paths
                    self._finish()
                    self.status.set(f"Готово! {count} сегментов, время обработки {elapsed:.1f} с.")
                    self.progress_text.set("Сохранено: " + ", ".join(p.name for p in paths))
                    self._set_preview(transcript)
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _finish(self):
        self.busy = False
        self.progress.stop()
        self.progress.configure(value=0)
        self.start_button.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.model_combo.configure(state="readonly")
        self.terms_checkbox.configure(state="normal")
        self.terms.configure(state="normal")

    def _open_folder(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(OUT_DIR))
        else:
            messagebox.showinfo("Папка с результатами", str(OUT_DIR))

    def _on_close(self):
        if self.busy and not messagebox.askyesno("Прервать работу?", "Идёт транскрибация. Если закрыть программу, текущий результат может не сохраниться. Закрыть?"):
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
