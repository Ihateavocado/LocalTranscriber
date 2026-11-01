# LocalTranscriber GPU Edition — установка на втором ноутбуке

Сборка предназначена для уже протестированного ноутбука Windows с NVIDIA RTX 3050 Ti 4 ГБ.
Здесь **исходный код**, а не готовый EXE. Запускайте его через Python 3.12 в `.venv`.

## Что сделано

- Whisper Small: выбор `GPU (CUDA)` или `CPU`; CUDA: `int8_float16`, CPU: `int8`.
- pyannote Community-1: независимый выбор GPU или CPU и числа спикеров.
- GPU-модели выполняются **последовательно в отдельных процессах**. После Whisper процесс завершается и освобождает CUDA-контекст перед pyannote.
- FFmpeg 7 Shared, проверенный с TorchCodec 0.9.1, подключается автоматически из папки `tools/ffmpeg7/ffmpeg-7.1.1-full_build-shared/bin`.
- Сохранены словарь терминов, TXT, SRT, Word, разметка говорящих и отчёт JSON.
- В заголовке состояния показывается **общее** время, а в сообщении — время работы нейросетевых этапов.
- Основной исходный код и данные не загружаются на сторонние API. При первом использовании модель может скачиваться из Hugging Face. После скачивания можно обрабатывать локально.

## Установка изменений

1. Закройте окно приложения, если оно запущено.
2. Откройте PowerShell и выполните:

   ```powershell
   cd "$HOME\LocalTranscriber"
   git branch --show-current
   ```

   Убедитесь, что ветка — `feature/gpu-acceleration`.
3. Распакуйте архив **прямо в папку `LocalTranscriber`** с заменой файлов `gui_transcriber.py`, `diarize.py`, `merge_transcript.py`. Также добавятся `whisper_worker.py`, `start_gpu.cmd` и эта инструкция. Например, если архив в «Загрузках»:

   ```powershell
   Expand-Archive -LiteralPath "$HOME\Downloads\LocalTranscriber_GPU_patch.zip" -DestinationPath "$HOME\LocalTranscriber" -Force
   ```

4. Уже настроенные `.venv`, `.venv-diar`, модели и `tools/ffmpeg7` **не удаляйте**. Они не входят в архив.
5. Для Word-экспорта установите пакет в Whisper-окружение (если ранее не ставили):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install "python-docx==1.2.0"
   ```

6. Запустите:

   ```powershell
   .\.venv\Scripts\python.exe .\gui_transcriber.py
   ```

   Или дважды щёлкните `start_gpu.cmd` в проводнике.

## Первый тест

- Выбрать `benchmark_clip.wav` (тот же 5-минутный файл, который тестировали).
- Модель: `small`; Whisper: `GPU (CUDA)`.
- Включить «Определять спикеров», участников: `2`, устройство: `GPU (CUDA)`.
- При запросе токена Hugging Face нажать OK с пустым полем, если вход уже выполнен через `hf auth login`.
- Нажать «Распознать».
- Открыть папку `output` и проверить файл с суффиксом `_speakers.docx`.

**Ожидания:** в отдельном тесте Whisper обрабатывал 300 с аудио за 8,0 с (без загрузки модели), pyannote — за 59,4 с. Полный GUI-проход будет дольше из-за запуска двух процессов, загрузки моделей, FFmpeg и экспорта. Его фактическую скорость ещё нужно измерить на ноутбуке.

## Если что-то пошло не так

- `No module named docx`: выполнить шаг 5.
- `FFmpeg 7 не найден`: убедиться в наличии `tools\ffmpeg7\ffmpeg-7.1.1-full_build-shared\bin\ffmpeg.exe`.
- `Could not load libtorchcodec`: проверить, что папка выше содержит `avutil-59.dll`, а в `.venv-diar` установлены Torch 2.9.1+cu126 и TorchCodec 0.9.1.
- `cublas64_12.dll` / `cudnn_ops64_9.dll` не найдены: проверить, что `.venv-diar\Lib\site-packages\torch\lib` существует; Whisper GPU ищет DLL именно там.
- `CUDA out of memory`: выбрать `CPU` для одного из этапов и повторить; модель `medium` при 4 ГБ VRAM может не поместиться.
- «Ошибка»: текст подробностей выводится в поле предпросмотра приложения. Если Whisper отработал, но pyannote дал ошибку, обычные TXT/SRT/DOCX сохраняются.
- Метки спикеров и текст обязательно проверить по аудио; длинный сегмент Whisper, где говорящие сменяются, может ошибочно получить метку одного спикера.

## После успешного теста

   ```powershell
   git status --short
   git add gui_transcriber.py diarize.py merge_transcript.py whisper_worker.py start_gpu.cmd GPU_SETUP.md
   git commit -m "Add sequential GPU Whisper and pyannote support"
   git push -u origin feature/gpu-acceleration
   ```

**Важно:** ни `.venv`, ни `.venv-diar`, ни файлы с записями, модели или Hugging Face token в репозиторий добавлять не нужно.
