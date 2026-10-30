@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Environment .venv not found. Check LocalTranscriber folder.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "gui_transcriber.py"
if errorlevel 1 (
    echo.
    echo App crashed. Copy the error output to ChatGPT.
    pause
)
