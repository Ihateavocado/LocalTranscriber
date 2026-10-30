@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Local .venv Python not found. Place these files in the LocalTranscriber project folder.
  pause
  exit /b 1
)
echo Installing Word export support into the existing LocalTranscriber environment...
".venv\Scripts\python.exe" -m pip install "python-docx>=1.1,<2"
if errorlevel 1 (
  echo ERROR: The package installation failed. Copy the output and send it to ChatGPT.
  pause
  exit /b 1
)
echo.
echo Ready. Double-click start_transcriber.bat to open LocalTranscriber.
pause
