@echo off
setlocal

echo.
echo ============================================================
echo   Voice Assistant — One-Click Setup
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo       Done.
echo.

:: ── Download faster-whisper ASR model ────────────────────────
echo [2/4] Downloading ASR model (faster-whisper base.en)...
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8'); print('       ASR model cached.')"
if %errorlevel% neq 0 (
    echo [ERROR] ASR model download failed.
    pause
    exit /b 1
)
echo.

:: ── Download Qwen3-TTS model ─────────────────────────────────
echo [3/4] Downloading TTS model (Qwen3-TTS-0.6B)...
echo       This is ~1.2 GB — may take a few minutes.
python -c "from transformers import AutoModel, AutoTokenizer; name='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'; print('       Downloading tokenizer...'); AutoTokenizer.from_pretrained(name); print('       Downloading model weights...'); AutoModel.from_pretrained(name); print('       TTS model cached.')"
if %errorlevel% neq 0 (
    echo [ERROR] TTS model download failed.
    echo       Check your internet connection and try again.
    pause
    exit /b 1
)
echo.

:: ── Verify LM Studio ────────────────────────────────────────
echo [4/4] Checking LM Studio connection...
python -c "import requests; r=requests.get('http://localhost:1234/api/v1/models',timeout=5); print('       LM Studio is running. Models:', len(r.json().get('models',[])))" 2>nul
if %errorlevel% neq 0 (
    echo       [WARN] LM Studio not detected at localhost:1234
    echo       Make sure LM Studio is running with qwen/qwen3-4b loaded.
    echo       Download it from: https://lmstudio.ai
)
echo.

echo ============================================================
echo   Setup complete!
echo.
echo   Run the assistant:
echo     python voice.py              (full pipeline)
echo     python voice.py --no-tts     (text only, no speech)
echo ============================================================
echo.
pause
