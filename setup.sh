#!/usr/bin/env bash
set -e

echo ""
echo "============================================================"
echo "  Voice Assistant — One-Click Setup"
echo "============================================================"
echo ""

# ── Check Python ─────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Install Python 3.11+"
    exit 1
fi

echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

# ── Download faster-whisper ASR model ────────────────────────
echo "[2/4] Downloading ASR model (faster-whisper base.en)..."
python3 -c "
from faster_whisper import WhisperModel
WhisperModel('base.en', device='cpu', compute_type='int8')
print('      ASR model cached.')
"
echo ""

# ── Download Qwen3-TTS model ─────────────────────────────────
echo "[3/4] Downloading TTS model (Qwen3-TTS-0.6B)..."
echo "      This is ~1.2 GB — may take a few minutes."
python3 -c "
from transformers import AutoModel, AutoTokenizer
name = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
print('      Downloading tokenizer...')
AutoTokenizer.from_pretrained(name)
print('      Downloading model weights...')
AutoModel.from_pretrained(name)
print('      TTS model cached.')
"
echo ""

# ── Verify LM Studio ────────────────────────────────────────
echo "[4/4] Checking LM Studio connection..."
python3 -c "
import requests
try:
    r = requests.get('http://localhost:1234/api/v1/models', timeout=5)
    count = len(r.json().get('models', []))
    print(f'      LM Studio is running. Models: {count}')
except Exception:
    print('      [WARN] LM Studio not detected at localhost:1234')
    print('      Make sure LM Studio is running with qwen/qwen3-4b loaded.')
    print('      Download it from: https://lmstudio.ai')
"
echo ""

echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Run the assistant:"
echo "    python voice.py              (full pipeline)"
echo "    python voice.py --no-tts     (text only, no speech)"
echo "============================================================"
