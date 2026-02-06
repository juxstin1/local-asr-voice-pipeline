# Local ASR Voice Pipeline

A self-hosted Siri/Alexa that runs entirely on your machine. No cloud, no API keys, no data leaving your hardware.

Three AI models wired into a real-time voice-to-voice loop — speech recognition, LLM reasoning, and neural text-to-speech — all running locally on consumer GPU hardware.

**Mic &rarr; Speech Recognition &rarr; LLM Reasoning &rarr; Text-to-Speech &rarr; Speaker**

## Architecture

```
┌─────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────┐
│   Mic   │────▸│  faster-whisper   │────▸│   LM Studio      │────▸│   Qwen3-TTS      │────▸│ Speaker │
│  Input  │     │  (ASR, base.en)  │     │  (qwen3-4b LLM)  │     │  (0.6B, 12Hz)    │     │ Output  │
└─────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘     └─────────┘
                   ~0.3s CPU                 ~1-2s GPU               ~1-2s GPU
```

### Pipeline Stages

| Stage | Model | Size | Runtime | Role |
|-------|-------|------|---------|------|
| **ASR** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `base.en` | 148 MB | CPU (int8) | Real-time speech-to-text via CTranslate2 |
| **LLM** | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 2.6 GB | GPU via LM Studio | Stateful multi-turn conversation |
| **TTS** | [Qwen3-TTS-0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 1.2 GB | GPU/CPU | Neural text-to-speech synthesis |

## Key Features

- **100% Local** — All inference runs on your machine. Zero network calls to external APIs.
- **Stateful Conversation** — Multi-turn context via LM Studio's stateful chat API with KV cache reuse.
- **Voice Activity Detection** — Automatic silence detection starts/stops recording without manual triggers.
- **Modular Design** — Each pipeline stage is independent. Swap models, skip TTS, or extend freely.
- **Fast Startup** — Skips redundant model loading. Detects already-loaded LM Studio instances.

## Requirements

- **Python** 3.11+
- **LM Studio** 0.4+ running locally ([download](https://lmstudio.ai))
- **OS** Windows 10/11 (tested), Linux (supported)
- **GPU** Optional but recommended for TTS. LLM runs through LM Studio's own GPU management.
- **Microphone** Any standard input device

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/juxstin1/local-asr-voice-pipeline.git
cd local-asr-voice-pipeline
```

**Windows (one-click):**
```bash
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh && ./setup.sh
```

The setup script installs all Python dependencies and downloads the ASR and TTS models (~1.4 GB total).

### 2. Start LM Studio

Open LM Studio and load `qwen/qwen3-4b` (or any compatible model). The assistant auto-detects and loads the model if LM Studio is running.

### 3. Run

```bash
# Full pipeline (voice in, voice out)
python voice.py

# Text-only mode (skip TTS for fast iteration)
python voice.py --no-tts
```

Press **Enter** to start recording. Speak naturally — recording stops automatically after 1.5 seconds of silence. Press **Ctrl+C** to quit.

## Configuration

All configuration lives at the top of `voice.py`:

```python
# Audio capture
SILENCE_THRESHOLD = 0.015     # RMS threshold for voice activity detection
SILENCE_DURATION  = 1.5       # Seconds of silence before stopping
MAX_RECORD_SECONDS = 30       # Safety cap on recording length

# LLM
LM_STUDIO_URL   = "http://localhost:1234"
LM_STUDIO_MODEL = "qwen/qwen3-4b"
SYSTEM_PROMPT    = "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max."

# TTS
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SPEAKER  = "Ryan"
TTS_LANGUAGE = "English"
```

## How It Works

### Speech Recognition (ASR)
Records audio from the microphone in 1024-sample chunks, computing RMS energy per chunk. Once speech crosses the threshold, audio is buffered until 1.5s of consecutive silence. The buffer is fed to faster-whisper's CTranslate2 engine for int8-quantized inference on CPU.

### LLM Conversation
Uses LM Studio's stateful `/api/v1/chat` endpoint. The first message embeds the system prompt. Subsequent turns pass `previous_response_id` to maintain conversation context with full KV cache reuse — no re-processing of history on each turn. The `/no_think` prefix disables chain-of-thought for faster responses.

### Text-to-Speech
Qwen3-TTS generates speech waveforms from the LLM response using a custom voice profile. Output plays through the system's default audio device via sounddevice.

## Project Structure

```
local-asr-voice-pipeline/
├── voice.py           # Complete pipeline — ASR, LLM, TTS, audio I/O
├── setup.bat          # Windows one-click installer (deps + models)
├── setup.sh           # Linux/macOS installer
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT
└── README.md
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `LM Studio not reachable` | Start LM Studio and ensure the server is running on port 1234 |
| `No speech detected` | Check your mic input device. Lower `SILENCE_THRESHOLD` for quieter environments |
| Recording never stops | Raise `SILENCE_THRESHOLD` — background noise is above the current threshold |
| Slow LLM responses | Ensure only one model instance is loaded in LM Studio (check loaded_instances) |
| TTS import error | Run `pip install qwen-tts` — requires transformers and torchaudio |

## License

MIT
