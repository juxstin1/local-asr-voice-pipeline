"""
Voice Assistant Pipeline
Mic -> faster-whisper ASR -> LM Studio (qwen3-4b) -> Qwen3-TTS -> Speaker

Usage:
    python voice.py              # press Enter to talk
    python voice.py --no-tts     # skip TTS, print only (fast testing)
"""

import sys
import time
import argparse
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import requests

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
SILENCE_THRESHOLD = 0.015     # RMS threshold (float32 audio, range -1..1)
SILENCE_DURATION = 1.5        # seconds of silence to stop recording
MAX_RECORD_SECONDS = 30       # safety cap

LM_STUDIO_URL = "http://localhost:1234"
LM_STUDIO_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max."

TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SPEAKER = "Ryan"
TTS_LANGUAGE = "English"


# ── LM Studio helpers ──────────────────────────────────────────────────────
previous_response_id = None


def ensure_model_loaded():
    """Load the LLM in LM Studio if no instance is running."""
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/api/v1/models", timeout=5)
        for m in resp.json().get("models", []):
            if m["key"] == LM_STUDIO_MODEL and m.get("loaded_instances"):
                print(f"  LLM already loaded: {LM_STUDIO_MODEL}")
                return
        print(f"  Loading {LM_STUDIO_MODEL} in LM Studio...")
        requests.post(
            f"{LM_STUDIO_URL}/api/v1/models/load",
            json={"model": LM_STUDIO_MODEL, "flash_attention": True, "context_length": 8192},
            timeout=30,
        )
        time.sleep(3)
        print("  LLM ready.")
    except requests.ConnectionError:
        print("ERROR: LM Studio not reachable at", LM_STUDIO_URL)
        sys.exit(1)


def chat(text: str) -> str:
    """Send text to LM Studio stateful chat, return response."""
    global previous_response_id

    # Embed system prompt in first message, stateful follow-ups auto-carry context
    if previous_response_id:
        prompt = f"/no_think {text}"
    else:
        prompt = f"/no_think [System: {SYSTEM_PROMPT}]\n\n{text}"

    payload = {"model": LM_STUDIO_MODEL, "input": prompt}
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    resp = requests.post(f"{LM_STUDIO_URL}/api/v1/chat", json=payload, timeout=120)
    data = resp.json()
    previous_response_id = data.get("response_id")

    # output is [{type: "reasoning"|"message", content: "..."}]
    for block in data.get("output", []):
        if block.get("type") == "message":
            return block.get("content", "").strip()
    return "(no response)"


# ── Audio I/O ───────────────────────────────────────────────────────────────
def record_until_silence() -> np.ndarray | None:
    """Record from mic until silence is detected after speech begins."""
    chunks = []
    silent_chunks = 0
    speech_detected = False
    max_silent = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SIZE) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(CHUNK_SIZE)
            rms = np.sqrt(np.mean(data ** 2))

            if rms > SILENCE_THRESHOLD:
                speech_detected = True
                silent_chunks = 0
            elif speech_detected:
                silent_chunks += 1

            if speech_detected:
                chunks.append(data.copy())

            if speech_detected and silent_chunks >= max_silent:
                break

    if not chunks:
        return None
    return np.concatenate(chunks).flatten()


def play_audio(wav: np.ndarray, sr: int):
    """Play a waveform through the default speaker."""
    sd.play(wav, sr)
    sd.wait()


# ── ASR ─────────────────────────────────────────────────────────────────────
def load_asr():
    print("  Loading ASR model (faster-whisper base.en)...")
    model = WhisperModel("base.en", device="cuda", compute_type="float16")
    print("  ASR ready.")
    return model


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    segments, _ = model.transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()


# ── TTS ─────────────────────────────────────────────────────────────────────
def load_tts():
    import torch
    from qwen_tts import Qwen3TTSModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tag = "GPU" if torch.cuda.is_available() else "CPU"

    print(f"  Loading TTS model ({tag})...")
    model = Qwen3TTSModel.from_pretrained(TTS_MODEL_ID, device_map=device, dtype=dtype)
    print("  TTS ready.")
    return model


def speak(model, text: str):
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=TTS_LANGUAGE,
        speaker=TTS_SPEAKER,
    )
    play_audio(wavs[0], sr)


# ── Main loop ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Voice Assistant Pipeline")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS (text output only)")
    args = parser.parse_args()

    print("\n=== Voice Assistant ===")
    print("Loading models...\n")

    asr_model = load_asr()
    ensure_model_loaded()

    tts_model = None
    if not args.no_tts:
        tts_model = load_tts()
    else:
        print("  TTS disabled (--no-tts)")

    print("\n--- Ready ---")
    print("Press Enter to speak. Ctrl+C to quit.\n")

    while True:
        input(">> ")
        print("   Listening...")

        audio = record_until_silence()
        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            print("   (no speech detected)\n")
            continue

        t0 = time.time()
        text = transcribe(asr_model, audio)
        asr_time = time.time() - t0

        if not text:
            print("   (empty transcription)\n")
            continue
        print(f"   You: {text}  ({asr_time:.1f}s)")

        t0 = time.time()
        response = chat(text)
        llm_time = time.time() - t0
        print(f"   Assistant: {response}  ({llm_time:.1f}s)")

        if tts_model:
            t0 = time.time()
            speak(tts_model, response)
            tts_time = time.time() - t0
            print(f"   (TTS: {tts_time:.1f}s)")

        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
