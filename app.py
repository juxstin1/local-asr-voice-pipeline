"""
Web UI Voice Assistant
FastAPI backend — WebSocket for streaming LLM, ASR, and TTS.

Usage:
    python app.py                  # start web server on port 8000
    python app.py --no-tts         # skip TTS model loading
    python app.py --port 9000      # custom port
"""

import asyncio
import argparse
import base64
import io
import json
import struct
import time
import wave
from contextlib import asynccontextmanager

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
LM_STUDIO_URL = "http://localhost:1234"
LM_STUDIO_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise — 1-3 sentences max."

TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_SPEAKER = "Ryan"
TTS_LANGUAGE = "English"

# ── Global state ────────────────────────────────────────────────────────────
asr_model = None
tts_model = None
gpu_lock = asyncio.Lock()  # serializes ASR + TTS on the same GPU
http_client: httpx.AsyncClient = None
cli_args = None


# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, tts_model, http_client

    print("\n=== Web Voice Assistant ===")
    print("Loading models...\n")

    # ASR
    from faster_whisper import WhisperModel
    print("  Loading ASR model (faster-whisper base.en, CUDA fp16)...")
    asr_model = WhisperModel("base.en", device="cuda", compute_type="float16")
    print("  ASR ready.")

    # LM Studio check
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(f"{LM_STUDIO_URL}/api/v1/models")
            models = resp.json().get("models", [])
            loaded = any(
                m["key"] == LM_STUDIO_MODEL and m.get("loaded_instances")
                for m in models
            )
            if loaded:
                print(f"  LLM already loaded: {LM_STUDIO_MODEL}")
            else:
                print(f"  Loading {LM_STUDIO_MODEL} in LM Studio...")
                await client.post(
                    f"{LM_STUDIO_URL}/api/v1/models/load",
                    json={
                        "model": LM_STUDIO_MODEL,
                        "flash_attention": True,
                        "context_length": 8192,
                    },
                    timeout=60,
                )
                await asyncio.sleep(3)
                print("  LLM ready.")
        except httpx.ConnectError:
            print(f"  WARNING: LM Studio not reachable at {LM_STUDIO_URL}")
            print("  Start LM Studio before sending chat messages.")

    # TTS (optional)
    if not cli_args.no_tts:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            tag = "GPU" if torch.cuda.is_available() else "CPU"

            print(f"  Loading TTS model ({tag})...")
            tts_model = Qwen3TTSModel.from_pretrained(
                TTS_MODEL_ID, device_map=device, dtype=dtype
            )
            print("  TTS ready.")
        except Exception as e:
            print(f"  TTS unavailable: {e}")
            tts_model = None
    else:
        print("  TTS disabled (--no-tts)")

    # Persistent HTTP client for LM Studio streaming
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    print(f"\n--- Ready at http://localhost:{cli_args.port} ---\n")

    yield

    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


# ── ASR helper ──────────────────────────────────────────────────────────────
def _transcribe_sync(pcm_bytes: bytes) -> tuple[str, float]:
    """Run faster-whisper on raw Int16 PCM bytes. Returns (text, seconds)."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) < SAMPLE_RATE * 0.3:
        return "", 0.0
    t0 = time.time()
    segments, _ = asr_model.transcribe(samples, language="en")
    text = " ".join(s.text for s in segments).strip()
    return text, time.time() - t0


async def transcribe_audio(pcm_bytes: bytes) -> tuple[str, float]:
    async with gpu_lock:
        return await asyncio.to_thread(_transcribe_sync, pcm_bytes)


# ── TTS helper ──────────────────────────────────────────────────────────────
def _generate_tts_sync(text: str) -> tuple[bytes, float]:
    """Run Qwen3-TTS, return (wav_bytes, seconds)."""
    t0 = time.time()
    wavs, sr = tts_model.generate_custom_voice(
        text=text, language=TTS_LANGUAGE, speaker=TTS_SPEAKER
    )
    audio = wavs[0]
    # Convert to 16-bit PCM WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (audio * 32767).astype(np.int16).tobytes()
        wf.writeframes(pcm)
    return buf.getvalue(), time.time() - t0


async def generate_tts(text: str) -> tuple[bytes, float]:
    async with gpu_lock:
        return await asyncio.to_thread(_generate_tts_sync, text)


# ── LLM streaming ──────────────────────────────────────────────────────────
async def chat_stream(
    ws: WebSocket,
    text: str,
    previous_response_id: str | None,
) -> tuple[str, str | None, float]:
    """
    Stream LLM response from LM Studio SSE endpoint.
    Sends llm_token messages via WebSocket as tokens arrive.
    Returns (full_text, new_response_id, elapsed_seconds).
    """
    if previous_response_id:
        prompt = f"/no_think {text}"
    else:
        prompt = f"/no_think [System: {SYSTEM_PROMPT}]\n\n{text}"

    payload = {"model": LM_STUDIO_MODEL, "input": prompt, "stream": True}
    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    t0 = time.time()
    full_text = ""
    new_response_id = previous_response_id

    try:
        async with http_client.stream(
            "POST", f"{LM_STUDIO_URL}/api/v1/chat", json=payload
        ) as resp:
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "message.delta":
                        token = event.get("content", "")
                        if token:
                            full_text += token
                            await ws.send_json({"type": "llm_token", "token": token})

                    elif etype == "chat.end":
                        result = event.get("result", {})
                        new_response_id = result.get("response_id", new_response_id)

    except httpx.ConnectError:
        await ws.send_json({"type": "error", "message": "LM Studio not reachable"})
        return "", previous_response_id, 0.0
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"LLM error: {e}"})
        return "", previous_response_id, 0.0

    elapsed = time.time() - t0
    return full_text.strip(), new_response_id, elapsed


# ── WebSocket handler ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    previous_response_id = None
    tts_enabled = tts_model is not None

    await ws.send_json({
        "type": "status",
        "message": "Connected",
        "tts_available": tts_model is not None,
    })

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            # ── Text input ──────────────────────────────────────────
            if msg_type == "text":
                query = msg.get("text", "").strip()
                if not query:
                    continue

                await ws.send_json({"type": "status", "message": "Thinking..."})
                full_text, previous_response_id, llm_time = await chat_stream(
                    ws, query, previous_response_id
                )
                await ws.send_json({
                    "type": "llm_done",
                    "text": full_text,
                    "time": round(llm_time, 2),
                })

                # TTS
                if tts_enabled and tts_model and full_text:
                    await ws.send_json({"type": "status", "message": "Generating speech..."})
                    try:
                        wav_bytes, tts_time = await generate_tts(full_text)
                        audio_b64 = base64.b64encode(wav_bytes).decode()
                        await ws.send_json({
                            "type": "tts_audio",
                            "audio": audio_b64,
                            "time": round(tts_time, 2),
                        })
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "message": f"TTS error: {e}",
                        })

            # ── Audio input ─────────────────────────────────────────
            elif msg_type == "audio":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    continue

                pcm_bytes = base64.b64decode(audio_b64)

                await ws.send_json({"type": "status", "message": "Transcribing..."})
                text, asr_time = await transcribe_audio(pcm_bytes)

                if not text:
                    await ws.send_json({
                        "type": "error",
                        "message": "No speech detected",
                    })
                    continue

                await ws.send_json({
                    "type": "transcription",
                    "text": text,
                    "time": round(asr_time, 2),
                })

                await ws.send_json({"type": "status", "message": "Thinking..."})
                full_text, previous_response_id, llm_time = await chat_stream(
                    ws, text, previous_response_id
                )
                await ws.send_json({
                    "type": "llm_done",
                    "text": full_text,
                    "time": round(llm_time, 2),
                })

                # TTS
                if tts_enabled and tts_model and full_text:
                    await ws.send_json({"type": "status", "message": "Generating speech..."})
                    try:
                        wav_bytes, tts_time = await generate_tts(full_text)
                        audio_b64 = base64.b64encode(wav_bytes).decode()
                        await ws.send_json({
                            "type": "tts_audio",
                            "audio": audio_b64,
                            "time": round(tts_time, 2),
                        })
                    except Exception as e:
                        await ws.send_json({
                            "type": "error",
                            "message": f"TTS error: {e}",
                        })

            # ── Clear conversation ──────────────────────────────────
            elif msg_type == "clear":
                previous_response_id = None
                await ws.send_json({
                    "type": "status",
                    "message": "Conversation cleared",
                })

            # ── Config update ───────────────────────────────────────
            elif msg_type == "config":
                if "tts_enabled" in msg:
                    tts_enabled = bool(msg["tts_enabled"]) and tts_model is not None
                    await ws.send_json({
                        "type": "status",
                        "message": f"TTS {'enabled' if tts_enabled else 'disabled'}",
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# ── Static files + index ───────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Voice Assistant")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS model loading")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    cli_args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=cli_args.port, log_level="info")
