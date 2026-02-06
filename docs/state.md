# Current State

Last updated: [2026-02-06] by Claude

## Just Completed
- Created `app.py` — FastAPI backend with WebSocket, streaming LLM (SSE), async ASR, optional TTS, GPU lock, per-connection session state
- Created `static/index.html` — Single-file dark-theme frontend with chat bubbles, text input, mic push-to-talk, waveform visualizer, streaming token display, TTS toggle, clear button, timing badges, auto-reconnect WebSocket, 16kHz audio downsampling
- Updated `requirements.txt` — added fastapi, uvicorn[standard], httpx, websockets
- Fixed SSE parsing: LM Studio uses `content` field (not `delta`) and `result.response_id` (not top-level)
- Tested: text streaming works, stateful conversation works, `voice.py` imports OK

## In Progress
- Nothing active

## Next Up
- Test mic recording + ASR in browser (needs mic hardware)
- Test TTS playback (run without `--no-tts`)
- Browser testing in Chrome/Edge

## Environment
- Branch: master
- Tests: N/A (no test suite yet)
- Build: N/A (no build step — single HTML file)
