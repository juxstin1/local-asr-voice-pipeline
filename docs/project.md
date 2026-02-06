# Project Roadmap

## Current Sprint
- [ ] Test web UI end-to-end with LM Studio running
- [ ] Test mic recording in browser (Chrome/Edge)
- [ ] Test TTS playback toggle

## Completed
- [x] [2026-02-06] Original CLI voice pipeline (voice.py)
- [x] [2026-02-06] Web UI + streaming LLM + text fallback (app.py, static/index.html)

## Technical Decisions
- [2026-02-06] WebSocket over REST+SSE — need bidirectional (client sends audio/text, server streams tokens/audio)
- [2026-02-06] Base64 audio in JSON — recordings are short (~few hundred KB), simplicity over binary frames
- [2026-02-06] Single HTML file — no npm/build tools, total JS under 500 lines
- [2026-02-06] asyncio.Lock for GPU — serializes ASR and TTS on shared 4090
- [2026-02-06] voice.py left untouched — original CLI pipeline still works independently

## Known Issues / Tech Debt
- ScriptProcessor is deprecated; should migrate to AudioWorklet for recording (works fine in current browsers)
- No authentication on WebSocket — fine for local use, not for public deployment
- No test suite
