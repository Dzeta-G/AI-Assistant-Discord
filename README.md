# discord-ark-bot

ARK-enabled Discord voice bot that listens in a voice channel, performs VAD, streams transcription to OpenAI Realtime API, routes events, and responds in ARK persona (via Chat/Realtime) with optional TTS playback.

## Features
- Voice join/leave, per-user audio capture (Opus→PCM), WebRTC VAD
- Always-on local transcription via whisper.cpp with rolling 30s context buffer
- Two-level memory: 90s short-term buffer plus persistent SQLite/JSON log with recall commands
- On-demand OpenAI Realtime ASR/Chat, activated only after wake-word, with HTTP Whisper fallback
- Router with wake-word, triggers `!ALERT`/`!CRIT`, default silent mode
- ARK agent persona with campaign configuration and character mapping
- TTS via OpenAI (fallback local stub), audio playback in voice channel
- Discord commands: `!join`, `!leave`, `!set-character`, `!ark on/off`, `!ark wakeword`, `!ark test`
- Memory utilities: `!ark recall <query>` and `!ark clear-log`
- Logging and short conversation context buffer
- Tests for router and basic pipeline smoke

## Requirements
- Python 3.11+
- FFmpeg installed and in PATH (for audio playback)
- whisper.cpp binary + GGML model (for offline ASR; see instructions below)
- macOS/Linux

## Quick Start

1) Create a Discord Bot in Developer Portal
- Go to https://discord.com/developers/applications
- New Application → Bot → Reset Token, copy
- Privileged Gateway Intents: enable Message Content Intent; enable Server Members Intent (for mentions and mapping)
- OAuth2 → URL Generator: scopes `bot`, permissions: `Connect`, `Speak`, `Use Slash Commands`, `Read Message History` (and any others as needed). Invite to your server.

2) OpenAI API Key
- Create key with Realtime access. Set in `.env` as `OPENAI_API_KEY`.

3) Setup repo
```
cd discord-ark-bot
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

4) Configure
- Edit `src/config/campaign.yml` for ARK tone, triggers, rules
- Edit `src/config/mapping.yml` to map `user_id` to character/persona
- Update `.env` with Discord and OpenAI tokens

5) Run
```
make run
python -m src.main
```

6) In Discord
- In a text channel: `!join` while connected to a voice channel
- Bot joins your voice channel and begins local transcription (no API cost)
- Say the wake-word (default: `арк`) to open a 12s active window, or use triggers `!ALERT`, `!CRIT`

## Configuration Files
- `src/config/campaign.yml`: rules, triggers, ARK tone
- `src/config/mapping.yml`: mapping from Discord `user_id` to character name or role metadata

## Environment Variables (`.env`)
See `.env.example` for all keys.
- `DISCORD_TOKEN`: Discord bot token
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_REALTIME_MODEL`: Realtime model id (e.g., `gpt-4o-realtime-preview`)
- `OPENAI_TTS_MODEL`: TTS model id (e.g., `gpt-4o-mini-tts` or `tts-1-hd`)
- `OPENAI_WHISPER_MODEL`: fallback ASR model (e.g., `whisper-1`)
- `LOCAL_ASR_MODEL`: whisper.cpp GGML model path or preset (e.g., `~/models/ggml-base.bin`)
- `LOCAL_ASR_BIN`: optional path to the compiled `whisper.cpp` binary (defaults to `whisper.cpp/main`)
- `ARK_LOG_BACKEND`: `sqlite` or `json` backend for persistent memory
- `ARK_LOG_FILE`: output file for the persistent log

## Local ASR (whisper.cpp)
1. Clone whisper.cpp
```
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp && make
```
2. Download a GGML model, e.g. `models/ggml-base.bin`
```
./models/download-ggml-model.sh base
```
3. Point the bot to the binary and model (edit `.env`):
```
LOCAL_ASR_BIN=/path/to/whisper.cpp/main
LOCAL_ASR_MODEL=/path/to/whisper.cpp/models/ggml-base.bin
```
On Apple Silicon, `make` produces a binary with NEON acceleration; no extra flags required.
If the binary/model are not detected the bot will still run, but wake-word detection and context rely on local ASR so responses will be limited.

## Notes on Voice Receive
This project expects a Discord library that supports voice receive. The code targets `discord.py` 2.x API surface but relies on sink-like behavior commonly available in Py-cord (import name remains `discord`). If your `discord` package lacks recording, either install `py-cord>=2.5` or use the provided lightweight UDP capture in `discord_voice.py` if supported by your environment. See inline comments in `src/discord_voice.py`.

## FAQ
- Bot joins but no transcription?
  - Ensure FFmpeg is installed. Confirm Py-cord voice receive (sinks) is available. Check `.env` and model names.
- Realtime API blocked?
  - The ASR module falls back to HTTP Whisper if websocket fails, producing final-only segments.
- TTS fails?
  - Fallback will reply in text channel; ensure `OPENAI_TTS_MODEL` is valid.

## Testing
```
pytest -q
```
