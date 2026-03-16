# Live Transcript

Real-time system audio transcription for Windows using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with GPU acceleration.

Captures WASAPI loopback audio (what your speakers play) and produces a live transcript. Built for transcribing Zoom lectures, meetings, podcasts, or any audio playing on your system.

![PySide6](https://img.shields.io/badge/GUI-PySide6-blue)
![Python 3.12](https://img.shields.io/badge/python-3.12-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## Features

- Real-time transcription with Whisper turbo model (~0.1s inference per 30s chunk on RTX 4070)
- WASAPI loopback capture (system audio, no virtual cables needed)
- Optional microphone recording with audio mixing
- Audio visualizer (EQ frequency bars)
- Automatic silence detection with heartbeat status
- Session-based output with plain text and timestamped transcripts
- Hold-back boundary merging (zero word loss at chunk boundaries)
- Dark mode PySide6 GUI
- 7-layer hallucination prevention (VAD, no-speech filter, compression ratio, etc.)

## Requirements

- Windows 11
- NVIDIA GPU with CUDA support (tested on RTX 4070 12GB)
- NVIDIA driver 535+

## Setup

### 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install dependencies

```bash
uv python install 3.12
uv sync
```

### 3. Verify GPU

```bash
uv run python scripts/verify_gpu.py
```

The turbo model (~1.5 GB) will download automatically on first run.

## Run

```bash
uv run python -m whisper_transcriber
```

1. Select model (turbo recommended) and language
2. Optionally check "Mic" to also record your microphone
3. Click **Start** to begin transcription
4. Click **Open Folder** to view transcripts
5. Click **Stop** when done

## Output

Sessions are saved with timestamped folders:

```
transcripts/[DD-MM-YY] - [HH-MM]/
    transcript.txt                    # plain text
    transcript_with_timestamps.txt    # [HH:MM:SS -> HH:MM:SS] text

logs/[DD-MM-YY] - [HH-MM]/
    session.log                       # detailed session log
```

## Architecture

4-thread pipeline with queue-based communication:

```
WASAPI Loopback ─┐
                  ├─► Accumulator ─► Transcription Worker ─► GUI + File
Microphone (opt) ─┘   (30s chunks    (faster-whisper        (PySide6 +
                       5s overlap)     turbo model)           transcript writer)
```

Key design decisions:
- **SimpleQueue** for PortAudio callback thread safety
- **resample_poly(1,3)** for exact 48kHz→16kHz conversion
- **Hold-back + text merge** at chunk boundaries prevents word loss
- **Safe zone** filtering defers overlap-region segments to the next chunk
- Worker writes directly to TranscriptWriter (no Qt signal race condition)

## CUDA Troubleshooting

If CTranslate2 reports missing CUDA libraries:

**Option A** (default): The `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages are included. If CTranslate2 can't find them, add to your PATH:
```
.venv/Lib/site-packages/nvidia/cublas/bin
.venv/Lib/site-packages/nvidia/cudnn/bin
```

**Option B** (fallback): Download CUDA12_v3 from [Purfview/whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs). Extract DLLs to the project root.

## Development

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

## Legal Note

Ensure compliance with your institution's recording policies and the terms of service of any conferencing software before capturing audio. This tool captures system audio output; inform all participants as required by local regulations.

## License

[MIT](LICENSE)
