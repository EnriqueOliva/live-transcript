# Live Transcript

Real-time system audio transcription for Windows using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

Captures WASAPI loopback audio (what your speakers play) and produces a live transcript. Built for transcribing Zoom lectures, meetings, podcasts, or any audio playing on your system.

![PySide6](https://img.shields.io/badge/GUI-PySide6-blue)
![Python 3.12](https://img.shields.io/badge/python-3.12-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## Features

- Real-time transcription with Whisper turbo model
- Works on **any Windows machine** (CPU or NVIDIA GPU)
- WASAPI loopback capture (system audio, no virtual cables needed)
- Optional microphone recording with audio mixing
- Audio visualizer (EQ frequency bars)
- Automatic silence detection with heartbeat status
- Session-based output with plain text and timestamped transcripts
- Hold-back boundary merging (zero word loss at chunk boundaries)
- Dark mode PySide6 GUI

## Requirements

- Windows 11
- Python 3.12 (managed by uv)

**For GPU acceleration (optional):** NVIDIA GPU with CUDA support and driver 535+

## Setup

### 1. Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install dependencies

**CPU only** (works on any machine):
```bash
uv python install 3.12
uv sync
```

**With NVIDIA GPU acceleration** (faster inference):
```bash
uv python install 3.12
uv sync --group cuda
```

### 3. Verify setup

```bash
uv run python scripts/verify_gpu.py
```

The Whisper model (~1.5 GB) downloads automatically on first run.

## Run

```bash
uv run python -m whisper_transcriber
```

1. Select model (turbo recommended) and language
2. Optionally check "Mic" to also record your microphone
3. Click **Start** to begin transcription
4. Click **Open Folder** to view transcripts
5. Click **Stop** when done

### Performance

| Hardware | Model | Inference per 30s chunk |
|----------|-------|------------------------|
| RTX 4070 (CUDA) | turbo, float16 | ~0.1s |
| i5-12th gen (CPU) | turbo, int8 | ~3-8s |
| i5-12th gen (CPU) | small, int8 | ~1-2s |

The app auto-detects GPU availability. On CPU, it uses `int8` quantization for maximum speed.

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
Microphone (opt) ─┘   (30s chunks    (faster-whisper         (PySide6 +
                       5s overlap)     turbo model)            transcript writer)
```

## CUDA Troubleshooting

If CTranslate2 reports missing CUDA libraries after `uv sync --group cuda`:

**Option A**: The `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages are included. Add to your PATH:
```
.venv/Lib/site-packages/nvidia/cublas/bin
.venv/Lib/site-packages/nvidia/cudnn/bin
```

**Option B**: Download CUDA12_v3 from [Purfview/whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs). Extract DLLs to the project root.

## Development

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

## Legal Note

Ensure compliance with your institution's recording policies and the terms of service of any conferencing software before capturing audio.

## License

[MIT](LICENSE)
