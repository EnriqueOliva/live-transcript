# Setup Guide

## Prerequisites

The only thing to install by hand is **uv**, the Python package manager. Python, the dependencies and the speech model are handled automatically.

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal afterwards so `uv` is on your PATH.

## Clone and run

```bash
git clone git@github.com:EnriqueOliva/livevox.git
cd livevox
```

### Desktop (NVIDIA GPU)

```bash
uv sync --group cuda
uv run python -m livevox
```

### Laptop (no NVIDIA GPU)

```bash
uv sync
uv run python -m livevox
```

The first run downloads the Whisper turbo model (about 1.5 GB). Later runs start right away. If Sotvox already downloaded it, it is reused.

## What each command does

| Command | What it does |
|---|---|
| `uv sync` | Installs Python 3.12 and the dependencies. CPU transcription. |
| `uv sync --group cuda` | Same, plus NVIDIA cuBLAS (about 800 MB on disk). Enables GPU transcription. |
| `uv run python -m livevox` | Launches the app. Uses the GPU when cuBLAS is available, the CPU otherwise. |
| `uv run python scripts/verify_gpu.py` | Shows the detected hardware and the mode the app will use. |

## Verify the setup (optional)

```bash
uv run python scripts/verify_gpu.py
```

With a GPU ready it ends with:

```
Mode:             GPU (CUDA)
Recommended:      turbo model, float16
```

On a CPU-only machine:

```
Mode:             CPU
Recommended:      turbo model, int8
```

## Notes

- **NVIDIA driver**: version 535 or newer (`nvidia-smi` shows it). The CUDA Toolkit is not needed, and neither are PyTorch or cuDNN: CTranslate2 4.6.3 and later only needs cuBLAS.
- **GPU present but not set up**: the app notices that cuBLAS is missing and transcribes on the CPU instead of failing.
- **Firewall or proxy**: the first run downloads the model from huggingface.co.
