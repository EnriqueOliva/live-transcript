# Setup Guide

## Prerequisites

The only thing you need to install manually is **uv** (the Python package manager). Everything else — Python, dependencies, models — is handled automatically.

### Install uv

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal after installing so `uv` is on your PATH.

That's it. No need to install Python, CUDA, cuDNN, or anything else manually.

---

## Clone and run

```bash
git clone git@github.com:EnriqueOliva/live-transcript.git
cd live-transcript
```

### Desktop (NVIDIA GPU)

```bash
uv sync --group cuda
uv run python -m whisper_transcriber
```

### Laptop (no NVIDIA GPU)

```bash
uv sync
uv run python -m whisper_transcriber
```

The first run will download the Whisper turbo model (~1.5 GB). Subsequent runs are instant.

---

## What each command does

| Command | What it does |
|---|---|
| `uv sync` | Installs Python 3.12 + all dependencies (~500 MB). CPU-only inference. |
| `uv sync --group cuda` | Same as above + PyTorch CUDA + NVIDIA DLLs (~4 GB total). Enables GPU acceleration. |
| `uv run python -m whisper_transcriber` | Launches the app. Auto-detects GPU/CPU. |
| `uv run python scripts/verify_gpu.py` | Shows detected hardware and recommended config. |

---

## Verify setup (optional)

```bash
uv run python scripts/verify_gpu.py
```

**Desktop output** (NVIDIA GPU detected):
```
Mode:             GPU (CUDA)
Recommended:      turbo model, float16
```

**Laptop output** (CPU only):
```
Mode:             CPU
Recommended:      turbo model, int8
```

---

## Notes

- **NVIDIA driver**: If you have an NVIDIA GPU, make sure the driver is version 535 or newer. Check with `nvidia-smi`. No need to install CUDA Toolkit separately — the pip packages handle it.
- **Windows Developer Mode**: If the turbo model fails to download with a "privilege" error, enable Developer Mode in Settings > System > For Developers. This allows the HuggingFace cache to create symlinks.
- **Firewall/proxy**: The first run downloads models from huggingface.co. Make sure it's not blocked.
