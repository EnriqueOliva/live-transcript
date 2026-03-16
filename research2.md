# Lean Windows Live-Transcription App Blueprint for Claude Code

## Context and hard constraints

You want a **Windows desktop app** that can run during a university class (e.g., via Zoom) and automatically produce a **high-accuracy transcript** in a **.txt file**, without the workflow of recording and uploading elsewhere. The transcript does not need to be truly real time; **up to ~60 seconds transcription delay is acceptable**.

Your machine/environment constraints are very specific:

- **GPU:** entity["company","NVIDIA","gpu maker, us"] RTX 4070 12GB OC (non-super)
- **CPU:** entity["company","Intel","cpu maker, us"] i5-13600K
- **RAM:** 64GB DDR4
- **Storage:** 2 NVMEs
- **OS:** Windows 11 (latest update)
- **Editor:** VS Code
- **Python constraint:** **No global/system Python**. Must use **uv** (or equivalent) so Python + deps are local/project-scoped in the working folder. uv can also install/manage Python versions on demand. citeturn18view0
- **App requirements:** extremely minimal UI, strict clean code practices, and **files ≤150 lines** (enforced by design). Must include **(1) live transcript view**, **(2) live log view**, and **(3) persistent log files** in a dedicated folder.

The key technical decisions that dominate “latest best practice” here are: (a) the transcription engine configuration for accuracy + stability, and (b) how to capture **Windows system audio** reliably in a way that works during a Zoom class.

## Whisper model selection for maximum accuracy and stable long sessions

### Why Whisper and which checkpoint

entity["company","OpenAI","ai lab, us"]’s Whisper is a general-purpose ASR model trained on diverse data and designed for multilingual transcription and related tasks. citeturn14view0turn2search22

For your “highest accuracy possible” constraint, Whisper’s own docs emphasize a **size ↔ accuracy ↔ speed** trade-off. The official table lists the large model family at ~**10GB VRAM**, and the **turbo** model at ~**6GB VRAM** with ~8× faster relative speed (vs. large), while describing turbo as an optimized `large-v3` with “minimal degradation in accuracy.” citeturn14view0turn15view0turn3search24

Given your **12GB VRAM**, a reasonable “max accuracy first” default is:

- **Default model:** `large-v3` (or `large`, which Whisper maintainers have stated aliases to the latest in the series, i.e. `large-v3`, in recent package versions) citeturn2search31turn14view0
- **Performance fallback:** `turbo` (still strong accuracy, substantially faster, and likely smoother while Zoom is also using system resources) citeturn14view0turn3search24

### Long-form / continuous transcription behavior and what it implies

Whisper’s `transcribe()` performs long-form transcription by processing audio with a **sliding 30-second window** internally. citeturn14view0turn15view0  
The implementation also explicitly describes detecting language using **up to the first 30 seconds** when language is not specified. citeturn10view1

This strongly suggests a robust design pattern for your “≤ 60s delay” requirement:

- Capture system audio continuously.
- Batch audio into chunks aligned with Whisper’s internal assumptions (e.g., 25–30s steps with a small overlap).
- Append segments with timestamps, using segment time filtering to prevent duplicated overlap output.

### Hallucination control knobs worth exposing in the UI

Whisper’s own transcription code exposes the stability/anti-hallucination controls you’ll want surfaced (even if tucked into an “Advanced” collapsible panel):

- `no_speech_threshold` (silence detection gating behavior)
- `logprob_threshold` and `compression_ratio_threshold` (failure heuristics / repetition detection)
- `condition_on_previous_text` (trade consistency vs. risk of repetition loops)
- `initial_prompt` / `carry_initial_prompt` (context biasing for proper nouns/domain vocab)
- `hallucination_silence_threshold` (skip silent stretches when a hallucination is suspected, when word timestamps enabled) citeturn4view0turn10view1

These are preferable to ad-hoc heuristics because they are **first-class in upstream Whisper** (i.e., “official” behavior, not folk wisdom).

### Model storage inside the project folder

By default, Whisper caches model checkpoints under a user cache directory based on `~/.cache/whisper` (or `$XDG_CACHE_HOME/whisper`). The maintainer explanation makes clear it’s regular “cache reuse” behavior. citeturn12view0

For your “keep everything enclosed in the working folder” preference, the app should load the model with a **project-local** `download_root`, so the model cache can live under something like `./data/models/whisper/`. citeturn12view0

## Capturing live Windows system audio during Zoom classes

### Official Windows approach 1: WASAPI loopback (endpoint capture)

On Windows, loopback mode in WASAPI allows capturing “the audio stream that is being played by a rendering endpoint device.” This is the canonical “record what you hear” path. citeturn0search0

In practical terms, this solves your core requirement: capture Zoom’s audio output without needing to record a microphone input. Documentation and mainstream tooling guidance align that WASAPI loopback is designed to capture computer playback even when “Stereo Mix” isn’t available. citeturn0search0turn6search24

### Official Windows approach 2: process-restricted capture (Zoom-only capture)

If you want to avoid capturing _all_ system audio (notifications, other apps) and focus only on Zoom, Microsoft published an **ApplicationLoopback** sample showing how to restrict captured audio to a **specific process tree** (or exclude a process tree), using `ActivateAudioInterfaceAsync` with a newer initialization structure. It also notes this method is **not tied to a specific audio endpoint** (so you don’t need separate per-device capture clients). citeturn7view0turn8view0turn3search19

This sample explicitly requires **Windows 10 build 20348 or later** (Windows 11 is fine). citeturn7view0turn3search19

**Implication for your app design:** a “best practice” architecture is to implement:

- **Basic mode:** system-wide WASAPI loopback capture (simpler)
- **Precision mode (optional):** Zoom-only process capture, using the ApplicationLoopback approach

The second mode is more complex but aligns strongly with your “accuracy above all” idea, because the transcript will be less polluted by unrelated audio.

### A pragmatic Python-layer capture strategy

Because you require project-local Python tooling, a pragmatic “lean app” approach is a Python audio capture wrapper that supports loopback devices. A documented example is the SoundCard library interface that can enumerate microphones including loopback (“virtual microphones that record the output of a speaker”) via `all_microphones(include_loopback=True)`. citeturn6search4turn6search0

However, if you later need “Zoom-only” capture, the Microsoft sample is the canonical reference point for a native helper. citeturn7view0turn8view0

## Toolchain and packaging aligned with your “no global Python” requirement

### uv-managed, project-scoped Python

uv can install and manage Python versions and will automatically download missing Python versions as needed. citeturn18view0  
This matches your “no system Python” constraint: the project can be self-contained with uv-managed Python + `.venv`.

uv also supports standard project workflows where `uv sync` creates `.venv` and a lockfile. citeturn0search5

### Installing PyTorch (CUDA build) in an uv project

PyTorch packaging is unusual because accelerator-specific wheels often live on dedicated indexes, and uv documents how to configure explicit indexes and map packages to those indexes. citeturn19view0

This is important because you want Whisper to use the RTX 4070. A first-class (documented) `uv` approach is:

- Define `[[tool.uv.index]]` entries (e.g., `https://download.pytorch.org/whl/cu128`)
- Map `torch` / `torchaudio` to that index via `[tool.uv.sources]`, preferably with `explicit = true` so only PyTorch packages are pulled from that index. citeturn19view0

PyTorch’s own installation page recommends verifying installation with `torch.cuda.is_available()`. citeturn16view0

### Minimal Windows GUI packaging

For a minimal native-feeling Windows desktop UI in Python, Qt for Python provides the official PySide6 bindings. citeturn1search13turn5search36  
Qt’s deployment docs describe `pyside6-deploy` (shipping since Qt for Python 6.4) as a deployment tool and states it is a wrapper around Nuitka for producing platform executables. citeturn5search3turn5search24

This is a strong “industry standard, official toolchain” option for a Windows `.exe` output that does not require a global Python install.

### Logging architecture: file logs plus live UI logs

Python’s logging cookbook and handler references document `QueueHandler`/`QueueListener` as the standard pattern for dispatching log records from performance-critical threads to handlers using an internal listener thread. citeturn5search14turn5search6turn5search23

This pattern is ideal for a GUI app with background capture/transcription threads, because you can:

- log to rotating files on disk
- mirror logs into a GUI widget through a queue-driven bridge  
  without blocking the UI thread.

## Claude Code prompt file

```markdown
# Prompt for Claude Code: Build a Lean Windows App That Live-Transcribes System Audio with Whisper (Project-Local Python via uv)

You are Claude Code. Build a Windows desktop application **from zero** that transcribes **live Windows system audio** (Zoom class audio) into a `.txt` transcript file while the session is running.

## My PC / environment (must be assumed and referenced in the design decisions)

- GPU: NVidia RTX 4070 12GB OC (non-super)
- CPU: Intel i5-13600K
- RAM: 64GB DDR4
- Storage: 2× NVMe
- OS: Windows 11 (latest update)
- Editor: VS Code
- CRITICAL CONSTRAINT: No Python installed globally. You MUST use `uv` (or another project-level solution) so Python + deps are installed and run from the project folder (enclosed in the folder where we work). Do not rely on system Python.
- Internet is allowed only for dependency/model downloads during setup; app runtime must be offline and local.

## Product goal

A minimal Windows app that I run during a Zoom university class. When the class ends, I have:

- `transcript.txt` (human-readable, new text appended live)
- optional `transcript_with_timestamps.txt` (recommended)
- a full log file saved to a dedicated logs folder
- the UI shows the transcript being written and shows logs live

I accept up to ~60 seconds delay: it does NOT need to be perfect real-time.

### Non-negotiable requirements

- Extremely minimalistic UI, extremely lean codebase.
- Strict clean code / SOLID-ish layering, durable error handling.
- Every source file must be **≤ 150 lines**. If you need more code, split into more files/modules.
- One responsibility per module; avoid “god files.”
- A complete logging system:
  - logs to `./data/logs/` (rotating or one log per run)
  - logs visible live in the app UI
- Must allow choosing:
  - Whisper model (turbo, large-v3 at least; plus tiny/base/small/medium optional)
  - language (Auto + fixed options)
  - capture device mode at minimum: system audio output loopback (what-you-hear)
  - chunk size / delay settings (defaults should “just work”)
- Must be safe to stop/start; should never corrupt transcript/log files.
- Must be able to run while Zoom is running (do not freeze the UI).

### Strong preference: official + well-known approaches only

Prefer official documentation and widely-used tools. Avoid shady/unknown packages.

## High-level technical approach (you must follow and refine)

### Language + UI

- Use Python with a Qt UI: `PySide6`.
- Keep the UI to a single main window:
  - top control row: Start/Stop, Model dropdown, Language dropdown, Output folder selector, Settings (optional)
  - center: Transcript view (read-only, append as we write)
  - bottom: Log view (read-only, tail-like)
  - status bar: “Recording”, “Transcribing…”, GPU/CPU indicator, buffer health

### Packaging / “no global Python”

- Use `uv` to create and manage:
  - project
  - virtual environment
  - Python version (uv-managed)
- The repo must run with:
  - `uv sync`
  - `uv run python -m app` (or equivalent)
- Provide a `README.md` with exact commands.

### GPU acceleration (RTX 4070)

- Use PyTorch with a CUDA build so Whisper uses the RTX 4070.
- Configure PyTorch in uv using the **official PyTorch wheel index** approach (explicit index), so it’s deterministic on Windows.
- Provide a one-command GPU verification script: prints torch version, CUDA availability, device name.

### Whisper usage

- Use `openai-whisper` (official).
- Default model should prioritize accuracy: `large-v3` (or `large` if it maps to large-v3). Provide turbo as a fast fallback.
- Store Whisper model files inside project folder (example: `./data/models/whisper/`) using `download_root` when loading the model. Do not stash models only in the user home directory.

### Audio capture (system output)

- Must capture Windows playback audio (what I hear) while Zoom runs.
- Minimum implementation: system-wide loopback capture. Provide device selection, and a sensible default (default speaker loopback).
- If implementing “Zoom-only capture” is feasible without bloating the app: add an optional advanced mode that captures audio from a target process tree using Microsoft’s ApplicationLoopback approach. But keep the base app simple; correctness first.

### Chunking strategy (≤60s delay)

Implement a robust chunking strategy that avoids duplicates and missing boundary words:

- Capture continuously into a ring buffer (in-memory).
- Resample/downmix to Whisper’s expected audio format (16kHz mono float32).
- Every N seconds, extract a chunk with a small overlap (e.g., 30s chunk, 5s overlap).
- Transcribe the chunk in a background worker.
- Use timestamps returned by Whisper segments to:
  - only append segments that come after `last_written_time`
  - prevent overlap duplicates
- Write to transcript incrementally, flush to disk frequently.

### Hallucination / silence robustness

Expose and set reasonable defaults for key Whisper controls:

- `no_speech_threshold`
- `logprob_threshold`
- `compression_ratio_threshold`
- `condition_on_previous_text`
- optional `initial_prompt` (UI field)
  Make defaults conservative so it doesn’t spam random text during silence.

### Logging

Use a queue-based logging architecture:

- background threads log normally
- a QueueListener writes to file handlers
- the UI subscribes to the same queue for live display

## Repository structure (must follow; keep each file ≤150 lines)

Use a `src/` layout. Example:

- `pyproject.toml` (uv project + dependencies + formatting/lint config)
- `uv.lock`
- `README.md`
- `src/app/__main__.py` (entry point)
- `src/app/bootstrap.py` (wire dependencies)
- `src/app/ui/main_window.py` (UI layout + signals)
- `src/app/ui/widgets/transcript_view.py`
- `src/app/ui/widgets/log_view.py`
- `src/app/services/session_controller.py` (start/stop session orchestration)
- `src/app/audio/capture.py` (loopback capture abstraction)
- `src/app/audio/devices.py` (device discovery)
- `src/app/audio/ring_buffer.py` (ring buffer)
- `src/app/audio/resample.py` (to 16kHz mono)
- `src/app/stt/whisper_engine.py` (model load + transcribe chunk)
- `src/app/stt/chunk_merger.py` (timestamp-based dedupe + append logic)
- `src/app/io/transcript_writer.py` (atomic-ish append, flush)
- `src/app/io/paths.py` (data/logs/sessions folders)
- `src/app/logging/log_setup.py` (QueueHandler/QueueListener + file handlers)
- `src/app/logging/log_bus.py` (bridge logs to UI)
- `src/app/config/settings.py` (persist user settings in `./data/settings.json`)
- `scripts/verify_gpu.py`
- `scripts/run_dev.ps1` (optional convenience)

If any file grows past 150 lines, split it.

## Dependency rules (uv)

1. Use `uv python install` to provision a managed Python (choose a stable modern version compatible with torch + pyside6; prefer 3.12 unless there is a compelling reason).
2. Use `uv` to add dependencies and lock them.
3. Configure the PyTorch CUDA wheel index in `pyproject.toml` using:
   - `[[tool.uv.index]]` with `explicit = true`
   - `[tool.uv.sources]` mapping `torch` and `torchaudio` to that index for Windows.
4. Dependencies (minimum):
   - `openai-whisper`
   - `torch`
   - `torchaudio`
   - `PySide6`
   - `numpy`
   - one reputable Windows loopback capture library (keep it simple)

## UX requirements (minimal but complete)

- Start button becomes Stop while active.
- Show current session folder path (where transcript/log/audio are being written).
- Transcript view updates as new segments are committed to disk.
- Log view shows live logs; include a “Copy logs” button optional.
- Settings persisted across runs:
  - model
  - language
  - chunk size
  - overlap seconds
  - output folder (default to `./data/sessions/`)

## Output files

For each session, create a folder like:
`./data/sessions/YYYY-MM-DD_HH-mm-ss/`

Inside:

- `transcript.txt` (append-only)
- `transcript_with_timestamps.txt` (recommended; segment-level timestamps)
- `session.log` (or link to central log + session id)
- optionally `audio.wav` (optional; if you implement it, do it efficiently)

## Build / run instructions (must be included in README)

- Install uv
- `uv python install ...`
- `uv sync`
- `uv run python -m app`
- Verify GPU: `uv run python scripts/verify_gpu.py`

## Quality gates (must implement)

- Add `ruff` config and run it cleanly.
- Add `mypy` config (reasonable strictness).
- Add a minimal `pytest` test suite:
  - ring buffer behavior
  - timestamp-based dedupe logic
  - settings load/save
- Provide a `make-like` command list in README (Windows-friendly).

## Legal/ethical note (brief)

Add a short note in README: make sure the user complies with university and Zoom recording policies.

## Deliverables

1. Full codebase with all files created.
2. README with exact setup/run steps.
3. Clean architecture, file size limits honored.
4. App runs on Windows 11 and can transcribe system audio during a Zoom class.

## Helpful official references (use these while building; do not paste large quotes)

- OpenAI Whisper repo: https://github.com/openai/whisper
- Whisper transcribe internals (parameters): https://github.com/openai/whisper/blob/main/whisper/transcribe.py
- Microsoft WASAPI loopback: https://learn.microsoft.com/windows/win32/coreaudio/loopback-recording
- Microsoft ApplicationLoopback sample: https://learn.microsoft.com/samples/microsoft/windows-classic-samples/applicationloopbackaudio-sample/
- uv docs (Python install + PyTorch integration):
  - https://docs.astral.sh/uv/guides/install-python/
  - https://docs.astral.sh/uv/guides/integration/pytorch/
- Qt for Python deployment (`pyside6-deploy`): https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html
- Python logging QueueListener: https://docs.python.org/3/howto/logging-cookbook.html

Now implement the project exactly as specified. Keep it lean, readable, and robust.
```
