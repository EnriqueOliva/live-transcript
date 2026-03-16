# Definitive research for a Windows audio transcription app

**This report compiles verified, source-backed findings across seven research domains needed to write a precise Claude Code prompt for building a Windows desktop audio transcription app.** Every version number, API detail, and code snippet below reflects the latest stable releases as of March 2026. The stack converges on **faster-whisper 1.2.1** for transcription, **PyAudioWPatch 0.2.12.8** for WASAPI loopback capture, **CustomTkinter 5.2.2** for the GUI, all managed by **uv 0.10.10** with **PyTorch 2.9.1 (CUDA 12.8)** acceleration on an RTX 4070.

---

## 1. faster-whisper and OpenAI Whisper models

### Core library details

**faster-whisper 1.2.1** (MIT license, PyPI) is a CTranslate2-based reimplementation of OpenAI's Whisper that delivers **4× faster sequential inference** and up to **12–16× faster batched inference** compared to the original PyTorch implementation. It requires Python ≥3.9 and depends on `ctranslate2`, `av` (bundles FFmpeg internally — no system FFmpeg needed), `huggingface-hub`, `onnxruntime`, and `tokenizers`. Install with `pip install faster-whisper`.

The library automatically downloads CTranslate2-converted models from Hugging Face Hub when you pass a model name string to `WhisperModel()`. The full model lineup and their characteristics on an RTX 4070 with `float16` compute:

| Model                  | Parameters | VRAM (FP16) | VRAM (INT8) | Relative speed | English WER (approx)       | Notes                                      |
| ---------------------- | ---------- | ----------- | ----------- | -------------- | -------------------------- | ------------------------------------------ |
| `tiny` / `tiny.en`     | 39M        | <1 GB       | <1 GB       | ~10×           | ~5.6% (en) / ~7.6% (multi) | Fastest, lowest quality                    |
| `base` / `base.en`     | 74M        | ~1 GB       | <1 GB       | ~7×            | ~4.3% (en) / ~5.0% (multi) | Good for prototyping                       |
| `small` / `small.en`   | 244M       | ~1–2 GB     | ~1 GB       | ~4×            | ~3.0% (en) / ~3.4% (multi) | Solid mid-range                            |
| `medium` / `medium.en` | 769M       | ~3 GB       | ~2 GB       | ~2×            | ~3.0% (en) / ~2.9% (multi) | Strong multilingual                        |
| `large-v2`             | 1,550M     | ~5–6 GB     | ~3–4 GB     | 1×             | ~2.7%                      | Sometimes fewer hallucinations than v3     |
| `large-v3`             | 1,550M     | ~5–6 GB     | ~3–4 GB     | 1×             | ~2.4%                      | Best accuracy, 128 Mel bins                |
| `turbo`                | 809M       | ~3–4 GB     | ~2–3 GB     | ~8×            | ~2.5%                      | Decoder reduced from 32→4 layers           |
| `distil-large-v3`      | ~756M      | ~4 GB       | ~2–3 GB     | **6.3×**       | Within 1% of large-v3      | **English only**, optimal chunk_length=25s |

**distil-large-v3** deserves special attention. Created by Hugging Face via knowledge distillation, it keeps the full large-v3 encoder but shrinks the decoder from 32 to 2 layers. It runs **6.3× faster** than large-v3 with accuracy within 1% WER on sequential long-form and actually **outperforms large-v3 by 0.1%** on chunked long-form (fewer hallucinations). The critical constraint: **English only**. For multilingual, use `large-v3` or `turbo`.

### GPU acceleration on RTX 4070 (Windows 11)

The **RTX 4070 has Compute Capability 8.9** (Ada Lovelace, AD104 chip) with **12 GB GDDR6X VRAM**. faster-whisper's GPU path requires **CUDA 12 + cuDNN 9** via the CTranslate2 backend. On Windows, the recommended setup uses **Purfview's whisper-standalone-win** library archive (CUDA12_v3: cuBLAS v12.8.4.1 + cuDNN v9.8.0.87) — download the DLLs and place them on your system PATH or in the application directory. The `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` pip packages work on Linux but are less reliable on Windows.

NVIDIA driver must be **535+** or newer to support CUDA 12. Verify with `nvidia-smi`. Note that **PyTorch bundles its own CUDA runtime** separately from what CTranslate2 needs, so both can coexist — PyTorch's CUDA wheels are self-contained.

### INT8 vs float16 compute types

CTranslate2 offers several quantization modes. For the RTX 4070, two matter most:

**`float16`** keeps all weights and computation in half precision. It's the **recommended default** — fast, excellent accuracy, and the RTX 4070 has native FP16 tensor cores. With `large-v3`, expect ~5–6 GB VRAM usage.

**`int8_float16`** quantizes embedding and linear layers to 8-bit integers while keeping non-quantizable layers in FP16. This is the **best option when VRAM is constrained** — `large-v3` fits in ~3–4 GB with virtually no measurable accuracy loss. Requires Compute Capability ≥7.0 (RTX 4070's 8.9 supports it fully). Additional compute types include `float32` (wasteful on GPU), `bfloat16` (requires CC ≥8.0, works on 4070), and `int8` (auto-converts to `int8_float16` on GPU). The `auto` setting selects the fastest supported type.

```python
# Recommended for RTX 4070 with 12GB VRAM:
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
# Or for maximum VRAM efficiency:
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
```

### Python API essentials

The two main classes are `WhisperModel` for sequential transcription and `BatchedInferencePipeline` for batched (parallel chunk) transcription:

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Sequential (default):
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    "audio.mp3",       # str path, URL, or numpy float32 array
    language="en",     # or None for auto-detect
    beam_size=5,       # default 5 (vs OpenAI's default of 1)
    vad_filter=True,   # enable Silero VAD
    word_timestamps=False,
    condition_on_previous_text=True,
)
# IMPORTANT: segments is a GENERATOR — iterate to run transcription
for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")

# Batched (3-4× faster with VAD):
batched = BatchedInferencePipeline(model=model)
segments, info = batched.transcribe("audio.mp3", batch_size=16)
```

Key `transcribe()` parameters: `temperature` (list for fallback, default `[0.0, 0.2, ..., 1.0]`), `compression_ratio_threshold=2.4` (hallucination detection), `no_speech_threshold=0.6`, `initial_prompt` (hint text), `hotwords` (boost recognition of specific terms), `chunk_length` (default 30s, use 25s for distil models). The return `info` object contains `.language`, `.language_probability`, `.duration`, `.duration_after_vad`.

**Known Windows caveats**: CUDA DLL version mismatches are the #1 issue (cuBLAS/cuDNN must match ctranslate2's expectations). The `large-v3` model can hallucinate during silence more than `large-v2` — always use VAD. The `segments` generator is lazy; call `list(segments)` to force execution if needed.

---

## 2. Voice Activity Detection in faster-whisper

faster-whisper integrates **Silero VAD v6**, a lightweight ~1.8 MB ONNX model that runs via `onnxruntime`. It processes audio in 512-sample windows and outputs speech probability per window. Enabling VAD typically **reduces compute by 40–60%** for audio with significant silence and prevents Whisper's notorious tendency to hallucinate text (e.g., "Thank you for listening") during quiet segments.

Enable with `vad_filter=True` on `WhisperModel.transcribe()`. In `BatchedInferencePipeline`, VAD is **enabled by default** and integral to the batching mechanism. The `VadOptions` dataclass controls behavior:

```python
segments, info = model.transcribe(
    audio,
    vad_filter=True,
    vad_parameters=dict(
        threshold=0.5,              # Speech probability threshold (0–1)
        min_speech_duration_ms=250, # Discard speech chunks shorter than this
        max_speech_duration_s=float("inf"),  # Split chunks longer than this
        min_silence_duration_ms=2000,  # Wait this long before splitting
        speech_pad_ms=400,          # Padding before/after speech segments
    ),
)
```

faster-whisper's defaults are **deliberately conservative** compared to raw Silero VAD — `min_silence_duration_ms` is **2000ms** (vs Silero's 100ms) and `speech_pad_ms` is **400ms** (vs 30ms). This ensures no speech gets accidentally clipped. For more aggressive silence removal in a real-time transcription scenario, reduce `min_silence_duration_ms` to **500ms** and `speech_pad_ms` to **200ms** while keeping `threshold` at 0.5.

The `info.duration_after_vad` field reports actual speech duration processed, useful for monitoring efficiency. Alternative VAD options exist (pyannote for better non-English performance, webrtcvad for a lightweight non-AI approach) but Silero VAD is the best default for most use cases.

---

## 3. Windows loopback audio capture with pyaudiowpatch

### Why pyaudiowpatch wins

Three Python libraries were evaluated for capturing Windows system audio (loopback — what the speakers are playing). **sounddevice does not support WASAPI loopback** — feature request #281 has been open since 2020 with no implementation, and the underlying PortAudio hasn't merged the required patch. **SoundCard 0.4.5** supports loopback via `include_loopback=True` but has several documented Windows bugs: single-channel recording produces garbage, a silence-at-start bug on Windows 11 where the stream terminates if no audio is playing, and data discontinuity warnings between chunks.

**PyAudioWPatch 0.2.12.8** (released January 14, 2026, Apache-2.0 license) is a fork of PyAudio that patches PortAudio to expose WASAPI loopback devices as virtual input devices. It provides dedicated helper methods for loopback device discovery, prebuilt Windows wheels for Python 3.7–3.14, and is the **most reliable, purpose-built option** for this use case.

### Installation and device enumeration

```bash
pip install PyAudioWPatch
```

Loopback devices appear as input devices with `[Loopback]` suffix appended to the output device name. The library adds several convenience methods:

```python
import pyaudiowpatch as pyaudio

with pyaudio.PyAudio() as p:
    # Quick default loopback device:
    default_lb = p.get_default_wasapi_loopback()
    print(f"Default loopback: {default_lb['name']}")
    print(f"  Sample rate: {int(default_lb['defaultSampleRate'])} Hz")
    print(f"  Channels: {default_lb['maxInputChannels']}")

    # Enumerate all WASAPI loopback devices:
    for device in p.get_loopback_device_info_generator():
        print(f"[{device['index']}] {device['name']}")
```

### Capture pattern for transcription pipeline

WASAPI loopback captures at the device's native sample rate (typically **48000 Hz stereo**). Whisper requires **16000 Hz mono float32**. The capture callback should be minimal — just queue raw bytes — with resampling handled in a separate processing step:

```python
import pyaudiowpatch as pyaudio
import numpy as np
import queue
import threading
from scipy.signal import resample

class LoopbackCapture:
    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.sample_rate = None
        self.channels = None

    def _callback(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def run(self):
        with pyaudio.PyAudio() as p:
            device = p.get_default_wasapi_loopback()
            self.sample_rate = int(device["defaultSampleRate"])
            self.channels = device["maxInputChannels"]

            with p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                frames_per_buffer=1024,
                input=True,
                input_device_index=device["index"],
                stream_callback=self._callback,
            ) as stream:
                while not self.stop_event.is_set():
                    self.stop_event.wait(timeout=0.1)

def convert_for_whisper(raw_bytes, source_rate, source_channels):
    """Convert loopback audio to Whisper format: 16kHz mono float32."""
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if source_channels > 1:
        audio = audio.reshape(-1, source_channels).mean(axis=1)
    if source_rate != 16000:
        num_samples = int(len(audio) * 16000 / source_rate)
        audio = resample(audio, num_samples)
    return audio
```

Buffer sizes of **1024–4096 frames** are typical. Smaller buffers reduce latency but increase callback frequency. For speech transcription, 1024 is a good default. The callback runs in PortAudio's internal audio thread — keep it lightweight (just queue the data).

---

## 4. uv project management and PyTorch CUDA setup

### uv essentials (v0.10.10)

**uv** by Astral (written in Rust, 10–100× faster than pip) is the modern Python project manager. Install on Windows with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`. Core workflow:

```bash
uv init whisper-transcriber --python 3.12   # Creates project scaffold
cd whisper-transcriber
uv add faster-whisper customtkinter numpy scipy PyAudioWPatch  # Add deps
uv add --group dev pytest ruff              # Dev dependencies
uv run python -m whisper_transcriber        # Run the app
```

uv generates a `.python-version` file (pins interpreter), `pyproject.toml` (PEP 621 metadata + `[tool.uv]` config), and `uv.lock` (cross-platform lockfile). It auto-creates `.venv/` on first run and auto-downloads Python if missing. Both `pyproject.toml` and `uv.lock` should be committed to version control.

### PyTorch 2.9.1 with CUDA 12.8 via uv

**PyTorch 2.9.1** is the latest stable release (with torchvision 0.24.1, torchaudio 2.9.1). For this version, CUDA variants available are **cu126, cu128, and cu130** — CUDA 12.1 and 12.4 indexes are no longer available for the latest PyTorch. **CUDA 12.8 is the recommended choice.** PyTorch pip wheels bundle the necessary CUDA runtime — no separate CUDA Toolkit installation needed for PyTorch itself (though faster-whisper/CTranslate2 needs its own CUDA libs separately).

The official uv integration for PyTorch uses named indexes with `explicit = true` to prevent dependency confusion:

```toml
[project]
name = "whisper-transcriber"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.9.1",
    "faster-whisper>=1.2.1",
    "customtkinter>=5.2.0",
    "PyAudioWPatch>=0.2.12",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

The `explicit = true` flag is critical — it ensures only `torch` (and any other packages you explicitly assign) are fetched from the PyTorch index, preventing random dependencies like `jinja2` from resolving against it. For cross-platform projects, add `marker = "sys_platform == 'linux' or sys_platform == 'win32'"` since PyTorch doesn't publish CUDA builds for macOS.

Running `uv sync` after this configuration will fetch the CUDA-enabled PyTorch wheel (~2.5 GB) and all other dependencies. Verify with:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## 5. CustomTkinter for the minimal GUI

**CustomTkinter 5.2.2** (MIT license, ~262 KB) is the clear best choice for this use case. Built atop Tkinter, it provides custom-drawn widgets with rounded corners, hover effects, and **first-class dark/light theme support** — one line enables it: `customtkinter.set_appearance_mode("Dark")`. It auto-detects Windows 11 system theme with the `"System"` mode. The widget set includes everything needed: `CTkButton` (start/stop), `CTkTextbox` (scrolling transcription display + log panel), `CTkOptionMenu` (model/language dropdowns), `CTkFrame` (layout), and `CTkLabel` (status).

**Maintenance note**: No new PyPI release in 12+ months, but the library is feature-complete for this use case, has 13.2k GitHub stars, and is built on stdlib Tkinter (guaranteed maintenance). PySide6 6.10.2 (LGPL, Qt Company) is the fallback if long-term maintenance becomes critical, but adds ~100+ MB of dependencies and significant boilerplate for a 5-widget app.

### Threading safety pattern

Tkinter is **not thread-safe** — widgets must only be modified from the main thread. The standard pattern uses `queue.Queue` + `root.after()` polling:

```python
import customtkinter as ctk
import queue

class MainWindow(ctk.CTk):
    def __init__(self, result_queue: queue.Queue):
        super().__init__()
        self.result_queue = result_queue

        ctk.set_appearance_mode("Dark")
        self.title("Audio Transcription")
        self.geometry("700x500")

        # Controls frame
        controls = ctk.CTkFrame(self)
        controls.pack(fill="x", padx=10, pady=(10, 5))

        self.model_var = ctk.StringVar(value="large-v3")
        ctk.CTkOptionMenu(controls, variable=self.model_var,
            values=["tiny", "base", "small", "medium", "large-v3",
                    "distil-large-v3", "turbo"]
        ).pack(side="left", padx=5, pady=5)

        self.lang_var = ctk.StringVar(value="en")
        ctk.CTkOptionMenu(controls, variable=self.lang_var,
            values=["en", "es", "fr", "de", "ja", "zh", "auto"]
        ).pack(side="left", padx=5, pady=5)

        self.start_btn = ctk.CTkButton(controls, text="▶ Start",
            command=self.toggle, fg_color="green")
        self.start_btn.pack(side="right", padx=5, pady=5)

        # Transcription display
        self.transcript = ctk.CTkTextbox(self, height=300, wrap="word")
        self.transcript.pack(fill="both", expand=True, padx=10, pady=5)

        # Log panel
        self.log_box = ctk.CTkTextbox(self, height=80, wrap="word")
        self.log_box.pack(fill="x", padx=10, pady=(0, 10))

        self._poll_results()

    def _poll_results(self):
        try:
            while True:
                text = self.result_queue.get_nowait()
                self.transcript.insert("end", text + "\n")
                self.transcript.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_results)  # Poll every 100ms
```

Compared alternatives: **PyQt6 6.10.2** — GPL license (viral copyleft, commercial license $550+/yr), heavy (~100+ MB), overkill. **PySide6 6.10.2** — LGPL (permissive), same weight, better license but still overkill. **Tkinter + sv-ttk 2.6.1** — works but lacks rounded corners and consistent dark theming across all widgets. **DearPyGui 2.1.1** — GPU-rendered ImGui style, doesn't look native, wrong paradigm for a desktop app.

---

## 6. Project structure and architecture

### Recommended directory layout (src layout)

The `src/` layout is the **recommended standard** per the Python Packaging User Guide, PyOpenSci, and Real Python. It prevents accidental local imports and ensures tests run against the installed package:

```
whisper-transcriber/
├── .python-version              # "3.12"
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── whisper_transcriber/
│       ├── __init__.py
│       ├── __main__.py          # Entry: `python -m whisper_transcriber`
│       ├── app.py               # Orchestrator (~100 lines)
│       ├── audio/
│       │   ├── __init__.py
│       │   └── capture.py       # Loopback capture (~120 lines)
│       ├── transcription/
│       │   ├── __init__.py
│       │   └── engine.py        # faster-whisper wrapper (~130 lines)
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── main_window.py   # GUI layout + polling (~130 lines)
│       │   └── components.py    # Reusable widgets (~100 lines)
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py      # Dataclass + JSON persistence (~80 lines)
│       └── utils/
│           ├── __init__.py
│           └── logging.py       # Logging setup (~80 lines)
├── tests/
├── logs/                        # Runtime logs (gitignored)
└── config/                      # User settings JSON
```

### Why threading, not asyncio

**Threading is the correct concurrency model** for this application for three reasons. First, Tkinter runs its own event loop — asyncio's event loop creates conflicts without complex integration. Second, the core blocking operations (`sounddevice`/`pyaudiowpatch` I/O, faster-whisper inference) are C-level calls that **release the GIL**, so other Python threads run freely during audio capture and model inference. Third, only 2–3 threads are needed (audio capture, transcription, main/GUI) — well within threading's sweet spot, with no need for asyncio's thousands-of-tasks scaling.

### Queue-based pipeline architecture

```
┌──────────────┐  audio_queue   ┌──────────────────┐  result_queue  ┌──────────────┐
│ Audio Capture │ ─────────────→│  Transcription    │ ──────────────→│  GUI (main   │
│    Thread     │  Queue(100)   │     Thread        │   Queue()     │   thread)    │
└──────────────┘                └──────────────────┘                └──────────────┘
       ↑                                ↑                                  │
       └────────────── stop_event (threading.Event) ──────────────────────┘
```

The orchestrator (`app.py`) creates shared `queue.Queue` instances and a `threading.Event` for shutdown signaling, then starts daemon threads and launches the GUI mainloop:

```python
def main():
    settings = AppSettings.load()
    audio_queue = queue.Queue(maxsize=100)  # Backpressure
    result_queue = queue.Queue()
    stop_event = threading.Event()

    capturer = AudioCapturer(audio_queue, stop_event, settings)
    engine = TranscriptionEngine(audio_queue, result_queue, stop_event, settings)

    threading.Thread(target=capturer.run, daemon=True).start()
    threading.Thread(target=engine.run, daemon=True).start()

    app = MainWindow(result_queue, stop_event, settings)
    app.mainloop()
    stop_event.set()
```

### Logging setup

The logging module should create three handlers: a `RotatingFileHandler` (5 MB max, 3 backups, UTF-8, writes to `logs/` directory), a `StreamHandler` for console output, and a custom `QueueHandler` that feeds log messages to the GUI's log panel via a queue (polled by `after()`):

```python
import logging
import logging.handlers
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

def setup_logging(gui_queue=None):
    Path("logs").mkdir(exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # File handler with rotation
    fh = logging.handlers.RotatingFileHandler(
        "logs/app.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
```

Every module uses `logger = logging.getLogger(__name__)` for hierarchical, per-module logging. Never use `print()` for diagnostics.

### Configuration with dataclasses

A `dataclass` with JSON persistence is the simplest approach for a desktop app with a handful of settings:

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class AppSettings:
    model_size: str = "large-v3"
    language: str = "en"
    compute_type: str = "float16"
    chunk_duration: float = 5.0
    audio_device: int | None = None
    theme: str = "dark"

    @classmethod
    def load(cls, path="config/settings.json"):
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path="config/settings.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))
```

---

## 7. Complete pyproject.toml reference

Combining all findings, the full `pyproject.toml` for this project:

```toml
[project]
name = "whisper-transcriber"
version = "0.1.0"
description = "Windows desktop app for real-time system audio transcription"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "faster-whisper>=1.2.1",
    "torch>=2.9.1",
    "PyAudioWPatch>=0.2.12",
    "customtkinter>=5.2.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/whisper_transcriber"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.ruff]
line-length = 120
```

---

## Version summary and key decisions

| Component             | Package                                                  | Version  | Source                                   |
| --------------------- | -------------------------------------------------------- | -------- | ---------------------------------------- |
| Transcription engine  | `faster-whisper`                                         | 1.2.1    | PyPI, GitHub SYSTRAN/faster-whisper      |
| Audio capture         | `PyAudioWPatch`                                          | 0.2.12.8 | PyPI, GitHub s0d3s/PyAudioWPatch         |
| GUI framework         | `customtkinter`                                          | 5.2.2    | PyPI, GitHub TomSchimansky/CustomTkinter |
| Deep learning runtime | `torch` (CUDA 12.8)                                      | 2.9.1    | pytorch.org                              |
| Package manager       | `uv`                                                     | 0.10.10  | docs.astral.sh/uv                        |
| Python version        | CPython                                                  | 3.12     | python.org                               |
| VAD                   | Silero VAD v6 (built into faster-whisper)                | —        | onnxruntime                              |
| Recommended model     | `large-v3` (multilingual) or `distil-large-v3` (English) | —        | Hugging Face Hub                         |
| Compute type          | `float16` (default) or `int8_float16` (VRAM-saving)      | —        | CTranslate2                              |
| RTX 4070              | Compute Capability 8.9, 12 GB VRAM                       | —        | NVIDIA                                   |

The architecture uses **three threads** (audio capture → transcription → GUI) communicating via **thread-safe queues** with a shared `threading.Event` for shutdown. The GUI polls the result queue every 100ms via Tkinter's `after()` mechanism. VAD should always be enabled (`vad_filter=True`) to prevent hallucinations during silence. The `src/` layout with modules capped at ~150 lines each keeps the codebase clean and navigable.
