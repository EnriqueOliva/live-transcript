# Handoff: turn Livevox into Sotvox's twin

Written 2026-10-02 so the next session can start cold. Read it top to bottom before touching code.

## 1. The task

Refactor Livevox so it is **almost exactly equal to [Sotvox](https://github.com/EnriqueOliva/sotvox)** (local clone `C:\workshops\mine\sotvox`). The only difference between the two apps must be **what they do**:

- **Sotvox** transcribes audio and video files you drop on it, in batch.
- **Livevox** transcribes what the PC is playing (and optionally the microphone), live.

Everything else should match Sotvox: the look, the code layout, how it is installed, how it is built, how it is tested, where it writes, how the README reads. The owner confirmed this reading on 2026-10-02 and asked to do it later. Nothing of the refactor has started yet.

## 2. Where Livevox stands today

Repo `EnriqueOliva/livevox` (public, renamed from `live-transcript` on 2026-10-02), branch `main`. Local clone `C:\workshops\mine\livevox`.

What it is now:

- Python 3.12 package `src/livevox/` run with `uv run python -m livevox`, a **PySide6** dark window, dependencies locked in `pyproject.toml` + `uv.lock`.
- A transcription pipeline built so **no audio is dropped on the way to the model**. This is the part that must survive the refactor untouched in behavior:

| Module | What it guarantees |
| --- | --- |
| `audio/capture.py` | WASAPI loopback via PyAudioWPatch, a silent keep-alive render stream so loopback keeps flowing in silence, optional microphone, reconnects when the Windows default output changes (polled through COM in `audio/devices.py`), restarts dead streams |
| `audio/conversion.py` | int16 to 16 kHz mono float, channels averaged, stateful PyAV resampler (output is identical whatever the block size) |
| `audio/mixer.py` | mixes loopback and microphone without ever dropping a sample (bounded skew, then pass-through) |
| `audio/pipeline.py` | capture messages to converter, mixer, `recording.wav`, levels and segmenter, always ends with `EndOfStream` even if something breaks |
| `stt/vad.py` | stateful streaming Silero VAD using the ONNX model bundled in faster-whisper (identical to its batch output) |
| `stt/segmenter.py` | cuts only inside pauses, the pieces cover the stream end to end (asserted), forced cut at 22 s with a 3 s overlap, long non-speech emitted in 20 s pieces, digital silence flagged |
| `stt/handoff.py` | after a forced cut, word timestamps pick the boundary, biased to repeating a word rather than losing one |
| `stt/whisper_engine.py` | faster-whisper with every text-dropping feature off (`no_speech_threshold=None`, `vad_filter=False`, `without_timestamps=True`), word timestamps, lazy segment iteration that resumes decoding only while confident speech (VAD p >= 0.5) remains 0.2 s past the last word, discards only continuations over 40 letters/s |
| `stt/worker.py` | never skips a piece, retries 3 times with GPU to CPU fallback, writes a placeholder on failure, previous 40 words as prompt, grey partial line when idle on GPU, coverage accounting into `session_report.txt` |
| `stt/cuda_runtime.py` | GPU only when a CUDA device exists AND `cublas64_12.dll` loads, also looks in `%LOCALAPPDATA%\Sotvox\cuda` |
| `io/recording.py` | crash-safe `recording.wav` (header refreshed every second, fsync every 5 s) |
| `offline.py` | `--transcribe-file` runs the same pipeline on any file |

- Outputs: `Documents\livevox-transcripts\[DD-MM-YY] - [HH-MM]\` with `transcript.txt`, `transcript_with_timestamps.txt` (doubtful lines marked `(?)`, device changes noted), `recording.wav`, `session_report.txt`. Settings and logs in `%LOCALAPPDATA%\Livevox`.
- Tests: 380 in `tests/`, lint (ruff), types (mypy) and CI on `windows-latest` are green. A mutation pass broke each of 21 guarantees on purpose and every one was caught by a test.
- Measured on a 30-minute Spanish meeting against a full-file transcription of the same recording: 100% of the audio processed, 90.7% of reference words found (the old 30 s chunk pipeline found 84.2% and dropped a whole 35-word sentence), 22 low-confidence lines, 74 s of compute for 30 min on an RTX 4070.

## 3. Sotvox, the reference to copy

Sotvox v1.3.0 (`main` at `ce2c018`). Its committed files:

| Sotvox file | Role |
| --- | --- |
| `src/constants.py` | `APP_NAME`, `IS_FROZEN`, `RESOURCE_DIR` (`sys._MEIPASS` when frozen), `APP_DATA_DIR = %LOCALAPPDATA%\Sotvox`, `DEFAULT_OUTPUT_DIR = Documents\sotvox-transcripts`, `LOG_DIR`, `CUDA_DIR`, `ASSETS_DIR`, `ICON_PATH`, `LANG_MAP` |
| `src/engine.py` | transcription functions (`transcribe_audio`, multilingual, `probe_file`, `save_transcript`) |
| `src/gpu_pack.py` | "Enable GPU Acceleration": finds the newest `win_amd64` wheels of `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` (major 9) on PyPI, downloads with progress and cancel, extracts the DLLs into `%LOCALAPPDATA%\Sotvox\cuda`, `detect_nvidia_gpu()` via `nvidia-smi`, `libraries_available()` via `ctypes.WinDLL` |
| `src/main.py` | DPI awareness, registers the CUDA dir, flags `--transcribe <folder>` (headless batch, report in `logs\batch.txt`, exit codes 0/1/2) and `--selftest <media>` (report in `logs\selftest.txt`), then `ui.SotvoxApp().run()` |
| `src/ui.py` (1500 lines) | the whole Tkinter UI in authentic Windows 95 style: borderless window with its own navy title bar, File/Edit/Help menus, bevelled group boxes (`_bevel`, `_group`), custom buttons (`_mk_button`), combos, checkboxes, scrollbar (`_Win95Scrollbar`), block progress bar, status bar with resize grip, modal dialogs, About box, DPI scaling (`self.S`, `px()`), taskbar icon fix, drag-and-drop via `tkinterdnd2`, `winsound.MessageBeep` for done/failed, a session log (`_slog`) flushed to `%LOCALAPPDATA%\Sotvox\logs\[dd-mm-YYYY] - [HH-MM-SS].txt` with a system header (OS, versions, GPU, RAM) |
| `installer/sotvox.spec` | PyInstaller onedir, `collect_all` for `faster_whisper`, `ctranslate2`, `av`, `tkinterdnd2`, excludes `nvidia` and `torch`, filters any CUDA DLL out of the bundle, icon and `version_info.txt` |
| `installer/build.ps1` | freezes, fails if a CUDA library leaked into the bundle, prints the size, compiles `sotvox.iss` with Inno Setup 6 |
| `installer/sotvox.iss` | Inno Setup: `AppId` GUID, version, Program Files install, Start Menu and optional desktop shortcut, `PrivilegesRequired=admin` with the dialog override, lzma2, output `Sotvox-Setup.exe` at the repo root |
| `installer/version_info.txt` | Windows version resource |
| `setup/setup.ps1` + `setup.vbs` | developer environment: installs uv, Python 3.11, a `.venv`, then `uv pip install faster-whisper tkinterdnd2 pyinstaller` (+ CUDA wheels if an NVIDIA GPU is present) |
| `launch.vbs` | runs `src\main.py` with `pythonw` from the dev venv |
| `assets/` | `sotvox.ico`, `sotvox_16/32/48.png`, `screenshot.png` |
| `automation/auto-transcribe/` | README and an OBS watch script for the headless mode |
| `README.md` | short, for people who only want to install and use it: Install, Use it, Options table, Good to know, License |

Also in the Sotvox working tree but **not committed** as of 2026-10-01 (check `git status` there before relying on it): `pyproject.toml` (pytest, coverage and ruff config only), `requirements-dev.txt`, `.github/workflows/ci.yml` (Python 3.11 + pip, `ruff check src tests`, `pytest -m "not gui"` with coverage), and `tests/` (19 files: `conftest.py` with an autouse fixture that blocks the network, `doubles.py` with `FakeWhisperModel` and Tk doubles that run real `SotvoxApp` methods on an app whose constructor was skipped, markers `gui`, `integration`, `slow`, `allow_network`). One of those tests (`test_engine_output.py`) leaks about 25 junk `.txt` files into the Sotvox repo root.

Sotvox code style: plain Python without type hints, module-level functions in `engine.py`, one big `ui.py` class, constants in `constants.py`, `sys.path` insert in `main.py` when not frozen.

## 4. The plan

Mirror Sotvox file for file and keep Livevox's pipeline behind it.

1. **Layout.** Flat `src/` like Sotvox: `constants.py`, `engine.py`, `gpu_pack.py`, `main.py`, `ui.py`, plus the live modules that have no Sotvox counterpart (capture, conversion, mixer, pipeline, vad, segmenter, handoff, worker, recording, report). Decide whether those stay as a small subpackage or go flat. Sotvox is flat, so lean flat.
2. **UI.** Rebuild the window in Tkinter with Sotvox's Win95 widgets (copy the helpers from Sotvox's `ui.py`, do not reinvent them). Same title bar, menus, group boxes, buttons, status bar, dialogs, About box, DPI scaling, icon handling, sounds. PySide6 goes away. Livevox needs: Start/Stop, Language, Model, Device, a Mic checkbox, output folder, a live transcript area (Tk `Text` with tags for normal, uncertain (dim italic), failure and the grey partial line that is replaced in place, auto-scroll only when already at the bottom), a level meter in Win95 style where Sotvox has its progress bar, the Log group, and a status bar with lag ("Xs behind"). Keep the current behaviors: Stop and window close wait until the last word is written, a second close asks before abandoning, errors stop the session through the normal path.
3. **Constants and paths.** `APP_NAME = "Livevox"`, `%LOCALAPPDATA%\Livevox` for logs, settings and `cuda`, `Documents\livevox-transcripts`, frozen-aware `RESOURCE_DIR`.
4. **GPU.** Port Sotvox's `gpu_pack.py` and its dialog, but download only `nvidia-cublas-cu12` (CTranslate2 >= 4.6.3 does not need cuDNN, verified: its `ctranslate2.dll` imports only `cublas64_12.dll`). Keep looking in `%LOCALAPPDATA%\Sotvox\cuda` too, so a machine that already has Sotvox's pack needs no download.
5. **main.py flags.** `--selftest <media>` like Sotvox. Map the current `--transcribe-file` to Sotvox's naming (`--transcribe`) or keep both. Livevox has no headless batch use case beyond re-transcribing a recording.
6. **Packaging.** `installer/livevox.spec` (add `pyaudiowpatch` and `onnxruntime` to `collect_all`, keep the faster-whisper assets because the VAD model `silero_vad_v6.onnx` lives there), `installer/build.ps1`, `installer/livevox.iss` with a **new AppId GUID** (never reuse Sotvox's), `installer/version_info.txt`, output `Livevox-Setup.exe`, publish it on GitHub Releases like Sotvox.
7. **Dev setup.** `setup/setup.ps1`, `setup/setup.vbs`, `launch.vbs` like Sotvox.
8. **Assets.** A Livevox icon set (`livevox.ico`, 16/32/48 png) in Sotvox's visual style, and a screenshot for the README.
9. **Tests and CI.** Move the suite to Sotvox's test conventions (`conftest.py`, `doubles.py`, markers, network blocked by default, `requirements-dev.txt`, CI with `pytest -m "not gui"`). Port every existing test, especially the lossless ones (segmenter coverage over random scenes, handoff no-word-lost property, streaming conversion and VAD equivalence, mixer sum preservation, worker never skipping, capture device switch, pipeline always ending). Re-run the mutation check (each guarantee broken on purpose must fail a test). UI tests need the Tk-double approach Sotvox uses.
10. **Docs.** README in Sotvox's install-and-use style, keeping one short section on the no-lost-words guarantee and its honest limit (the model can still mishear). Drop `SETUP.md` if Sotvox has no equivalent. Keep `scripts/evaluate_recall.py` and `scripts/live_smoke_test.py` working.

## 5. Decisions

Made:

- Name **Livevox** everywhere (done 2026-10-02: repo, package `livevox`, window, folders).
- Keep the whole lossless pipeline and its defaults as they are (they were measured, see section 2).

Still open, ask the owner:

- **Window contents.** The proposal in section 4.2 was put to the owner and is not confirmed yet.
- **Python 3.11 like Sotvox, or 3.12.** Twin points to 3.11.
- **Dependencies.** Sotvox has no lock file (`uv pip install` in `setup.ps1`). Livevox has `pyproject.toml` + `uv.lock`. Matching Sotvox exactly means dropping the lock, so confirm.
- **Version numbering.** Sotvox is 1.3.0. Livevox could start at 1.0.0.

## 6. Do not break

- The coverage guarantee: every captured sample ends in a piece, every piece reaches the worker, `session_report.txt` says 100% when nothing failed.
- No filter that throws away model text, except the 40 letters/s continuation rule. Doubtful text is kept and marked `(?)`.
- Stop and close wait for the last word.
- How to check after the refactor:
  - `pytest`, ruff, and the mutation script idea (copy the repo, break each guarantee, every break must fail a test).
  - `scripts/evaluate_recall.py <recording> <reference.txt>` on a real meeting recording, using Sotvox's transcript of the same file as the reference. Recall should stay around 90% on the same file.
  - `scripts/live_smoke_test.py` for 40 s of real capture. It records whatever is playing, so delete its output afterwards.

## 7. Pitfalls already paid for

- faster-whisper with `without_timestamps=True` alone sometimes stops mid-piece and loses the rest. With `word_timestamps=True` it resumes from the last word but cascades "Gracias." hallucinations over trailing silence (one 9 s piece took 13 passes). The lazy-iteration gate in `whisper_engine.py` is the fix, do not "simplify" it into `list(segments)`.
- `no_speech_prob` is always about 0 in this decoding mode, so it cannot be used to spot hallucinations.
- A game holding the GPU makes transcription 5 to 10 times slower and can look like a hang.
- Mutating source files while background processes import them corrupts those runs, mutate a copy.
- `astral-sh/setup-uv` has no floating major tag, pin a full version (`v10.2.0` today).
- PyAV's own stereo-to-mono downmix uses a different gain, average the channels before resampling.

## 8. Related follow-ups outside Livevox

- Sotvox's GPU pack downloads about 950 MB of cuDNN it no longer needs.
- Sotvox's test suite, CI and `pyproject.toml` are uncommitted, and one test leaks junk `.txt` files into its repo root.
