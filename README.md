# Live Transcript

Real-time transcription of everything your Windows PC plays (Zoom, Meet, Teams, YouTube, a lecture), on your own machine with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Text appears about a second after each sentence ends.

![PySide6](https://img.shields.io/badge/GUI-PySide6-blue)
![Python 3.12](https://img.shields.io/badge/python-3.12-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

## No word is dropped by the pipeline

The app is built so that no audio is ever thrown away on the way to the model:

- **Every captured sample is transcribed.** Audio is cut into pieces only inside pauses, the pieces cover the stream end to end, and every piece goes to the model. Only stretches of exact digital silence are skipped. Nothing is skipped because the PC is busy: pieces wait in a queue until they are done.
- **Cuts never split a word.** The cut goes to the quietest moment of a pause found by Silero VAD. If someone talks for 22 seconds without a single pause, the next piece starts 3 seconds earlier and word timestamps decide where one piece stops and the next begins, leaning toward repeating a word rather than losing one.
- **Nothing inside faster-whisper may discard text.** Its silence skip, its own VAD and its timestamp filtering are off.
- **Sentences the model stops reading halfway are recovered.** Whisper sometimes ends a piece early and drops the rest of it. Decoding resumes from the last word for as long as the VAD is confident speech remains. The only text ever thrown away is a resumed fragment too long to fit the audio it covers (over 40 letters per second, like a "¡Gracias!" squeezed into 50 ms), the signature of Whisper's silence hallucinations.
- **Doubtful text is kept, never deleted.** Text from stretches with no detected speech, and low-confidence resumed fragments, are shown dimmed and marked `(?)` in the timestamped file.
- **The audio is saved.** Every session writes `recording.wav`, updated every second, so even a crash or a failed piece can be transcribed again later.
- **Every session is audited.** `session_report.txt` states how much audio was captured and how much was processed (100% unless something went wrong, and then it says what).

What it cannot promise: the model itself can still mishear a word, like any speech recognizer. When that matters, the recording is there to check.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```powershell
git clone git@github.com:EnriqueOliva/live-transcript.git
cd live-transcript
uv sync                 # any PC, CPU only
uv sync --group cuda    # NVIDIA GPU: adds cuBLAS (about 800 MB), no PyTorch or cuDNN needed
```

If [Sotvox](https://github.com/EnriqueOliva/sotvox) is installed with GPU acceleration enabled, its CUDA libraries are reused automatically and `--group cuda` is not needed.

The speech model (about 1.5 GB) downloads on first use and is shared with Sotvox.

## Use

```powershell
uv run python -m whisper_transcriber
```

1. Pick the model (`turbo` is the default) and the language (Spanish by default, or `Auto`)
2. Tick **Mic** to also transcribe your own voice
3. Click **Start**
4. Click **Stop**. The app keeps transcribing until the last word is written, then plays a sound. Closing the window does the same.

The grey line at the bottom is the sentence still being spoken. It is replaced by the final text when the sentence ends.

If the default output device changes (headphones plugged in, Bluetooth connecting), capture follows it automatically.

## Output

Each session gets its own folder in `Documents\live-transcripts\[DD-MM-YY] - [HH-MM]\`:

| File | Content |
| --- | --- |
| `transcript.txt` | the plain transcript |
| `transcript_with_timestamps.txt` | `[HH:MM:SS -> HH:MM:SS] text`, doubtful lines marked `(?)`, device changes noted |
| `recording.wav` | the full session audio (16 kHz mono, about 115 MB per hour) |
| `session_report.txt` | coverage audit: captured vs processed audio, forced cuts, failures |

Logs and settings live in `%LOCALAPPDATA%\LiveTranscript`.

## Transcribe a recording or any file

The same pipeline runs on a file:

```powershell
uv run python -m whisper_transcriber --transcribe-file "path\to\recording.wav" --language es
```

## How it works

```
WASAPI loopback ─┐                                    ┌─► recording.wav
(+ silent         ├─► 16 kHz mono ─► mixer ───────────┤
 keep-alive)      │   (stateful                       └─► Silero VAD ─► pieces cut in pauses
Microphone (opt) ─┘    resampling)                                         │
                                                                           ▼
                                   transcript files + window ◄── faster-whisper worker
```

- Capture: `PyAudioWPatch` WASAPI loopback. A silent stream keeps the loopback running during silence, the same trick OBS uses.
- Conversion: channels averaged, resampled with a stateful PyAV resampler, so block boundaries leave no clicks or gaps.
- Segmentation: streaming Silero VAD (the model bundled with faster-whisper). A pause of 0.5 s ends a sentence, shorter breaths are accepted as a sentence gets long, 22 s is the hard limit.
- Transcription: `turbo` (large-v3-turbo), beam 5, temperature fallback, word timestamps, and the previous sentence's words as context.

## Measuring it

`scripts/evaluate_recall.py` replays a recording through the full pipeline (48 kHz capture, conversion, segmentation, model) and compares the result word by word with a reference transcript:

```powershell
uv run python scripts/evaluate_recall.py meeting.mp4 reference.txt
```

On a 30-minute Spanish meeting, measured against a full-file transcription of the same recording:

| | Previous version (30 s chunks) | This version |
| --- | --- | --- |
| Audio processed | not tracked | 100% |
| Reference words found | 84.2% | 90.7% |
| Missed stretches of 4+ words | 31 (301 words, the longest a whole 35-word sentence) | 8 (50 words, mostly repetitions like "sí sí sí sí") |

`scripts/live_smoke_test.py` runs the real capture on whatever is playing for 40 seconds and prints the session report.

## Development

```powershell
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
```

## Legal note

Make sure recording is allowed by your institution and by the terms of the conferencing software you use.

## License

[MIT](LICENSE)
