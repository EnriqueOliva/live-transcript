from __future__ import annotations

import argparse
import difflib
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from livevox.audio.capture import CaptureData, CaptureFinished, CaptureFormat
from livevox.audio.pipeline import AudioPipeline
from livevox.audio.timeline import SAMPLE_RATE
from livevox.io.recording import WavRecorder
from livevox.io.transcript_writer import TranscriptWriter
from livevox.offline import decode_media, transcribe_samples
from livevox.session.events import LoggingEvents
from livevox.session.messages import PipelineStatistics
from livevox.stt.cuda_runtime import register_library_directories
from livevox.stt.segmenter import SegmenterConfig, SpeechSegmenter
from livevox.stt.vad import StreamingVad
from livevox.stt.whisper_engine import WhisperEngine
from livevox.stt.worker import DecodingOptions, TranscriptionWorker

WORD_PATTERN = re.compile(r"[\w']+", re.UNICODE)
CAPTURE_SAMPLE_RATE = 48000
CAPTURE_CHANNELS = 2
CAPTURE_BLOCK_FRAMES = 2048
MINIMUM_REPORTED_RUN = 4


class QuietEvents(LoggingEvents):
    def transcript_line(self, line: object) -> None:
        pass


def normalize_words(text: str) -> list[str]:
    return [word.lower() for word in WORD_PATTERN.findall(text)]


def compare(reference_text: str, hypothesis_text: str) -> dict:
    reference = normalize_words(reference_text)
    hypothesis = normalize_words(hypothesis_text)
    matcher = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    missing_runs = []
    for tag, reference_start, reference_end, hypothesis_start, hypothesis_end in matcher.get_opcodes():
        if tag in ("delete", "replace") and reference_end - reference_start >= MINIMUM_REPORTED_RUN:
            missing_runs.append({
                "reference_index": reference_start,
                "reference": " ".join(reference[reference_start:reference_end]),
                "hypothesis": " ".join(hypothesis[hypothesis_start:hypothesis_end]),
            })
    return {
        "reference_words": len(reference),
        "hypothesis_words": len(hypothesis),
        "matched": matched,
        "recall": matched / max(1, len(reference)),
        "precision": matched / max(1, len(hypothesis)),
        "missing_runs": missing_runs,
    }


def capture_messages(samples: np.ndarray) -> list[object]:
    from av.audio.frame import AudioFrame
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="s16", layout="stereo", rate=CAPTURE_SAMPLE_RATE)
    frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = SAMPLE_RATE
    converted = [part.to_ndarray().reshape(-1) for part in resampler.resample(frame) + resampler.resample(None)]
    interleaved = np.concatenate(converted).astype(np.int16)
    messages: list[object] = [CaptureFormat("loopback", 1, CAPTURE_SAMPLE_RATE, CAPTURE_CHANNELS, "simulated")]
    block = CAPTURE_BLOCK_FRAMES * CAPTURE_CHANNELS
    messages.extend(
        CaptureData("loopback", 1, interleaved[start : start + block].tobytes(), 0)
        for start in range(0, interleaved.size, block)
    )
    messages.append(CaptureFinished())
    return messages


def run_capture_path(samples: np.ndarray, engine: WhisperEngine, output_dir: Path, language: str,
                     config: SegmenterConfig, decoding: DecodingOptions) -> object:
    raw_queue: queue.SimpleQueue = queue.SimpleQueue()
    piece_queue: queue.Queue = queue.Queue()
    statistics = PipelineStatistics()
    segmenter = SpeechSegmenter(StreamingVad(), config)
    recorder = WavRecorder(output_dir / "recording.wav")
    recorder.open()
    writer = TranscriptWriter(output_dir)
    writer.open()
    events = QuietEvents()
    pipeline = AudioPipeline(raw_queue, piece_queue, segmenter, recorder, events, statistics)
    worker = TranscriptionWorker(
        piece_queue, engine, writer, events, language,
        captured_samples=lambda: statistics.captured_samples,
        report_path=output_dir / "session_report.txt", partials=False, decoding=decoding,
    )
    threads = [threading.Thread(target=pipeline.run), threading.Thread(target=worker.run)]
    for thread in threads:
        thread.start()
    for message in capture_messages(samples):
        raw_queue.put(message)
    for thread in threads:
        thread.join()
    return worker.report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure how many reference words the live pipeline keeps")
    parser.add_argument("media", type=Path)
    parser.add_argument("reference", type=Path, help="reference transcript (plain text) of the same media")
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="es")
    parser.add_argument("--path", choices=("capture", "offline"), default="capture",
                        help="capture: 48 kHz stereo through conversion, mixing and recording; offline: 16 kHz direct")
    parser.add_argument("--hard-maximum", type=float, help="override the forced-cut length to stress the handoff")
    parser.add_argument("--limit", type=float, help="only use the first N seconds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", type=float, default=DecodingOptions().context_seconds,
                        help="seconds of preceding audio given to the model as context")
    parser.add_argument("--prompt", action=argparse.BooleanOptionalAction, default=DecodingOptions().prompt_previous_text,
                        help="prompt the model with the previous committed words")
    arguments = parser.parse_args()

    register_library_directories()
    samples = decode_media(arguments.media)
    if arguments.limit:
        samples = samples[: int(arguments.limit * SAMPLE_RATE)]
    config = SegmenterConfig()
    if arguments.hard_maximum:
        config = replace(config, hard_maximum_seconds=arguments.hard_maximum,
                         forced_cut_search_seconds=min(config.forced_cut_search_seconds, arguments.hard_maximum / 2))
    output_dir = arguments.output or Path(tempfile.mkdtemp(prefix="livevox-eval-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = WhisperEngine(arguments.model)
    decoding = DecodingOptions(context_seconds=arguments.context, prompt_previous_text=arguments.prompt)
    print(f"Decoding: {decoding}")
    started = time.perf_counter()
    if arguments.path == "capture":
        report = run_capture_path(samples, engine, output_dir, arguments.language, config, decoding)
    else:
        report = transcribe_samples(samples, engine, output_dir, arguments.language, QuietEvents(), config, decoding)
    elapsed = time.perf_counter() - started
    hypothesis = (output_dir / "transcript.txt").read_text(encoding="utf-8")
    reference = arguments.reference.read_text(encoding="utf-8")
    result = compare(reference, hypothesis)
    print(report.to_text())  # type: ignore[attr-defined]
    print(f"Audio: {samples.size / SAMPLE_RATE:.1f}s processed in {elapsed:.1f}s")
    print(f"Reference words: {result['reference_words']}  Hypothesis words: {result['hypothesis_words']}")
    print(f"Recall: {result['recall']:.4f}  Precision: {result['precision']:.4f}")
    print(f"Runs of {MINIMUM_REPORTED_RUN}+ reference words not matched: {len(result['missing_runs'])}")
    for run in result["missing_runs"]:
        print(f"  @word {run['reference_index']}: REF «{run['reference']}»  HYP «{run['hypothesis']}»")
    print(f"Transcript: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
