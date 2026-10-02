from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from livevox.audio.timeline import samples_to_seconds

PERCENT = 100.0


@dataclass
class SessionReport:
    model: str = ""
    device: str = ""
    language: str = ""
    captured_samples: int = 0
    covered_samples: int = 0
    gap_samples: int = 0
    transcribed_samples: int = 0
    silent_samples: int = 0
    failed_samples: int = 0
    pieces: int = 0
    forced_cuts: int = 0
    failed_pieces: int = 0
    uncertain_lines: int = 0
    overflow_events: int = 0
    device_switches: int = 0
    recorded_samples: int = 0
    recording_path: str = ""
    recording_failed: bool = False
    model_failed: bool = False
    devices: list[str] = field(default_factory=list)
    finished_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def coverage_percent(self) -> float:
        if self.captured_samples == 0:
            return PERCENT
        else:
            return PERCENT * min(self.covered_samples, self.captured_samples) / self.captured_samples

    @property
    def is_complete(self) -> bool:
        return (
            self.gap_samples == 0
            and self.covered_samples >= self.captured_samples
            and self.failed_pieces == 0
            and not self.model_failed
            and self.overflow_events == 0
        )

    def summary(self) -> str:
        captured = samples_to_seconds(self.captured_samples)
        if self.is_complete:
            return f"Session complete: {captured:.0f}s captured, 100% of the audio was transcribed"
        else:
            problems = []
            if self.model_failed:
                problems.append("the model could not be loaded")
            if self.failed_pieces:
                problems.append(f"{self.failed_pieces} piece(s) could not be transcribed")
            if self.gap_samples:
                problems.append(f"{samples_to_seconds(self.gap_samples):.2f}s were never processed")
            if self.overflow_events:
                problems.append(f"Windows reported {self.overflow_events} capture overflow(s)")
            details = ", ".join(problems) or "coverage incomplete"
            return f"Session finished with problems ({details}). The audio is saved in {self.recording_path}"

    def to_text(self) -> str:
        lines = [
            "Livevox session report",
            f"Finished: {self.finished_at}",
            f"Result: {'COMPLETE' if self.is_complete else 'INCOMPLETE'}",
            "",
            f"Model: {self.model} on {self.device}",
            f"Language: {self.language}",
            f"Audio devices: {', '.join(self.devices) or 'unknown'}",
            "",
            f"Captured audio: {samples_to_seconds(self.captured_samples):.2f}s",
            f"Processed audio: {samples_to_seconds(self.covered_samples):.2f}s ({self.coverage_percent:.3f}%)",
            f"Never processed: {samples_to_seconds(self.gap_samples):.2f}s",
            f"Sent to the model: {samples_to_seconds(self.transcribed_samples):.2f}s",
            f"Skipped as digital silence: {samples_to_seconds(self.silent_samples):.2f}s",
            f"Failed to transcribe: {samples_to_seconds(self.failed_samples):.2f}s in {self.failed_pieces} piece(s)",
            f"Pieces: {self.pieces} ({self.forced_cuts} forced cut(s) without a pause)",
            f"Low-confidence lines kept and marked (?): {self.uncertain_lines}",
            f"Capture overflows reported by Windows: {self.overflow_events}",
            f"Audio device switches: {self.device_switches}",
            "",
            f"Recording: {self.recording_path or 'not recorded'}",
            f"Recorded audio: {samples_to_seconds(self.recorded_samples):.2f}s"
            + (" (recording had write errors)" if self.recording_failed else ""),
        ]
        return "\n".join(lines) + "\n"
