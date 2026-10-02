from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineStatistics:
    captured_samples: int = 0
    overflow_events: int = 0
    device_switches: int = 0
    recording_path: Path | None = None
    recorded_samples: int = 0
    recording_failed: bool = False
    devices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Notice:
    sample: int
    text: str


@dataclass(frozen=True)
class EndOfStream:
    statistics: PipelineStatistics
