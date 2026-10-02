from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class LineStyle(enum.Enum):
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    NOTICE = "notice"
    FAILURE = "failure"


@dataclass(frozen=True)
class TranscriptLine:
    text: str
    start: float
    end: float
    style: LineStyle


class SessionEvents(Protocol):
    def status(self, state: str, detail: str = "") -> None: ...

    def message(self, text: str) -> None: ...

    def transcript_line(self, line: TranscriptLine) -> None: ...

    def partial_text(self, text: str) -> None: ...

    def progress(self, captured_seconds: float, transcribed_seconds: float) -> None: ...

    def audio_levels(self, levels: list[float]) -> None: ...

    def notice(self, text: str) -> None: ...

    def error(self, text: str) -> None: ...

    def finished(self, summary: str) -> None: ...


class LoggingEvents:
    def status(self, state: str, detail: str = "") -> None:
        logger.info("Status: %s %s", state, detail)

    def message(self, text: str) -> None:
        logger.info(text)

    def transcript_line(self, line: TranscriptLine) -> None:
        print(f"[{line.start:8.2f} -> {line.end:8.2f}] {line.text}", flush=True)

    def partial_text(self, text: str) -> None:
        logger.debug("Partial: %s", text)

    def progress(self, captured_seconds: float, transcribed_seconds: float) -> None:
        logger.debug("Progress: %.1fs captured, %.1fs transcribed", captured_seconds, transcribed_seconds)

    def audio_levels(self, levels: list[float]) -> None:
        pass

    def notice(self, text: str) -> None:
        logger.warning(text)

    def error(self, text: str) -> None:
        logger.error(text)

    def finished(self, summary: str) -> None:
        logger.info(summary)
