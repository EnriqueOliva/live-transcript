from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class TranscriptWriter:
    def __init__(self, session_dir: Path) -> None:
        self._session_dir = session_dir
        self._plain_path = session_dir / "transcript.txt"
        self._stamped_path = session_dir / "transcript_with_timestamps.txt"
        self._plain: TextIO | None = None
        self._stamped: TextIO | None = None
        self._has_error = False

    @property
    def has_error(self) -> bool:
        return self._has_error

    def open(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._plain = open(self._plain_path, "a", encoding="utf-8")  # noqa: SIM115
        self._stamped = open(self._stamped_path, "a", encoding="utf-8")  # noqa: SIM115
        logger.info("Transcript files opened in %s", self._session_dir)

    def write_segment(self, text: str, start: float, end: float) -> None:
        if self._plain is None or self._stamped is None:
            logger.error("TranscriptWriter not opened")
            return
        stripped = text.strip()
        if not stripped:
            return
        try:
            self._plain.write(stripped + "\n")
            self._flush(self._plain)
            ts_start = _format_timestamp(start)
            ts_end = _format_timestamp(end)
            self._stamped.write(f"[{ts_start} -> {ts_end}] {stripped}\n")
            self._flush(self._stamped)
        except OSError:
            self._has_error = True
            logger.exception("Failed to write transcript segment")

    def _flush(self, handle: TextIO) -> None:
        try:
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            logger.warning("fsync failed, data may not be durable")

    def close(self) -> None:
        for handle in (self._plain, self._stamped):
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    logger.warning("Error closing transcript file")
        self._plain = None
        self._stamped = None
        self._has_error = False
        logger.info("Transcript files closed")

    def __enter__(self) -> TranscriptWriter:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
