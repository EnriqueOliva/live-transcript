from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

_GUI_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
_GUI_DATE_FORMAT = "%H:%M:%S"
_MAX_MESSAGE_LENGTH = 300


class _Emitter(QObject):
    log_record = Signal(str)


class GuiBridge(logging.Handler):
    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._emitter = _Emitter()
        self._formatter = logging.Formatter(_GUI_FORMAT, datefmt=_GUI_DATE_FORMAT)

    @property
    def signal(self) -> Signal:
        return self._emitter.log_record

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self._formatter.format(record)
            if len(message) > _MAX_MESSAGE_LENGTH:
                message = message[:_MAX_MESSAGE_LENGTH] + "..."
            if record.exc_info and record.exc_info[1] is not None:
                message = message.split("\n")[0]
            self._emitter.log_record.emit(message)
        except RuntimeError:
            pass
