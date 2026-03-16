from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from queue import Queue

from whisper_transcriber.io.paths import APP_LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

_listener: logging.handlers.QueueListener | None = None
_session_handler: logging.FileHandler | None = None


def setup(gui_handler: logging.Handler | None = None) -> None:
    global _listener

    APP_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        APP_LOGS_DIR / "app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [file_handler, console_handler]
    if gui_handler is not None:
        handlers.append(gui_handler)

    log_queue: Queue[logging.LogRecord] = Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(queue_handler)

    _listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    _listener.start()

    sys.excepthook = _handle_exception
    threading.excepthook = _handle_thread_exception

    logging.getLogger(__name__).debug("Logging initialised")


def add_session_handler(session_dir: Path) -> None:
    global _session_handler
    if _listener is None:
        return
    session_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    _session_handler = logging.FileHandler(
        session_dir / "session.log", encoding="utf-8"
    )
    _session_handler.setLevel(logging.DEBUG)
    _session_handler.setFormatter(formatter)
    _listener.handlers = (*_listener.handlers, _session_handler)


def remove_session_handler() -> None:
    global _session_handler
    if _listener is None or _session_handler is None:
        return
    _listener.handlers = tuple(
        h for h in _listener.handlers if h is not _session_handler
    )
    _session_handler.flush()
    _session_handler.close()
    _session_handler = None


def shutdown() -> None:
    if _listener is not None:
        _listener.stop()


def _handle_exception(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logging.getLogger("unhandled").critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )


def _handle_thread_exception(args):  # type: ignore[no-untyped-def]
    if args.exc_type is SystemExit:
        return
    logging.getLogger("unhandled.thread").critical(
        "Uncaught thread exception in %s",
        args.thread.name if args.thread else "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
