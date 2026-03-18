from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from datetime import datetime
from queue import Queue

from whisper_transcriber.io.paths import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_listener: logging.handlers.QueueListener | None = None
_file_handler: logging.FileHandler | None = None


def setup(gui_handler: logging.Handler | None = None) -> None:
    global _listener, _file_handler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stamp = datetime.now().strftime("[%d-%m-%Y] - [%H-%M-%S]")
    _file_handler = logging.FileHandler(
        LOG_DIR / f"{stamp}.txt", encoding="utf-8",
    )
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(formatter)

    class FlushHandler(logging.Handler):
        def __init__(self, target: logging.FileHandler) -> None:
            super().__init__(logging.DEBUG)
            self._target = target
            self.setFormatter(formatter)

        def emit(self, record: logging.LogRecord) -> None:
            self._target.emit(record)
            self._target.flush()

    flush_handler = FlushHandler(_file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [flush_handler, console_handler]
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


def shutdown() -> None:
    if _listener is not None:
        _listener.stop()
    if _file_handler is not None:
        _file_handler.flush()
        _file_handler.close()


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
