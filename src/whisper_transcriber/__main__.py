from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from whisper_transcriber.stt.cuda_runtime import register_library_directories


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="whisper_transcriber", description="Live Transcript")
    parser.add_argument(
        "--transcribe-file",
        type=Path,
        help="transcribe an audio or video file (for example a session recording.wav) and exit",
    )
    parser.add_argument("--model", default="turbo")
    parser.add_argument("--language", default="es", help="language code, or Auto")
    parser.add_argument("--output", type=Path, help="folder for the transcript files")
    parser.add_argument("--compute-type", default="auto")
    return parser.parse_args(arguments)


def _run_file_transcription(arguments: argparse.Namespace) -> int:
    from whisper_transcriber.offline import transcribe_file

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    media_path: Path = arguments.transcribe_file
    output_dir = arguments.output or media_path.parent / f"{media_path.stem}-transcript"
    return transcribe_file(media_path, arguments.model, arguments.language, output_dir, arguments.compute_type)


def _run_gui() -> int:
    from PySide6.QtWidgets import QApplication

    from whisper_transcriber.bootstrap import Application
    from whisper_transcriber.ui.main_window import MainWindow
    from whisper_transcriber.ui.theme import apply_dark_theme

    qt_app = QApplication(sys.argv)
    apply_dark_theme(qt_app)

    app = Application()
    window = MainWindow(app.worker_signals, app.gui_bridge, app.settings)
    window.start_requested.connect(app.start_session)
    window.stop_requested.connect(app.stop_session)
    window.open_folder_requested.connect(app.open_session_folder)
    window.session_ended.connect(app.finalize_session)
    window.show()

    exit_code = qt_app.exec()
    app.shutdown()
    return exit_code


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    register_library_directories()
    arguments = _parse_arguments(sys.argv[1:])
    if arguments.transcribe_file is not None:
        sys.exit(_run_file_transcription(arguments))
    else:
        sys.exit(_run_gui())


if __name__ == "__main__":
    main()
