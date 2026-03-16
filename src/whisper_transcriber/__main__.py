from __future__ import annotations

import os
import site
import sys
from pathlib import Path


def _ensure_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    packages = site.getsitepackages()
    if not packages:
        return
    site_packages = Path(packages[0])
    for subdir in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        d = site_packages / subdir
        if d.exists():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    _ensure_cuda_dlls()

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
    window.show()

    exit_code = qt_app.exec()
    app.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
