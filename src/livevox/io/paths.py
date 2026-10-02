from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

APP_FOLDER_NAME = "Livevox"
TRANSCRIPTS_FOLDER_NAME = "livevox-transcripts"
SESSION_STAMP_FORMAT = "[%d-%m-%y] - [%H-%M]"
FIRST_DUPLICATE_SUFFIX = 2


def _resolve_app_data_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_FOLDER_NAME
    else:
        return Path.home() / f".{APP_FOLDER_NAME.lower()}"


APP_DATA_DIR: Path = _resolve_app_data_dir()
LOG_DIR: Path = APP_DATA_DIR / "logs"
SETTINGS_PATH: Path = APP_DATA_DIR / "settings.json"
TRANSCRIPTS_DIR: Path = Path.home() / "Documents" / TRANSCRIPTS_FOLDER_NAME


def ensure_dirs() -> None:
    for directory in (APP_DATA_DIR, LOG_DIR, TRANSCRIPTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_session_paths(now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime(SESSION_STAMP_FORMAT)
    session_dir = TRANSCRIPTS_DIR / stamp
    duplicate_number = FIRST_DUPLICATE_SUFFIX
    while session_dir.exists():
        session_dir = TRANSCRIPTS_DIR / f"{stamp} ({duplicate_number})"
        duplicate_number += 1
    session_dir.mkdir(parents=True)
    logger.info("Session dir: %s", session_dir)
    return session_dir
