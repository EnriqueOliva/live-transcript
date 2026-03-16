from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = DATA_DIR / "models"
APP_LOGS_DIR: Path = DATA_DIR / "logs"
SETTINGS_PATH: Path = DATA_DIR / "settings.json"
TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "transcripts"
SESSION_LOGS_DIR: Path = PROJECT_ROOT / "logs"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, MODELS_DIR, APP_LOGS_DIR, TRANSCRIPTS_DIR, SESSION_LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_session_paths() -> tuple[Path, Path]:
    stamp = datetime.now().strftime("[%d-%m-%y] - [%H-%M]")
    transcript_dir = TRANSCRIPTS_DIR / stamp
    log_dir = SESSION_LOGS_DIR / stamp
    transcript_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Session dirs: %s | %s", transcript_dir, log_dir)
    return transcript_dir, log_dir
