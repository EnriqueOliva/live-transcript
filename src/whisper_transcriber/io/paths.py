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
LOG_DIR: Path = PROJECT_ROOT / "log"
SETTINGS_PATH: Path = DATA_DIR / "settings.json"
SOUNDS_DIR: Path = PROJECT_ROOT / "sounds"
TRANSCRIPTS_DIR: Path = PROJECT_ROOT / "transcripts"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, MODELS_DIR, LOG_DIR, TRANSCRIPTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_session_paths() -> Path:
    stamp = datetime.now().strftime("[%d-%m-%y] - [%H-%M]")
    transcript_dir = TRANSCRIPTS_DIR / stamp
    transcript_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Session dir: %s", transcript_dir)
    return transcript_dir
