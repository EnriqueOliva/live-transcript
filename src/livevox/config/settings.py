from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from livevox.io.paths import SETTINGS_PATH

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "turbo"
DEFAULT_LANGUAGE = "es"
DEFAULT_COMPUTE_TYPE = "auto"
VALID_MODELS: set[str] = {
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "turbo", "large-v3-turbo", "distil-large-v3",
}
VALID_COMPUTE: set[str] = {"float16", "int8_float16", "int8", "float32", "auto"}
VALID_THEMES: set[str] = {"dark", "light", "system"}


@dataclass
class AppSettings:
    model_size: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    theme: str = "dark"
    record_mic: bool = False
    initial_prompt: str = ""
    hotwords: str = ""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.model_size not in VALID_MODELS:
            logger.warning("Invalid model_size %r, resetting to %s", self.model_size, DEFAULT_MODEL)
            self.model_size = DEFAULT_MODEL
        if self.compute_type not in VALID_COMPUTE:
            logger.warning("Invalid compute_type %r, resetting to %s", self.compute_type, DEFAULT_COMPUTE_TYPE)
            self.compute_type = DEFAULT_COMPUTE_TYPE
        if self.theme not in VALID_THEMES:
            self.theme = "dark"

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        target = path or SETTINGS_PATH
        if not target.exists():
            logger.info("No settings file found, using defaults")
            return cls()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted settings file (%s), using defaults", exc)
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError as exc:
            logger.warning("Bad field types in settings (%s), using defaults", exc)
            return cls()

    def save(self, path: Path | None = None) -> None:
        target = path or SETTINGS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, target)
        except OSError:
            logger.exception("Failed to save settings")
            tmp.unlink(missing_ok=True)
