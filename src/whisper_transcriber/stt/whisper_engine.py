from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ALIASES: dict[str, str] = {
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
LOCAL_SUBDIR = "turbo-local"


class WhisperEngine:
    def __init__(
        self,
        model_size: str,
        language: str,
        compute_type: str,
        model_dir: str | Path,
        initial_prompt: str = "",
        hotwords: str = "",
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._compute_type = compute_type
        self._model_dir = Path(model_dir)
        self._initial_prompt = initial_prompt or None
        self._hotwords = hotwords or None
        self._model: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_model_path(self) -> str:
        if self._model_size == "turbo":
            local = self._model_dir / LOCAL_SUBDIR
            if local.exists() and any(local.iterdir()):
                logger.info("Using local turbo model at %s", local)
                return str(local)
        return self._model_size

    def load_model(self) -> None:
        from faster_whisper import WhisperModel

        model_path = self._resolve_model_path()
        logger.info("Loading model '%s' (compute=%s, device=cuda)", model_path, self._compute_type)
        try:
            self._model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=self._compute_type,
                download_root=str(self._model_dir),
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("CUDA OOM, trying int8_float16")
                self._fallback_load(model_path)
            else:
                raise
        except OSError as exc:
            if "1314" in str(exc) or "privilege" in str(exc).lower():
                logger.warning("Symlink error, trying pre-downloaded model or fallback")
                self._fallback_load(model_path)
            else:
                raise
        self._log_gpu_info()
        logger.info("Model loaded: %s (%s)", self._model_size, self._compute_type)

    def _fallback_load(self, original_path: str) -> None:
        from faster_whisper import WhisperModel

        gc.collect()
        self._cuda_empty_cache()
        try:
            self._model = WhisperModel(
                original_path, device="cuda", compute_type="int8_float16",
                download_root=str(self._model_dir),
            )
            self._compute_type = "int8_float16"
            return
        except Exception:
            pass
        gc.collect()
        self._cuda_empty_cache()
        logger.warning("Falling back to large-v3")
        self._model = WhisperModel(
            "large-v3", device="cuda", compute_type="float16",
            download_root=str(self._model_dir),
        )
        self._model_size = "large-v3"
        self._compute_type = "float16"

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            self._cuda_empty_cache()
            logger.info("Model unloaded")

    def transcribe(self, audio: np.ndarray) -> tuple[list, Any]:
        segments_gen, info = self._model.transcribe(
            audio,
            language=self._language if self._language != "Auto" else None,
            beam_size=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            word_timestamps=False,
            initial_prompt=self._initial_prompt,
            hotwords=self._hotwords,
        )
        segments = list(segments_gen)
        return segments, info

    @staticmethod
    def _cuda_empty_cache() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _log_gpu_info(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), total)
        except ImportError:
            pass
