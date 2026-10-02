from __future__ import annotations

import os
from typing import Protocol

import numpy as np

FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (1, 1, 128)
VAD_MODEL_FILE = "silero_vad_v6.onnx"


class FrameClassifier(Protocol):
    def __call__(self, frames: np.ndarray) -> np.ndarray: ...


class StreamingVad:
    def __init__(self, session: object | None = None) -> None:
        self._session = session if session is not None else _create_session()
        self._hidden = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._cell = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        previous_contexts = np.concatenate([self._context[None, :], frames[:-1, -CONTEXT_SAMPLES:]], axis=0)
        batched = np.concatenate([previous_contexts, frames], axis=1)
        output, self._hidden, self._cell = self._session.run(  # type: ignore[attr-defined]
            None, {"input": batched, "h": self._hidden, "c": self._cell},
        )
        self._context = frames[-1, -CONTEXT_SAMPLES:].copy()
        return np.asarray(output, dtype=np.float32).reshape(-1)


def _create_session() -> object:
    import onnxruntime
    from faster_whisper.utils import get_assets_path

    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.log_severity_level = 4
    return onnxruntime.InferenceSession(
        os.path.join(get_assets_path(), VAD_MODEL_FILE),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
