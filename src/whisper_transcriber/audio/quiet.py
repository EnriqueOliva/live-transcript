from __future__ import annotations

import numpy as np

DEFAULT_WINDOW_SAMPLES = 128


def find_quiet_point(
    samples: np.ndarray,
    samples_start: int,
    lower: int,
    upper: int,
    window: int = DEFAULT_WINDOW_SAMPLES,
    weights: np.ndarray | None = None,
) -> int | None:
    lower = max(lower, samples_start)
    upper = min(upper, samples_start + samples.size)
    first_window = -(-lower // window) * window
    window_count = (upper - first_window) // window
    if window_count <= 0:
        return None
    offset = first_window - samples_start
    region = samples[offset : offset + window_count * window].reshape(window_count, window)
    energy = np.sqrt(np.mean(np.square(region, dtype=np.float64), axis=1))
    if weights is not None:
        energy = energy * weights[:window_count]
    quietest = np.flatnonzero(energy == energy.min())
    chosen = int(quietest[quietest.size // 2])
    return first_window + chosen * window + window // 2


def window_starts(lower: int, upper: int, window: int = DEFAULT_WINDOW_SAMPLES) -> np.ndarray:
    first_window = -(-lower // window) * window
    window_count = max(0, (upper - first_window) // window)
    return first_window + np.arange(window_count) * window
