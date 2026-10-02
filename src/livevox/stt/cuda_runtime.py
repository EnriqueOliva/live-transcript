from __future__ import annotations

import ctypes
import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_LIBRARIES = ("cublas64_12.dll", "cublasLt64_12.dll")
SOTVOX_CUDA_DIRECTORY = ("Sotvox", "cuda")
PIP_LIBRARY_SUBDIRECTORIES = (("nvidia", "cublas", "bin"),)
GPU_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
GPU_ONLY_COMPUTE_TYPES = {"float16", "int8_float16"}

_registered_directories: list[Path] = []
_loaded_libraries: list[object] = []


def candidate_library_directories() -> list[Path]:
    candidates = [
        Path(site_directory).joinpath(*parts)
        for site_directory in site.getsitepackages()
        for parts in PIP_LIBRARY_SUBDIRECTORIES
    ]
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data).joinpath(*SOTVOX_CUDA_DIRECTORY))
    return candidates


def register_library_directories() -> list[Path]:
    if sys.platform != "win32":
        return []
    for directory in candidate_library_directories():
        if directory.is_dir() and directory not in _registered_directories:
            try:
                os.add_dll_directory(str(directory))
            except OSError:
                logger.warning("Could not register CUDA library directory %s", directory)
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            _registered_directories.append(directory)
            logger.info("CUDA library directory available: %s", directory)
    return list(_registered_directories)


def cuda_libraries_loadable() -> bool:
    if sys.platform != "win32":
        return False
    try:
        _loaded_libraries.extend(ctypes.WinDLL(library) for library in REQUIRED_LIBRARIES)
        return True
    except OSError as error:
        logger.info("CUDA libraries not available (%s)", error)
        return False


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        logger.exception("Could not query CUDA devices")
        return 0


def resolve_device(compute_type_setting: str) -> tuple[str, str]:
    has_device = cuda_device_count() > 0
    if has_device and cuda_libraries_loadable():
        device = "cuda"
    elif has_device:
        logger.warning(
            "An NVIDIA GPU is present but cuBLAS is not installed, transcribing on CPU. "
            "Run 'uv sync --group cuda' to enable the GPU."
        )
        device = "cpu"
    else:
        device = "cpu"
    return device, compute_type_for(device, compute_type_setting)


def compute_type_for(device: str, compute_type_setting: str) -> str:
    if compute_type_setting == "auto":
        return GPU_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE
    elif device == "cpu" and compute_type_setting in GPU_ONLY_COMPUTE_TYPES:
        return CPU_COMPUTE_TYPE
    else:
        return compute_type_setting
