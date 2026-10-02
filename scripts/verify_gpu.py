from __future__ import annotations

import importlib.util
import os
import sys

from livevox.stt.cuda_runtime import (
    cuda_device_count,
    cuda_libraries_loadable,
    register_library_directories,
    resolve_device,
)


def main() -> None:
    print("=" * 60)
    print("System Verification")
    print("=" * 60)

    print(f"Python:           {sys.version.split()[0]}")
    print(f"Platform:         {sys.platform}")
    print(f"CPU cores:        {os.cpu_count()}")
    print()

    directories = register_library_directories()
    for directory in directories:
        print(f"CUDA libraries:   {directory}")

    import ctranslate2

    print(f"CTranslate2:      {ctranslate2.__version__}")
    print(f"CPU types:        {ctranslate2.get_supported_compute_types('cpu')}")
    device_count = cuda_device_count()
    print(f"CUDA GPU count:   {device_count}")
    libraries_ready = cuda_libraries_loadable()
    print(f"cuBLAS loadable:  {libraries_ready}")
    if device_count > 0 and libraries_ready:
        print(f"CUDA types:       {ctranslate2.get_supported_compute_types('cuda')}")
    print()

    has_faster_whisper = importlib.util.find_spec("faster_whisper") is not None
    print(f"faster-whisper:   {'OK' if has_faster_whisper else 'NOT INSTALLED'}")
    print()

    device, compute_type = resolve_device("auto")
    print(f"Mode:             {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print(f"Recommended:      turbo model, {compute_type}")
    if device_count > 0 and not libraries_ready:
        print("Tip:              Run 'uv sync --group cuda' to enable GPU acceleration")
    print("=" * 60)


if __name__ == "__main__":
    main()
