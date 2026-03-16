from __future__ import annotations

import os
import sys


def main() -> None:
    print("=" * 60)
    print("System Verification")
    print("=" * 60)

    print(f"Python:           {sys.version.split()[0]}")
    print(f"Platform:         {sys.platform}")
    print(f"CPU cores:        {os.cpu_count()}")

    print()

    try:
        import ctranslate2

        print(f"CTranslate2:      {ctranslate2.__version__}")
        cpu_types = ctranslate2.get_supported_compute_types("cpu")
        print(f"CPU types:        {cpu_types}")

        gpu_count = ctranslate2.get_cuda_device_count()
        print(f"CUDA GPU count:   {gpu_count}")

        if gpu_count > 0:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            print(f"CUDA types:       {cuda_types}")
    except ImportError:
        print("CTranslate2:      NOT INSTALLED")

    print()

    try:
        import torch

        print(f"PyTorch:          {torch.__version__}")
        print(f"CUDA available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:     {torch.version.cuda}")
            print(f"Device:           {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory
            print(f"VRAM:             {mem / (1024**3):.1f} GB")
    except ImportError:
        print("PyTorch:          not installed (OK for CPU mode)")

    print()

    try:
        from faster_whisper import WhisperModel

        print("faster-whisper:   OK")
    except ImportError:
        print("faster-whisper:   NOT INSTALLED")

    print()

    if ctranslate2.get_cuda_device_count() > 0:
        print("Mode:             GPU (CUDA)")
        print("Recommended:      turbo model, float16")
    else:
        print("Mode:             CPU")
        print("Recommended:      turbo model, int8")
        print("Tip:              Run 'uv sync --group cuda' to enable GPU acceleration")

    print("=" * 60)


if __name__ == "__main__":
    main()
