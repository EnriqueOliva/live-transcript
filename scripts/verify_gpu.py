from __future__ import annotations

import sys


def main() -> None:
    print("=" * 60)
    print("GPU Verification")
    print("=" * 60)

    try:
        import torch

        print(f"PyTorch version:  {torch.__version__}")
        print(f"CUDA available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version:     {torch.version.cuda}")
            print(f"cuDNN version:    {torch.backends.cudnn.version()}")
            print(f"Device count:     {torch.cuda.device_count()}")
            print(f"Device name:      {torch.cuda.get_device_name(0)}")
            print(f"Compute cap:      {torch.cuda.get_device_capability(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory
            print(f"VRAM:             {mem / (1024**3):.1f} GB")
        else:
            print("WARNING: CUDA not available")
            print(f"PyTorch CUDA:     {torch.version.cuda}")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: uv sync")
        sys.exit(1)

    print()

    try:
        import ctranslate2

        print(f"CTranslate2 ver:  {ctranslate2.__version__}")
        supported = ctranslate2.get_supported_compute_types("cuda")
        print(f"CT2 CUDA types:   {supported}")
    except ImportError:
        print("NOTE: CTranslate2 not directly importable (via faster-whisper)")
    except Exception as e:
        print(f"CT2 CUDA check:   Failed ({e})")

    print()

    try:
        from faster_whisper import WhisperModel

        print("faster-whisper:   OK")
    except ImportError:
        print("ERROR: faster-whisper not installed. Run: uv sync")

    print("=" * 60)


if __name__ == "__main__":
    main()
