from whisper_transcriber.stt import cuda_runtime


class TestComputeType:
    def test_auto_picks_float16_on_gpu_and_int8_on_cpu(self):
        assert cuda_runtime.compute_type_for("cuda", "auto") == "float16"
        assert cuda_runtime.compute_type_for("cpu", "auto") == "int8"

    def test_gpu_only_types_are_downgraded_on_cpu(self):
        assert cuda_runtime.compute_type_for("cpu", "float16") == "int8"
        assert cuda_runtime.compute_type_for("cpu", "int8_float16") == "int8"
        assert cuda_runtime.compute_type_for("cpu", "float32") == "float32"


class TestResolveDevice:
    def test_gpu_needs_both_a_device_and_cublas(self, monkeypatch):
        monkeypatch.setattr(cuda_runtime, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(cuda_runtime, "cuda_libraries_loadable", lambda: True)
        assert cuda_runtime.resolve_device("auto") == ("cuda", "float16")

    def test_gpu_without_cublas_uses_cpu_instead_of_failing_later(self, monkeypatch):
        monkeypatch.setattr(cuda_runtime, "cuda_device_count", lambda: 1)
        monkeypatch.setattr(cuda_runtime, "cuda_libraries_loadable", lambda: False)
        assert cuda_runtime.resolve_device("auto") == ("cpu", "int8")

    def test_no_gpu_uses_cpu(self, monkeypatch):
        monkeypatch.setattr(cuda_runtime, "cuda_device_count", lambda: 0)
        assert cuda_runtime.resolve_device("auto") == ("cpu", "int8")


class TestLibraryDirectories:
    def test_sotvox_gpu_pack_is_a_candidate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        candidates = cuda_runtime.candidate_library_directories()
        assert tmp_path / "Sotvox" / "cuda" in candidates
