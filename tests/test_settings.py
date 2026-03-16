import json

from whisper_transcriber.config.settings import AppSettings


class TestLoadValid:
    def test_load_valid_json(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_size": "turbo",
            "language": "es",
            "compute_type": "int8_float16",
            "chunk_duration": 10.0,
            "theme": "light",
        }
        settings_path.write_text(json.dumps(data), encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "turbo"
        assert loaded.language == "es"
        assert loaded.compute_type == "int8_float16"
        assert loaded.chunk_duration == 10.0
        assert loaded.theme == "light"


class TestLoadFallbacks:
    def test_load_missing_file_returns_defaults(self, settings_path):
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "turbo"
        assert loaded.language == "en"

    def test_load_corrupted_json_returns_defaults(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("{broken json", encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "turbo"

    def test_missing_fields_use_defaults(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps({"language": "fr"}), encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.language == "fr"
        assert loaded.model_size == "turbo"
        assert loaded.chunk_duration == 30.0


class TestSaveReload:
    def test_save_and_reload_roundtrip(self, settings_path):
        original = AppSettings(
            model_size="small",
            language="de",
            compute_type="float16",
            chunk_duration=15.0,
            theme="dark",
        )
        original.save(settings_path)
        reloaded = AppSettings.load(settings_path)
        assert reloaded.model_size == original.model_size
        assert reloaded.language == original.language
        assert reloaded.compute_type == original.compute_type
        assert reloaded.chunk_duration == original.chunk_duration


class TestUnknownFields:
    def test_unknown_fields_ignored(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_size": "tiny",
            "language": "ja",
            "some_future_field": True,
            "another_unknown": 42,
        }
        settings_path.write_text(json.dumps(data), encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "tiny"
        assert not hasattr(loaded, "some_future_field")


class TestValidation:
    def test_invalid_model_resets(self):
        s = AppSettings(model_size="nonexistent")
        assert s.model_size == "large-v3"

    def test_chunk_duration_clamped(self):
        s = AppSettings(chunk_duration=999.0)
        assert s.chunk_duration == 60.0
