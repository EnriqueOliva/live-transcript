import json

from livevox.config.settings import AppSettings


class TestLoadValid:
    def test_load_valid_json(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_size": "turbo",
            "language": "es",
            "compute_type": "int8_float16",
            "theme": "light",
        }
        settings_path.write_text(json.dumps(data), encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "turbo"
        assert loaded.language == "es"
        assert loaded.compute_type == "int8_float16"
        assert loaded.theme == "light"


class TestLoadFallbacks:
    def test_load_missing_file_returns_defaults(self, settings_path):
        loaded = AppSettings.load(settings_path)
        assert loaded.model_size == "turbo"
        assert loaded.language == "es"

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


class TestSaveReload:
    def test_save_and_reload_roundtrip(self, settings_path):
        original = AppSettings(
            model_size="small",
            language="de",
            compute_type="float16",
            theme="dark",
        )
        original.save(settings_path)
        reloaded = AppSettings.load(settings_path)
        assert reloaded.model_size == original.model_size
        assert reloaded.language == original.language
        assert reloaded.compute_type == original.compute_type


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
    def test_invalid_model_resets_to_turbo(self):
        s = AppSettings(model_size="nonexistent")
        assert s.model_size == "turbo"

    def test_invalid_compute_type_resets_to_auto(self):
        s = AppSettings(compute_type="bogus")
        assert s.compute_type == "auto"

    def test_settings_from_the_chunked_version_still_load(self, settings_path):
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        old = {"model_size": "turbo", "language": "es", "chunk_duration": 30.0, "overlap_seconds": 5.0,
               "audio_device": 12}
        settings_path.write_text(json.dumps(old), encoding="utf-8")
        loaded = AppSettings.load(settings_path)
        assert loaded.language == "es"
        assert not hasattr(loaded, "chunk_duration")
