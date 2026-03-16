import re

from whisper_transcriber.io.paths import create_session_paths


class TestSessionFolderFormat:
    def test_session_folder_matches_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setattr("whisper_transcriber.io.paths.TRANSCRIPTS_DIR", tmp_path / "transcripts")
        monkeypatch.setattr("whisper_transcriber.io.paths.SESSION_LOGS_DIR", tmp_path / "logs")
        transcript_dir, log_dir = create_session_paths()
        pattern = r"\[\d{2}-\d{2}-\d{2}\] - \[\d{2}-\d{2}\]"
        assert re.match(pattern, transcript_dir.name)
        assert re.match(pattern, log_dir.name)

    def test_both_dirs_created_on_disk(self, tmp_path, monkeypatch):
        monkeypatch.setattr("whisper_transcriber.io.paths.TRANSCRIPTS_DIR", tmp_path / "transcripts")
        monkeypatch.setattr("whisper_transcriber.io.paths.SESSION_LOGS_DIR", tmp_path / "logs")
        transcript_dir, log_dir = create_session_paths()
        assert transcript_dir.exists()
        assert log_dir.exists()

    def test_dirs_have_same_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr("whisper_transcriber.io.paths.TRANSCRIPTS_DIR", tmp_path / "transcripts")
        monkeypatch.setattr("whisper_transcriber.io.paths.SESSION_LOGS_DIR", tmp_path / "logs")
        transcript_dir, log_dir = create_session_paths()
        assert transcript_dir.name == log_dir.name
