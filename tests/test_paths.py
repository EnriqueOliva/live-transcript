import re
from datetime import datetime

from whisper_transcriber.io import paths
from whisper_transcriber.io.paths import create_session_paths

SESSION_PATTERN = r"\[\d{2}-\d{2}-\d{2}\] - \[\d{2}-\d{2}\]"


class TestSessionFolders:
    def test_session_folder_matches_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        session_dir = create_session_paths()
        assert re.fullmatch(SESSION_PATTERN, session_dir.name)
        assert session_dir.exists()

    def test_two_sessions_in_the_same_minute_never_share_a_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        moment = datetime(2026, 10, 1, 15, 30)
        first = create_session_paths(moment)
        second = create_session_paths(moment)
        third = create_session_paths(moment)
        assert len({first, second, third}) == 3
        assert second.name == f"{first.name} (2)"
        assert third.name == f"{first.name} (3)"


class TestLocations:
    def test_nothing_is_written_inside_the_repository(self):
        repository = paths.Path(__file__).resolve().parents[1]
        for location in (paths.APP_DATA_DIR, paths.LOG_DIR, paths.SETTINGS_PATH, paths.TRANSCRIPTS_DIR):
            assert repository not in location.resolve().parents

    def test_transcripts_live_in_documents(self):
        assert paths.TRANSCRIPTS_DIR.parent.name == "Documents"
