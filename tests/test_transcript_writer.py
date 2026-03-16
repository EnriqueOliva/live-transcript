
from whisper_transcriber.io.transcript_writer import TranscriptWriter


class TestWritePlainText:
    def test_write_plain_text(self, tmp_path):
        writer = TranscriptWriter(tmp_path)
        writer.open()
        writer.write_segment("Hello world", 1.0, 3.0)
        writer.close()
        content = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert "Hello world" in content

    def test_write_timestamped_text(self, tmp_path):
        writer = TranscriptWriter(tmp_path)
        writer.open()
        writer.write_segment("Hello world", 65.0, 68.0)
        writer.close()
        content = (tmp_path / "transcript_with_timestamps.txt").read_text(encoding="utf-8")
        assert "[00:01:05 -> 00:01:08] Hello world" in content


class TestMultipleAppends:
    def test_multiple_appends(self, tmp_path):
        writer = TranscriptWriter(tmp_path)
        writer.open()
        writer.write_segment("First", 0.0, 1.0)
        writer.write_segment("Second", 2.0, 3.0)
        writer.write_segment("Third", 4.0, 5.0)
        writer.close()
        lines = (tmp_path / "transcript.txt").read_text(encoding="utf-8").strip().split("\n")
        assert lines == ["First", "Second", "Third"]


class TestEncoding:
    def test_utf8_encoding(self, tmp_path):
        writer = TranscriptWriter(tmp_path)
        writer.open()
        writer.write_segment("Hola mundo. Cafe. Nino.", 0.0, 2.0)
        writer.close()
        content = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert "Hola mundo" in content


class TestEdgeCases:
    def test_empty_text_skipped(self, tmp_path):
        writer = TranscriptWriter(tmp_path)
        writer.open()
        writer.write_segment("", 0.0, 1.0)
        writer.write_segment("   ", 1.0, 2.0)
        writer.write_segment("Valid", 2.0, 3.0)
        writer.close()
        lines = (tmp_path / "transcript.txt").read_text(encoding="utf-8").strip().split("\n")
        assert lines == ["Valid"]

    def test_context_manager(self, tmp_path):
        with TranscriptWriter(tmp_path) as writer:
            writer.write_segment("Context managed", 0.0, 1.0)
        content = (tmp_path / "transcript.txt").read_text(encoding="utf-8")
        assert "Context managed" in content
