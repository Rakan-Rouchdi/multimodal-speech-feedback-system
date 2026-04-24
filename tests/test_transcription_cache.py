from __future__ import annotations

import json
from pathlib import Path

from app.transcription.cache import cache_key, transcribe_with_cache
from app.transcription.crisper_whisper import MODEL_ID
from app.transcription.types import TranscriptionResult


class CountingTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, file_path: str) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            transcript="Hello [UH] world.",
            clean_text="Hello uh world.",
            language="en",
            segments=[{"start": 0.0, "end": 1.2, "text": "Hello [UH] world."}],
            words=[
                {"text": "Hello", "start_s": 0.0, "end_s": 0.3},
                {"text": "[UH]", "start_s": 0.3, "end_s": 0.5},
                {"text": "world.", "start_s": 0.5, "end_s": 1.2},
            ],
        )


def test_transcription_cache_writes_then_hits(synthetic_speech_wav: Path, tmp_path: Path):
    cache_path = tmp_path / "transcriptions.json"
    transcriber = CountingTranscriber()

    first, first_hit = transcribe_with_cache(str(synthetic_speech_wav), transcriber, cache_path)
    second, second_hit = transcribe_with_cache(str(synthetic_speech_wav), transcriber, cache_path)

    assert first_hit is False
    assert second_hit is True
    assert first == second
    assert transcriber.calls == 1
    assert cache_path.exists()


def test_transcription_cache_rejects_entries_without_word_timestamps(
    synthetic_speech_wav: Path,
    tmp_path: Path,
):
    cache_path = tmp_path / "transcriptions.json"
    key = cache_key(str(synthetic_speech_wav))
    cache_path.write_text(
        json.dumps(
            {
                key: {
                    "model_id": MODEL_ID,
                    "cleaning_version": "final",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "transcript": "Cached text",
                    "clean_text": "Cached text",
                    "language": "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Cached text"}],
                    "words": [],
                }
            }
        ),
        encoding="utf-8",
    )
    transcriber = CountingTranscriber()

    result, cache_hit = transcribe_with_cache(str(synthetic_speech_wav), transcriber, cache_path)

    assert cache_hit is False
    assert result.transcript == "Hello [UH] world."
    assert transcriber.calls == 1
