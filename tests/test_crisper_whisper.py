from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline.runner import run_pipeline
from app.transcription.crisper_whisper import CrisperWhisperTranscriber, clean_transcript


class FakeCrisperWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        assert kwargs["word_timestamps"] is True
        assert kwargs["language"] == "en"
        self.calls.append(kwargs["vad_filter"])

        words = [
            SimpleNamespace(word=" Hello", start=0.1, end=0.4),
            SimpleNamespace(word=",world", start=0.4, end=0.8),
            SimpleNamespace(word=" [UH]", start=0.8, end=1.0),
        ]
        segments = [
            SimpleNamespace(start=0.1, end=1.0, text="Hello,world [UH]", words=words),
        ]
        info = SimpleNamespace(language="en")
        return segments, info


class FallbackCrisperWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, **kwargs):
        self.calls.append(kwargs["vad_filter"])
        info = SimpleNamespace(language="en")
        if kwargs["vad_filter"]:
            return [
                SimpleNamespace(
                    start=30.0,
                    end=31.0,
                    text="My.",
                    words=[SimpleNamespace(word=" My.", start=30.0, end=31.0)],
                )
            ], info

        words = [
            SimpleNamespace(word=" My", start=0.0, end=0.2),
            SimpleNamespace(word=" name", start=0.2, end=0.4),
            SimpleNamespace(word=" is", start=0.4, end=0.6),
        ] * 10
        return [
            SimpleNamespace(
                start=0.0,
                end=35.0,
                text="My,name,is,a,complete,answer.",
                words=words,
            )
        ], info


def test_clean_transcript_preserves_crisperwhisper_disfluencies_as_words():
    assert clean_transcript("Hello,[UH],this,is,[UM],fine.") == "Hello uh this is um fine."


def test_clean_transcript_splits_compact_period_separated_words():
    text = "Hello,my,name's.I'm.Currently,a.Second-year.Student."
    assert clean_transcript(text) == "Hello my name's I'm Currently a Second-year Student."


def test_crisper_whisper_adapter_returns_segments_and_word_timestamps():
    transcriber = CrisperWhisperTranscriber.__new__(CrisperWhisperTranscriber)
    transcriber.model = FakeCrisperWhisperModel()

    result = transcriber.transcribe("dummy.wav")

    assert result.language == "en"
    assert result.transcript == "Hello,world [UH]"
    assert result.clean_text == "Hello world uh"
    assert result.segments == [{"start": 0.1, "end": 1.0, "text": "Hello,world [UH]"}]
    assert result.words == [
        {"text": "Hello", "start_s": 0.1, "end_s": 0.4},
        {"text": "world", "start_s": 0.4, "end_s": 0.8},
        {"text": "[UH]", "start_s": 0.8, "end_s": 1.0},
    ]


def test_crisper_whisper_retries_without_vad_when_long_audio_looks_incomplete(monkeypatch):
    transcriber = CrisperWhisperTranscriber.__new__(CrisperWhisperTranscriber)
    model = FallbackCrisperWhisperModel()
    transcriber.model = model
    monkeypatch.setattr(transcriber, "_wav_duration", lambda audio_path: 40.0)

    result = transcriber.transcribe("long.wav")

    assert model.calls == [True, False]
    assert len(result.words) == 30
    assert result.clean_text == "My name is a complete answer."


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_CRISPERWHISPER_INTEGRATION") != "1",
    reason="Set RUN_CRISPERWHISPER_INTEGRATION=1 to run the real CrisperWhisper model on data/test.wav.",
)
def test_real_crisperwhisper_pipeline_on_test_wav_without_cache():
    audio_path = Path("data/test.wav")
    assert audio_path.exists()

    result = run_pipeline(str(audio_path), variant="multimodal")

    assert result["meta"]["transcription_source"] == "crisper_whisper"
    assert result["transcript"]
    assert result["text_metrics"]["clean_word_count"] > 0
    assert "transcription" in result["latency_ms"]
    assert 0 <= result["scores"]["confidence"] <= 100
