from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline import runner
from app.transcription.types import TranscriptionResult


def _stub_transcription(*args, **kwargs) -> TranscriptionResult:
    return TranscriptionResult(
        transcript="Um I think this is a clear clear answer.",
        clean_text="I think this is a clear clear answer.",
        language="en",
        segments=[{"start": 0.0, "end": 1.0, "text": "stub"}],
        words=[{"text": "I", "start_s": 0.0, "end_s": 0.1}],
    )


class StubCrisperWhisperTranscriber:
    def __init__(self):
        self.calls = []

    def transcribe(self, file_path: str) -> TranscriptionResult:
        self.calls.append(file_path)
        return _stub_transcription()


def test_pipeline_variants_run_with_expected_modules(monkeypatch, synthetic_speech_wav: Path):
    transcriber = StubCrisperWhisperTranscriber()
    monkeypatch.setattr(runner, "get_transcriber", lambda: transcriber)

    speech_only = runner.run_pipeline(str(synthetic_speech_wav), "speech_only")
    assert speech_only["speech_metrics"] is not None
    assert speech_only["text_metrics"] is None

    text_only = runner.run_pipeline(str(synthetic_speech_wav), "text_only")
    assert text_only["speech_metrics"] is None
    assert text_only["text_metrics"] is not None
    assert text_only["transcript"]

    multimodal = runner.run_pipeline(str(synthetic_speech_wav), "multimodal")
    assert multimodal["speech_metrics"] is not None
    assert multimodal["text_metrics"] is not None
    assert multimodal["meta"]["pipeline_variant"] == "multimodal"
    assert transcriber.calls == [str(synthetic_speech_wav), str(synthetic_speech_wav)]


def test_pipeline_invalid_variant_fails_clearly(synthetic_speech_wav: Path):
    with pytest.raises(ValueError, match="variant must be one of"):
        runner.run_pipeline(str(synthetic_speech_wav), "bad_variant")


def test_pipeline_output_contains_expected_logging_fields(monkeypatch, synthetic_speech_wav: Path):
    transcriber = StubCrisperWhisperTranscriber()
    monkeypatch.setattr(runner, "get_transcriber", lambda: transcriber)
    monkeypatch.setattr(
        runner,
        "predict_emotion",
        lambda *args, **kwargs: {"top_label": "calm", "probabilities": {"calm": 1.0}},
    )

    result = runner.run_pipeline(str(synthetic_speech_wav), "multimodal", use_emotion=True)

    assert result["filename"] == synthetic_speech_wav.name
    assert result["variant"] == "multimodal"
    assert result["transcript"]
    assert result["acoustic_features"] == result["speech_metrics"]
    assert result["text_features"] == result["text_metrics"]
    assert result["emotion_output"]["top_label"] == "calm"
    assert result["meta"]["transcription_source"] == "crisper_whisper"
    assert result["meta"]["transcription_cache_enabled"] is False
    assert result["meta"]["transcription_cache_hit"] is False
    assert transcriber.calls == [str(synthetic_speech_wav)]
    assert 0 <= result["confidence_score"] <= 100
    assert 0 <= result["clarity_score"] <= 100
    assert 0 <= result["engagement_score"] <= 100
    assert "total_runtime" in result["latency_timings"]


def test_valid_audio_input_returns_all_three_scores(monkeypatch, synthetic_speech_wav: Path):
    monkeypatch.setattr(runner, "get_transcriber", lambda: StubCrisperWhisperTranscriber())

    result = runner.run_pipeline(str(synthetic_speech_wav), "multimodal")

    scores = result["scores"]
    assert set(("confidence", "clarity", "engagement")).issubset(scores)


def test_pipeline_can_use_transcription_cache(monkeypatch, synthetic_speech_wav: Path, tmp_path: Path):
    transcriber = StubCrisperWhisperTranscriber()
    monkeypatch.setattr(runner, "get_transcriber", lambda: transcriber)
    cache_path = tmp_path / "transcriptions.json"

    first = runner.run_pipeline(
        str(synthetic_speech_wav),
        "text_only",
        use_transcription_cache=True,
        transcription_cache_path=cache_path,
    )
    second = runner.run_pipeline(
        str(synthetic_speech_wav),
        "text_only",
        use_transcription_cache=True,
        transcription_cache_path=cache_path,
    )

    assert first["meta"]["transcription_cache_hit"] is False
    assert second["meta"]["transcription_cache_hit"] is True
    assert transcriber.calls == [str(synthetic_speech_wav)]


@pytest.mark.integration
def test_integration_pipeline_json_save(monkeypatch, synthetic_speech_wav: Path, tmp_path: Path):
    from app.output.save_json import save_result_json

    monkeypatch.setattr(runner, "get_transcriber", lambda: StubCrisperWhisperTranscriber())

    result = runner.run_pipeline(str(synthetic_speech_wav), "multimodal")
    saved_path = save_result_json(result, base_outputs_dir=str(tmp_path))

    assert isinstance(result, dict)
    assert result["variant"] == "multimodal"
    assert 0 <= result["scores"]["confidence"] <= 100
    assert 0 <= result["scores"]["clarity"] <= 100
    assert 0 <= result["scores"]["engagement"] <= 100
    assert result["feedback"]["bullets"]
    assert Path(saved_path).exists()
