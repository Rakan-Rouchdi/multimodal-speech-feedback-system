from pathlib import Path

from app.audio.preprocessing import preprocess_audio


def test_preprocess_audio_returns_expected_tuple(synthetic_speech_wav: Path):
    waveform, sr, duration = preprocess_audio(str(synthetic_speech_wav))

    assert sr == 16000
    assert waveform.ndim == 1
    assert waveform.size > 0
    assert duration > 0
