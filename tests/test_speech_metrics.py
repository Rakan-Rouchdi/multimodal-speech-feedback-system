from pathlib import Path

from app.audio.preprocessing import preprocess_audio
from app.speech_analysis.metrics import compute_speech_metrics


def test_silent_audio_does_not_crash_acoustic_module(silent_wav: Path):
    waveform, sr, _ = preprocess_audio(str(silent_wav))
    metrics = compute_speech_metrics(waveform, sr)

    assert set(("energy_mean", "pause_count", "pitch_mean_hz", "pitch_std_hz")).issubset(metrics)


def test_very_short_audio_does_not_crash_acoustic_module(very_short_wav: Path):
    waveform, sr, _ = preprocess_audio(str(very_short_wav))
    metrics = compute_speech_metrics(waveform, sr)

    assert metrics["energy_mean"] >= 0.0
