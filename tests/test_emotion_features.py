import numpy as np

from app.emotion.predictor import extract_features


def test_emotion_feature_extraction_returns_model_shape(sample_rate: int):
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * 220.0 * t)

    features = extract_features(waveform, sample_rate)

    assert features is not None
    assert features.shape == (250, 53)
