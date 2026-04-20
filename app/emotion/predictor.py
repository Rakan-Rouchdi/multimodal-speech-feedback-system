from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa

# Lazy-load TensorFlow to avoid import overhead when not needed
_model = None

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / "VoiceModel" / "best_model.h5"

# Label order must match the LabelEncoder used during training (alphabetical)
EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised",
]

MAX_LEN = 250  # timesteps the model was trained on
# ─────────────────────────────────────────────────────────────────────────────


def _load_model():
    """Load the Keras model once and cache it (singleton)."""
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model  # type: ignore[import]
        _model = load_model(str(MODEL_PATH), compile=False)
    return _model


def extract_features(y: np.ndarray, sr: int, max_len: int = MAX_LEN) -> Optional[np.ndarray]:
    """
    Extract time-series features from a waveform for the CNN+BiLSTM model.

    Produces a (max_len, 40) matrix:
      13 MFCCs + 13 deltas + 13 delta2s + 1 ZCR + 12 chroma + 1 RMSE = 40

    Replicates the feature extraction from ``app/VoiceModel/app.py`` and
    the Training notebook exactly.
    """
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=40)

        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        rmse = librosa.feature.rms(y=y_trimmed)

        features = np.vstack([mfcc, delta, delta2, zcr, chroma, rmse]).T  # (time, 40)

        if features.shape[0] < max_len:
            pad_width = max_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
        else:
            features = features[:max_len, :]

        return features
    except Exception:
        return None


def predict_emotion(file_path: str) -> Dict:
    """
    Predict the emotion of an audio file.

    Returns:
        {
            "top_label": "neutral",
            "probabilities": {"angry": 0.05, "calm": 0.30, ...}
        }

    If feature extraction fails, returns neutral with uniform probabilities.
    """
    model = _load_model()

    y, sr = librosa.load(file_path, sr=None, mono=True)
    features = extract_features(y, sr)

    if features is None:
        uniform = 1.0 / len(EMOTION_LABELS)
        return {
            "top_label": "neutral",
            "probabilities": {lbl: uniform for lbl in EMOTION_LABELS},
        }

    # Model expects (batch, timesteps, features)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0)
    probs = prediction[0]

    top_idx = int(np.argmax(probs))
    top_label = EMOTION_LABELS[top_idx]

    probabilities = {
        label: round(float(probs[i]), 4)
        for i, label in enumerate(EMOTION_LABELS)
    }

    return {
        "top_label": top_label,
        "probabilities": probabilities,
    }
