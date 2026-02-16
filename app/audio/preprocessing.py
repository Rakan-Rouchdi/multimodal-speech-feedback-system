import librosa
import numpy as np


TARGET_SAMPLE_RATE = 16000


def load_audio(file_path: str):
    """
    Load audio file and resample to TARGET_SAMPLE_RATE.
    Returns waveform (mono) and sample rate.
    """
    waveform, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    return waveform, sr


def normalise_audio(waveform: np.ndarray):
    """
    Normalise waveform to range [-1, 1].
    """
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def trim_silence(waveform: np.ndarray, sr: int):
    """
    Trim leading and trailing silence.
    """
    trimmed_waveform, _ = librosa.effects.trim(waveform, top_db=20)
    return trimmed_waveform


def preprocess_audio(file_path: str):
    """
    Full preprocessing pipeline.
    Returns:
        waveform (np.ndarray)
        sample_rate (int)
        duration_sec (float)
    """
    waveform, sr = load_audio(file_path)
    waveform = normalise_audio(waveform)
    waveform = trim_silence(waveform, sr)

    duration_sec = len(waveform) / sr

    return waveform, sr, duration_sec
