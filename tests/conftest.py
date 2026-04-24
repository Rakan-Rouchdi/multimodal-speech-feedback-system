import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> Path:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return path


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks integration tests")


import pytest


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def synthetic_speech_wav(tmp_path: Path, sample_rate: int) -> Path:
    duration_sec = 1.2
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = 0.35 * np.sin(2 * np.pi * 220.0 * t)
    return _write_wav(tmp_path / "synthetic_speech.wav", samples, sample_rate)


@pytest.fixture
def silent_wav(tmp_path: Path, sample_rate: int) -> Path:
    samples = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
    return _write_wav(tmp_path / "silent.wav", samples, sample_rate)


@pytest.fixture
def very_short_wav(tmp_path: Path, sample_rate: int) -> Path:
    samples = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
    return _write_wav(tmp_path / "very_short.wav", samples, sample_rate)
