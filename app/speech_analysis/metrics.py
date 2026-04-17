from __future__ import annotations

import numpy as np
import librosa


def energy_mean(waveform: np.ndarray) -> float:
    """
    Simple intensity proxy: mean RMS energy across frames.
    """
    if waveform.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
    return float(np.mean(rms)) if rms.size else 0.0


def pause_metrics(
    waveform: np.ndarray,
    sr: int,
    top_db: int = 30,
    min_pause_sec: float = 0.15,
) -> dict:
    """
    Pause approximation using non-silent intervals.

    Changes from the original version:
    - Only INTERNAL pauses are counted.
    - Leading and trailing silence are ignored.
    - Very tiny gaps are ignored using min_pause_sec.

    Returns:
      pause_count, mean_pause_sec, total_pause_sec
    """
    if waveform.size == 0 or sr <= 0:
        return {"pause_count": 0, "mean_pause_sec": 0.0, "total_pause_sec": 0.0}

    intervals = librosa.effects.split(waveform, top_db=top_db)

    if intervals.size == 0:
        duration_sec = len(waveform) / sr
        return {
            "pause_count": 1 if duration_sec > 0 else 0,
            "mean_pause_sec": float(duration_sec),
            "total_pause_sec": float(duration_sec),
        }

    pause_durations = []

    # INTERNAL pauses only
    for (_, end_a), (start_b, _) in zip(intervals[:-1], intervals[1:]):
        gap_sec = (start_b - end_a) / sr
        if gap_sec >= min_pause_sec:
            pause_durations.append(gap_sec)

    if not pause_durations:
        return {"pause_count": 0, "mean_pause_sec": 0.0, "total_pause_sec": 0.0}

    total_pause = float(np.sum(pause_durations))
    mean_pause = float(np.mean(pause_durations))
    return {
        "pause_count": int(len(pause_durations)),
        "mean_pause_sec": mean_pause,
        "total_pause_sec": total_pause,
    }


def pitch_metrics(waveform: np.ndarray, sr: int) -> dict:
    """
    Pitch estimate using librosa.pyin (monophonic pitch tracking).
    Returns mean and std of voiced pitch (Hz). Unvoiced frames ignored.
    """
    if waveform.size == 0 or sr <= 0:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0}

    f0, voiced_flag, voiced_prob = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    if f0 is None:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0}

    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size == 0:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0}

    return {
        "pitch_mean_hz": float(np.mean(voiced_f0)),
        "pitch_std_hz": float(np.std(voiced_f0)),
    }


def compute_speech_metrics(waveform: np.ndarray, sr: int) -> dict:
    pauses = pause_metrics(waveform, sr, top_db=30, min_pause_sec=0.15)
    pitch = pitch_metrics(waveform, sr)
    return {
        "energy_mean": energy_mean(waveform),
        **pauses,
        **pitch,
    }