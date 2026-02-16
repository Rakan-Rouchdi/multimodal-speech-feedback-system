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


def pause_metrics(waveform: np.ndarray, sr: int, top_db: int = 30) -> dict:
    """
    Pause approximation using non-silent intervals.

    - Split waveform into non-silent intervals (above threshold).
    - Gaps between these intervals are treated as pauses.

    Returns:
      pause_count, mean_pause_sec, total_pause_sec
    """
    if waveform.size == 0 or sr <= 0:
        return {"pause_count": 0, "mean_pause_sec": 0.0, "total_pause_sec": 0.0}

    intervals = librosa.effects.split(waveform, top_db=top_db)  # shape: (n, 2) sample indices

    if intervals.size == 0:
        # everything is silence
        duration_sec = len(waveform) / sr
        return {"pause_count": 1 if duration_sec > 0 else 0, "mean_pause_sec": float(duration_sec), "total_pause_sec": float(duration_sec)}

    pause_durations = []

    # pause before first speech
    first_start = intervals[0][0]
    if first_start > 0:
        pause_durations.append(first_start / sr)

    # pauses between speech intervals
    for (start_a, end_a), (start_b, end_b) in zip(intervals[:-1], intervals[1:]):
        gap = start_b - end_a
        if gap > 0:
            pause_durations.append(gap / sr)

    # pause after last speech
    last_end = intervals[-1][1]
    if last_end < len(waveform):
        pause_durations.append((len(waveform) - last_end) / sr)

    if not pause_durations:
        return {"pause_count": 0, "mean_pause_sec": 0.0, "total_pause_sec": 0.0}

    total_pause = float(np.sum(pause_durations))
    mean_pause = float(np.mean(pause_durations))
    return {"pause_count": int(len(pause_durations)), "mean_pause_sec": mean_pause, "total_pause_sec": total_pause}


def pitch_metrics(waveform: np.ndarray, sr: int) -> dict:
    """
    Pitch estimate using librosa.pyin (monophonic pitch tracking).
    Returns mean and std of voiced pitch (Hz). Unvoiced frames ignored.
    """
    if waveform.size == 0 or sr <= 0:
        return {"pitch_mean_hz": 0.0, "pitch_std_hz": 0.0}

    # Typical human speech range
    f0, voiced_flag, voiced_prob = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),  # ~65 Hz
        fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
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
    pauses = pause_metrics(waveform, sr, top_db=30)
    pitch = pitch_metrics(waveform, sr)
    return {
        "energy_mean": energy_mean(waveform),
        **pauses,
        **pitch,
    }
