from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from app.contracts.constants import BANDS


def band_for(score: float) -> str:
    s = int(round(score))
    s = max(0, min(100, s))
    for name, lo, hi in BANDS:
        if lo <= s <= hi:
            return name
    return "Needs improvement"


def clamp_0_100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def score_from_range(value: float, ideal_min: float, ideal_max: float, hard_min: float, hard_max: float) -> float:
    """
    Maps a metric to 0–100 where being inside [ideal_min, ideal_max] scores high.
    Outside ideal range, score drops linearly until hard bounds.
    """
    if value <= hard_min or value >= hard_max:
        return 0.0
    if ideal_min <= value <= ideal_max:
        return 100.0

    # below ideal
    if value < ideal_min:
        return 100.0 * (value - hard_min) / (ideal_min - hard_min)

    # above ideal
    return 100.0 * (hard_max - value) / (hard_max - ideal_max)


def scoring_v1(speech: Dict, text: Dict) -> Dict:
    """
    v1 scoring based on simple, explainable heuristics.
    Expects:
      speech: pause_count, mean_pause_sec, total_pause_sec, energy_mean, pitch_std_hz, speech_rate_wpm
      text: filler_rate_per_100w, repeat_rate, readability_proxy
    """

    # --- Subscores (0–100) ---
    wpm = float(speech.get("speech_rate_wpm", 0.0))
    filler = float(text.get("filler_rate_per_100w", 0.0))
    repeat = float(text.get("repeat_rate", 0.0))
    readability = float(text.get("readability_proxy", 0.0))
    pitch_std = float(speech.get("pitch_std_hz", 0.0))
    mean_pause = float(speech.get("mean_pause_sec", 0.0))

    # Speech rate: ideal conversational range ~120–170 wpm
    wpm_score = score_from_range(wpm, ideal_min=120, ideal_max=170, hard_min=80, hard_max=220)

    # Filler rate per 100 words: lower is better
    # 0 -> 100, 6+ -> 0
    filler_score = clamp_0_100(100.0 - (filler * 16.7))

    # Repetition: lower is better (0.0 ideal, 0.35+ poor)
    repeat_score = clamp_0_100(100.0 - (repeat * 285.0))

    # Readability proxy already 0–100
    readability_score = clamp_0_100(readability)

    # Pitch variability: too flat sounds disengaged, too erratic sounds unstable
    # ideal std range (rough): 20–60 Hz, hard bounds: 5–120
    pitch_var_score = score_from_range(pitch_std, ideal_min=20, ideal_max=60, hard_min=5, hard_max=120)

    # Mean pause duration: ideal ~0.1–0.35 sec, hard bounds 0.05–1.0
    pause_score = score_from_range(mean_pause, ideal_min=0.10, ideal_max=0.35, hard_min=0.05, hard_max=1.0)

    # --- Headline scores (weighted averages) ---
    # Confidence: fillers + pauses + pace
    confidence = (0.45 * filler_score) + (0.30 * pause_score) + (0.25 * wpm_score)

    # Clarity: readability + repetition + pace
    clarity = (0.45 * readability_score) + (0.35 * repeat_score) + (0.20 * wpm_score)

    # Engagement: pitch variability + energy proxy + pace
    # energy_mean is dataset-dependent so we map it loosely: 0.01 -> low, 0.08 -> high
    energy = float(speech.get("energy_mean", 0.0))
    energy_score = score_from_range(energy, ideal_min=0.03, ideal_max=0.08, hard_min=0.005, hard_max=0.12)

    engagement = (0.45 * pitch_var_score) + (0.35 * energy_score) + (0.20 * wpm_score)

    confidence = clamp_0_100(confidence)
    clarity = clamp_0_100(clarity)
    engagement = clamp_0_100(engagement)

    return {
        "scores": {
            "confidence": round(confidence, 1),
            "clarity": round(clarity, 1),
            "engagement": round(engagement, 1),
            "bands": {
                "confidence": band_for(confidence),
                "clarity": band_for(clarity),
                "engagement": band_for(engagement),
            },
        },
        "subscores": {
            "wpm_score": round(wpm_score, 1),
            "filler_score": round(filler_score, 1),
            "repeat_score": round(repeat_score, 1),
            "readability_score": round(readability_score, 1),
            "pitch_var_score": round(pitch_var_score, 1),
            "pause_score": round(pause_score, 1),
            "energy_score": round(energy_score, 1),
        },
    }
