from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from app.contracts.constants import BANDS


# These weights are part of the dissertation-facing scoring design.
# Keeping them as named constants makes the formulas easier to cite and review.
CONFIDENCE_WEIGHTS = {
    "filler_score": 0.40,
    "repeat_score": 0.25,
    "mean_pause_score": 0.20,
    "wpm_score": 0.15,
    "emotion_confidence_score": 0.10,
}

CLARITY_WEIGHTS = {
    "readability_score": 0.35,
    "filler_score": 0.25,
    "repeat_score": 0.15,
    "wpm_score": 0.15,
    "mean_pause_score": 0.10,
}

ENGAGEMENT_WEIGHTS = {
    "pitch_var_score": 0.40,
    "energy_score": 0.30,
    "wpm_score": 0.20,
    "pause_rate_score": 0.10,
    "emotion_engagement_score": 0.10,
}


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
    0–100 score where being inside [ideal_min, ideal_max] is best.
    Falls linearly toward 0 outside the ideal range.
    """
    if value <= hard_min or value >= hard_max:
        return 0.0
    if ideal_min <= value <= ideal_max:
        return 100.0
    if value < ideal_min:
        return 100.0 * (value - hard_min) / (ideal_min - hard_min)
    return 100.0 * (hard_max - value) / (hard_max - ideal_max)


def score_lower_better(value: float, ideal_max: float, hard_max: float) -> float:
    """
    0–100 score where lower values are better.
    Up to ideal_max => 100, at/above hard_max => 0.
    """
    if value <= ideal_max:
        return 100.0
    if value >= hard_max:
        return 0.0
    return 100.0 * (hard_max - value) / (hard_max - ideal_max)


def coverage_adjusted_score(
    items: Iterable[Tuple[float, Optional[float]]],
    neutral: float = 50.0,
) -> float:
    """
    Weighted average over available evidence, then shrink toward a neutral score
    based on how much of the intended evidence was actually present.

    This prevents text_only or speech_only from getting perfect 100s just because
    the remaining available features happen to look ideal.
    """
    items = list(items)
    total_weight = sum(w for w, _ in items)
    valid = [(w, v) for w, v in items if v is not None]

    if total_weight <= 0:
        return neutral
    if not valid:
        return neutral

    available_weight = sum(w for w, _ in valid)
    raw = sum(w * v for w, v in valid) / available_weight
    coverage = available_weight / total_weight

    return neutral + coverage * (raw - neutral)


def round_or_zero(x: Optional[float]) -> float:
    return round(x, 1) if x is not None else 0.0


# ── Emotion-to-score mapping ──────────────────────────────────────────────────
# Maps each emotion label to a (confidence_value, engagement_value) pair on 0-100.
#   - Confidence: calm/neutral delivery scores highest
#   - Engagement: happy/surprised delivery scores highest
EMOTION_SCORE_MAP = {
    "neutral":   (85.0, 40.0),
    "calm":      (95.0, 35.0),
    "happy":     (70.0, 95.0),
    "surprised": (55.0, 85.0),
    "angry":     (40.0, 60.0),
    "fearful":   (25.0, 30.0),
    "sad":       (30.0, 20.0),
    "disgust":   (35.0, 25.0),
}


def _emotion_subscore(emotion_data: Optional[Dict], dimension: str) -> Optional[float]:
    """
    Derive a 0-100 subscore from emotion probabilities.

    Uses a probability-weighted blend of all emotion scores rather than
    relying solely on the argmax label — this is more robust when the model
    is uncertain.

    *dimension* must be ``"confidence"`` or ``"engagement"``.
    """
    if not emotion_data:
        return None

    probs = emotion_data.get("probabilities")
    if not probs:
        return None

    dim_idx = 0 if dimension == "confidence" else 1
    score = 0.0
    total_prob = 0.0

    for label, prob in probs.items():
        if label in EMOTION_SCORE_MAP:
            score += prob * EMOTION_SCORE_MAP[label][dim_idx]
            total_prob += prob

    if total_prob <= 0:
        return None

    return score / total_prob
# ─────────────────────────────────────────────────────────────────────────────


def scoring_v1(speech: Dict, text: Dict, use_emotion: bool = True) -> Dict:
    """
    Evidence-based asymmetric scoring:
    - confidence: text-led (0.65 text, 0.35 speech)
    - clarity: text-led (0.70 text, 0.30 speech)  
    - engagement: speech-led (0.85 speech, 0.15 text)
    - emotion opt-in
    """

    speech_available = bool(speech.get("available", bool(speech)))
    text_available = bool(text.get("available", bool(text)))

    # WPM can exist even for text_only because it comes from transcript + duration
    wpm_raw = speech.get("speech_rate_wpm", None)
    wpm = float(wpm_raw) if wpm_raw not in (None, "") else None

    filler = float(text.get("filler_rate_per_100w", 0.0)) if text_available else None
    repeat = float(text.get("repeat_rate", 0.0)) if text_available else None
    readability = float(text.get("readability_proxy", 0.0)) if text_available else None

    pitch_std = float(speech.get("pitch_std_hz", 0.0)) if speech_available else None
    mean_pause = float(speech.get("mean_pause_sec", 0.0)) if speech_available else None
    energy = float(speech.get("energy_mean", 0.0)) if speech_available else None
    pause_rate = float(speech.get("pause_rate_per_min", 0.0)) if speech_available else None
    pause_ratio = float(speech.get("pause_ratio", 0.0)) if speech_available else None

    # Emotion data (opt-in)
    emotion_data = None
    if use_emotion and speech_available:
        emotion_data = speech.get("emotion")

    # --- Subscores ---
    wpm_score = score_from_range(wpm, ideal_min=120, ideal_max=165, hard_min=85, hard_max=210) if (wpm is not None and wpm > 0) else None

    filler_score = score_lower_better(filler, ideal_max=1.0, hard_max=8.0) if filler is not None else None

    # assumes adjacent repetition rate from improved text metrics
    repeat_score = score_lower_better(repeat, ideal_max=0.03, hard_max=0.25) if repeat is not None else None

    readability_score = clamp_0_100(readability) if readability is not None else None

    pitch_var_score = (
        score_from_range(pitch_std, ideal_min=18, ideal_max=60, hard_min=5, hard_max=120)
        if pitch_std is not None else None
    )

    mean_pause_score = (
        score_from_range(mean_pause, ideal_min=0.12, ideal_max=0.40, hard_min=0.05, hard_max=1.20)
        if mean_pause is not None else None
    )

    pause_rate_score = (
        score_lower_better(pause_rate, ideal_max=12.0, hard_max=35.0)
        if pause_rate is not None else None
    )

    pause_ratio_score = (
        score_lower_better(pause_ratio, ideal_max=0.18, hard_max=0.50)
        if pause_ratio is not None else None
    )

    energy_score = (
        score_from_range(energy, ideal_min=0.025, ideal_max=0.08, hard_min=0.005, hard_max=0.14)
        if energy is not None else None
    )

    # Emotion-derived subscores
    emotion_confidence_score = _emotion_subscore(emotion_data, "confidence")
    emotion_engagement_score = _emotion_subscore(emotion_data, "engagement")

    # --- Headline scores (asymmetric evidence-based weights) ---
    # CONFIDENCE: text-led (0.65T 0.35S)
    confidence_items = [
        (CONFIDENCE_WEIGHTS["filler_score"], filler_score),
        (CONFIDENCE_WEIGHTS["repeat_score"], repeat_score),
        (CONFIDENCE_WEIGHTS["mean_pause_score"], mean_pause_score),
        (CONFIDENCE_WEIGHTS["wpm_score"], wpm_score),
    ]
    if emotion_confidence_score is not None:
        confidence_items.append(
            (CONFIDENCE_WEIGHTS["emotion_confidence_score"], emotion_confidence_score)
        )
    confidence = coverage_adjusted_score(confidence_items)

    # CLARITY: text-led (0.70T 0.30S)
    clarity = coverage_adjusted_score([
        (CLARITY_WEIGHTS["readability_score"], readability_score),
        (CLARITY_WEIGHTS["filler_score"], filler_score),
        (CLARITY_WEIGHTS["repeat_score"], repeat_score),
        (CLARITY_WEIGHTS["wpm_score"], wpm_score),
        (CLARITY_WEIGHTS["mean_pause_score"], mean_pause_score),
    ])

    # ENGAGEMENT: speech-led (0.85S 0.15T/shared)
    engagement_items = [
        (ENGAGEMENT_WEIGHTS["pitch_var_score"], pitch_var_score),
        (ENGAGEMENT_WEIGHTS["energy_score"], energy_score),
        (ENGAGEMENT_WEIGHTS["wpm_score"], wpm_score),
        (ENGAGEMENT_WEIGHTS["pause_rate_score"], pause_rate_score),
    ]
    if emotion_engagement_score is not None:
        engagement_items.append(
            (ENGAGEMENT_WEIGHTS["emotion_engagement_score"], emotion_engagement_score)
        )
    engagement = coverage_adjusted_score(engagement_items)

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
            "wpm_score": round_or_zero(wpm_score),
            "filler_score": round_or_zero(filler_score),
            "repeat_score": round_or_zero(repeat_score),
            "readability_score": round_or_zero(readability_score),
            "pitch_var_score": round_or_zero(pitch_var_score),
            "mean_pause_score": round_or_zero(mean_pause_score),
            "pause_rate_score": round_or_zero(pause_rate_score),
            "pause_ratio_score": round_or_zero(pause_ratio_score),
            "energy_score": round_or_zero(energy_score),
            "emotion_confidence_score": round_or_zero(emotion_confidence_score),
            "emotion_engagement_score": round_or_zero(emotion_engagement_score),
        },
    }
