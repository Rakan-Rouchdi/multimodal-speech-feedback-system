from __future__ import annotations

from typing import Dict, List


def _add_if(items: List[str], condition: bool, text: str):
    if condition:
        items.append(text)


def generate_feedback(scores: Dict, speech: Dict, text: Dict, subscores: Dict) -> Dict:
    """
    Deterministic feedback rules based on metrics and score bands.
    Returns: {summary: str, bullets: [str], next_practice: [str]}
    """

    # Headline scores + bands
    confidence = float(scores.get("confidence", 0.0))
    clarity = float(scores.get("clarity", 0.0))
    engagement = float(scores.get("engagement", 0.0))
    bands = scores.get("bands", {})

    # Emotion data (may be absent for text_only or if emotion analysis failed)
    emotion_data = speech.get("emotion")
    emotion_label: str = (
        emotion_data.get("top_label", "")
        if isinstance(emotion_data, dict)
        else ""
    )

    # Key metrics
    wpm = float(speech.get("speech_rate_wpm", 0.0))
    filler_rate = float(text.get("filler_rate_per_100w", 0.0))
    mean_pause = float(speech.get("mean_pause_sec", 0.0))
    pitch_std = float(speech.get("pitch_std_hz", 0.0))
    repeat_rate = float(text.get("repeat_rate", 0.0))
    readability = float(text.get("readability_proxy", 0.0))

    bullets: List[str] = []
    next_practice: List[str] = []

    summary = (
        f"Confidence: {bands.get('confidence', 'N/A')} ({confidence:.1f}/100), "
        f"Clarity: {bands.get('clarity', 'N/A')} ({clarity:.1f}/100), "
        f"Engagement: {bands.get('engagement', 'N/A')} ({engagement:.1f}/100)."
    )

    # Pace feedback
    _add_if(
        bullets,
        wpm > 175,
        f"Your pace is fast ({wpm:.0f} WPM). Slow slightly to make key points land more clearly."
    )
    _add_if(
        bullets,
        0 < wpm < 110,
        f"Your pace is slow ({wpm:.0f} WPM). Speeding up a little can improve engagement."
    )

    # Fillers
    _add_if(
        bullets,
        filler_rate >= 3.0,
        f"Filler words are noticeable ({filler_rate:.1f} per 100 words). Replace fillers with brief silent pauses."
    )

    # Pauses
    _add_if(
        bullets,
        mean_pause > 0.45,
        f"Pauses are long on average ({mean_pause:.2f}s). Try slightly shorter pauses to keep flow."
    )

    # Prosody
    _add_if(
        bullets,
        0 < pitch_std < 15,
        "Your delivery may sound a bit flat. Add more vocal variation by emphasising key words."
    )
    _add_if(
        bullets,
        pitch_std > 75,
        "Your pitch variation is very high. Try to keep your tone more stable for confident delivery."
    )

    # Repetition / clarity
    _add_if(
        bullets,
        repeat_rate > 0.20,
        "You repeat words/phrases fairly often. Pause briefly before restating to reduce repetition."
    )
    _add_if(
        bullets,
        0 < readability < 60,
        "Your sentences are long. Break ideas into shorter sentences for clearer delivery."
    )

    # Emotion-aware feedback (only when emotion data is available)
    if emotion_label in ("fearful", "sad"):
        bullets.append(
            "Your voice sounds nervous or downbeat. Try steady diaphragmatic breathing before you speak "
            "and remind yourself of your key message — composure comes through in your tone."
        )
    elif emotion_label == "angry":
        bullets.append(
            "Your delivery sounds intense. A measured, calm tone often reads as more confident and authoritative."
        )
    elif emotion_label in ("happy", "surprised"):
        bullets.append(
            "Your delivery sounds enthusiastic — great energy for keeping your audience engaged!"
        )
    elif emotion_label in ("calm", "neutral"):
        bullets.append(
            "Your delivery sounds composed and professional — a strong foundation for confident communication."
        )

    # If strong across the board, still give value
    if confidence < 40 and not any("confident" in bullet.lower() or "composed" in bullet.lower() for bullet in bullets):
        bullets.append(
            "Your delivery currently reads as hesitant. A steadier pace and cleaner phrasing will help you sound more confident."
        )

    if clarity < 40 and not any("clear" in bullet.lower() or "sentence" in bullet.lower() for bullet in bullets):
        bullets.append(
            "Clarity is the main area to improve. Keep each sentence focused on one idea and leave short pauses between ideas."
        )

    if engagement < 40 and not any("engage" in bullet.lower() or "energy" in bullet.lower() for bullet in bullets):
        bullets.append(
            "Engagement is limited at the moment. More vocal energy and emphasis on key words will help hold attention."
        )

    if not bullets:
        bullets.append("Your delivery is clear and well-paced. Maintain this rhythm and structure.")
        bullets.append("To improve further, add slightly more emphasis on your main message and key words.")

    # Practice tasks (simple and measurable)
    if filler_rate >= 3.0:
        next_practice.append("Do a 60-second talk and replace every filler with a 1-second pause.")
    if wpm > 175:
        next_practice.append("Re-read the same paragraph aiming for ~150 WPM using a timer.")
    if 0 < pitch_std < 15:
        next_practice.append("Record 30 seconds and emphasise 5 key words using clearer stress and intonation.")

    if not next_practice:
        next_practice.append("Record a 30-second response to an interview question: one clear point per sentence.")

    return {
        "summary": summary,
        "bullets": bullets,
        "next_practice": next_practice,
    }
