def speech_rate_wpm(word_count: int, duration_sec: float) -> float:
    """
    Words per minute from transcript word count and audio duration.
    """
    if duration_sec <= 0:
        return 0.0
    return (word_count / duration_sec) * 60.0
