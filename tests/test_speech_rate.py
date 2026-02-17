from app.speech_analysis.speech_rate import speech_rate_wpm


def test_speech_rate_wpm_basic():
    # 60 words in 30 seconds -> 120 WPM
    assert abs(speech_rate_wpm(60, 30.0) - 120.0) < 1e-6
