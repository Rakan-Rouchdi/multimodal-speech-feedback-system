from app.scoring.scoring import band_for, scoring


def test_scoring_returns_values_between_0_and_100():
    result = scoring(
        speech={
            "available": True,
            "speech_rate_wpm": 145.0,
            "mean_pause_sec": 0.25,
            "pitch_std_hz": 25.0,
            "energy_mean": 0.05,
            "pause_rate_per_min": 8.0,
            "pause_ratio": 0.12,
        },
        text={
            "available": True,
            "filler_rate_per_100w": 1.5,
            "repeat_rate": 0.05,
            "readability_proxy": 80.0,
        },
        use_emotion=False,
    )

    for value in result["scores"].values():
        if isinstance(value, dict):
            continue
        assert 0 <= value <= 100


def test_banding_returns_expected_category():
    assert band_for(38.9) == "Needs improvement"
    assert band_for(68.5) == "Developing"
    assert band_for(83.6) == "Strong"
    assert band_for(97.0) == "Excellent"
