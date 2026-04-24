from app.feedback.generator import generate_feedback


def test_low_clarity_triggers_clarity_feedback():
    feedback = generate_feedback(
        scores={
            "confidence": 52.0,
            "clarity": 22.0,
            "engagement": 58.0,
            "bands": {
                "confidence": "Developing",
                "clarity": "Needs improvement",
                "engagement": "Developing",
            },
        },
        speech={"mean_pause_sec": 0.2, "speech_rate_wpm": 145.0, "pitch_std_hz": 18.0},
        text={"filler_rate_per_100w": 1.0, "repeat_rate": 0.05, "readability_proxy": 82.0},
        subscores={},
    )

    bullets = " ".join(feedback["bullets"]).lower()
    assert "clarity" in bullets or "sentence" in bullets or "clear" in bullets
