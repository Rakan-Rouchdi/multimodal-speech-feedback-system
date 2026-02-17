from app.scoring.scoring_v1 import band_for


def test_band_for_boundaries():
    assert band_for(0) == "Needs improvement"
    assert band_for(39) == "Needs improvement"
    assert band_for(40) == "Developing"
    assert band_for(69) == "Developing"
    assert band_for(70) == "Strong"
    assert band_for(84) == "Strong"
    assert band_for(85) == "Excellent"
    assert band_for(100) == "Excellent"
