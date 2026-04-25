from app.text_analysis.metrics import (
    avg_clause_length,
    compute_text_metrics,
    count_fillers,
    count_words,
    filler_rate_per_100w,
    repetition_rate,
)


def test_count_fillers_single_and_multiword():
    text = "Um I think you know this is like, kind of tricky."
    # "um" (1) + "you know" (1) + "like" (1) + "kind of" (1) = 4
    assert count_fillers(text) == 4


def test_filler_rate_per_100w_nonzero():
    text = "um " * 10 + "hello world " * 10  # 10 fillers, 20 content words
    wc = count_words(text)
    rate = filler_rate_per_100w(text, wc)
    assert rate > 0


def test_repetition_rate_behaves_on_adjacent_repeats():
    text = "I I think this is very very clear"
    assert repetition_rate(text) > 0
    assert repetition_rate("I think this is very clear") == 0.0


def test_empty_transcript_does_not_crash_text_analysis():
    metrics = compute_text_metrics("", "")
    assert metrics["raw_word_count"] == 0
    assert metrics["clean_word_count"] == 0
    assert metrics["filler_rate_per_100w"] == 0.0


def test_compute_text_metrics_exposes_raw_and_clean_transcripts():
    metrics = compute_text_metrics("Um hello", "hello")
    assert metrics["transcript"] == "hello"
    assert metrics["raw_transcript"] == "Um hello"
    assert metrics["clean_transcript"] == "hello"
    assert metrics["avg_clause_length"] == 1.0
    assert metrics["estimated_clause_count"] == 1
    assert metrics["lexical_diversity"] == 1.0


def test_avg_clause_length_is_not_broken_by_missing_full_stops():
    text = " ".join(f"word{i}" for i in range(45))
    assert avg_clause_length(text) == 15.0
