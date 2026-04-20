from app.text_analysis.metrics import count_fillers, filler_rate_per_100w, count_words


def test_count_fillers_single_and_multiword():
    text = "Um I think you know this is like, kind of tricky."
    # "um" (1) + "you know" (1) + "like" (1) + "kind of" (1) = 4
    assert count_fillers(text) == 4


def test_filler_rate_per_100w_nonzero():
    text = "um " * 10 + "hello world " * 10  # 10 fillers, 20 content words
    wc = count_words(text)
    rate = filler_rate_per_100w(text, wc)
    assert rate > 0
