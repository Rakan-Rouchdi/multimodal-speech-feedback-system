import re
from collections import Counter
from typing import Dict, List, Tuple

from app.contracts.constants import FILLER_WORDS


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


def normalise_text(text: str) -> str:
    return text.lower().strip()


def tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalise_text(text))


def count_words(text: str) -> int:
    return len(tokens(text))


def count_fillers(text: str) -> int:
    """
    Counts filler occurrences.
    Handles both single-word fillers ("um") and multi-word fillers ("you know").
    Uses word-boundary matching to avoid counting inside other words.
    """
    t = normalise_text(text)

    total = 0
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(filler.lower()) + r"\b"
        matches = re.findall(pattern, t)
        total += len(matches)

    return total


def filler_rate_per_100w(text: str) -> float:
    wc = count_words(text)
    if wc == 0:
        return 0.0
    return (count_fillers(text) / wc) * 100.0


def repetition_rate(text: str) -> float:
    """
    Proportion of unique words that repeat at least once.
    0.0 means no repeats, 1.0 means every unique word repeats.
    """
    ws = tokens(text)
    if not ws:
        return 0.0

    counts = Counter(ws)
    unique = len(counts)
    repeats = sum(1 for _, c in counts.items() if c > 1)

    return repeats / unique if unique else 0.0


def avg_sentence_length(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0

    wc = count_words(text)
    return wc / len(sentences)


def readability_proxy(text: str) -> float:
    """
    Simple, explainable proxy (0–100):
    Lower average sentence length -> higher score.
    """
    asl = avg_sentence_length(text)
    if asl == 0.0:
        return 0.0

    # Typical conversational clarity: ~8–18 words per sentence.
    # Penalise long sentences gradually.
    score = 100.0 - (asl - 10.0) * 4.0
    return float(max(0.0, min(100.0, score)))


def compute_text_metrics(transcript: str) -> Dict:
    wc = count_words(transcript)
    fc = count_fillers(transcript)
    return {
        "transcript": transcript,
        "word_count": wc,
        "filler_count": fc,
        "filler_rate_per_100w": filler_rate_per_100w(transcript),
        "repeat_rate": repetition_rate(transcript),
        "readability_proxy": readability_proxy(transcript),
    }
