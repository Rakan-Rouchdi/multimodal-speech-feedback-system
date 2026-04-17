import re
from typing import Dict, List

from app.contracts.constants import FILLER_WORDS


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?,;:]+|\n+")


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
    Adjacent repetition rate.

    We only count immediate repeated words such as:
      - "I I think..."
      - "like like..."
      - "the the..."

    This is a better proxy for spoken disfluency than penalising every word
    that appears more than once in the whole transcript.
    """
    ws = tokens(text)
    if len(ws) < 2:
        return 0.0

    filler_set = {f.lower() for f in FILLER_WORDS if " " not in f}

    valid_pairs = 0
    repeated_pairs = 0

    for a, b in zip(ws[:-1], ws[1:]):
        # ignore adjacent filler tokens here because filler_rate already captures them
        if a in filler_set or b in filler_set:
            continue
        valid_pairs += 1
        if a == b:
            repeated_pairs += 1

    if valid_pairs == 0:
        return 0.0

    return repeated_pairs / valid_pairs


def avg_clause_length(text: str) -> float:
    """
    Use clause-like chunks rather than full sentence boundaries only.
    This works better for ASR transcripts, which often contain commas but
    weak or inconsistent full-stop punctuation.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    if not clauses:
        return 0.0

    lengths = [count_words(c) for c in clauses]
    lengths = [n for n in lengths if n > 0]
    if not lengths:
        return 0.0

    return sum(lengths) / len(lengths)


def _score_from_range(value: float, ideal_min: float, ideal_max: float, hard_min: float, hard_max: float) -> float:
    """
    Maps a metric to 0–100 where being inside [ideal_min, ideal_max] scores high.
    Outside ideal range, score drops linearly until hard bounds.
    """
    if value <= hard_min or value >= hard_max:
        return 0.0
    if ideal_min <= value <= ideal_max:
        return 100.0

    if value < ideal_min:
        return 100.0 * (value - hard_min) / (ideal_min - hard_min)

    return 100.0 * (hard_max - value) / (hard_max - ideal_max)


def readability_proxy(text: str) -> float:
    """
    Spoken-language readability proxy (0–100), based on average clause length.
    This is deliberately softer than the original sentence-length-only version.

    Ideal spoken clause length is roughly 6–16 words.
    """
    acl = avg_clause_length(text)
    if acl == 0.0:
        return 0.0

    return float(
        max(
            0.0,
            min(
                100.0,
                _score_from_range(
                    acl,
                    ideal_min=6.0,
                    ideal_max=16.0,
                    hard_min=2.0,
                    hard_max=30.0,
                ),
            ),
        )
    )


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