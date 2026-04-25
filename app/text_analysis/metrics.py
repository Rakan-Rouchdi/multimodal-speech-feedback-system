import re
from typing import Dict, List, Optional

from app.contracts.constants import DISFLUENCY_MARKERS, FILLER_WORDS


_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
_BRACKET_TOKEN_RE = re.compile(r"\[([A-Z]+)\]")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?,;:]+|\n+")
_MAX_ASR_CLAUSE_WORDS = 20


def normalise_text(text: str) -> str:
    return text.lower().strip()


def tokens(text: str) -> List[str]:
    return _WORD_RE.findall(normalise_text(text))


def count_words(text: str) -> int:
    return len(tokens(text))


def count_disfluencies(text: str) -> int:
    """
    Count CrisperWhisper bracket disfluency markers like [UH], [UM].
    These are explicit speech disfluencies detected by the model.
    """
    matches = _BRACKET_TOKEN_RE.findall(text)
    markers_upper = {m.strip("[]") for m in DISFLUENCY_MARKERS}
    return sum(1 for m in matches if m in markers_upper)


def count_fillers(text: str) -> int:
    """
    Counts filler occurrences.
    Handles both single-word fillers ("um") and multi-word fillers ("you know").
    Uses word-boundary matching to avoid counting inside other words.
    Also counts CrisperWhisper disfluency markers ([UH], [UM]).
    """
    t = normalise_text(text)

    total = 0
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        pattern = r"\b" + re.escape(filler.lower()) + r"\b"
        matches = re.findall(pattern, t)
        total += len(matches)

    # Also count bracket disfluency markers from CrisperWhisper
    total += count_disfluencies(text)

    return total


def filler_rate_per_100w(transcript: str, clean_word_count: int) -> float:
    if clean_word_count == 0:
        return 0.0
    fc = count_fillers(transcript)  # raw transcript for [UH]/[UM] detection
    return (fc / clean_word_count) * 100.0


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


def clause_lengths(text: str, max_words: int = _MAX_ASR_CLAUSE_WORDS) -> List[int]:
    """
    Return punctuation-aware, ASR-safe clause lengths.

    CrisperWhisper punctuation is often sparse, so a whole paragraph can arrive
    as one "sentence". Very long punctuation chunks are therefore split into
    fixed word windows. This keeps readability features from being dominated by
    missing full stops.
    """
    rough_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]
    lengths: List[int] = []

    for clause in rough_clauses:
        words = tokens(clause)
        if not words:
            continue
        if len(words) <= max_words:
            lengths.append(len(words))
            continue

        for start in range(0, len(words), max_words):
            lengths.append(len(words[start:start + max_words]))

    return lengths


def avg_clause_length(text: str) -> float:
    """
    Average length of punctuation-aware, ASR-safe spoken chunks.
    """
    lengths = clause_lengths(text)
    if not lengths:
        return 0.0

    return sum(lengths) / len(lengths)


def lexical_diversity(text: str) -> float:
    """
    Ratio of unique words to total words.

    This is a simple proxy for vocabulary variety. It is intentionally kept
    lightweight because transcripts are short and ASR punctuation is imperfect.
    """
    ws = tokens(text)
    if not ws:
        return 0.0
    return len(set(ws)) / len(ws)


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


def compute_text_metrics(transcript: str, clean_text: Optional[str] = None) -> Dict:
    if clean_text is None:
        clean_text = transcript  # fallback, will be broken but consistent
    
    wc_raw = count_words(transcript)
    wc_clean = count_words(clean_text)
    fc = count_fillers(transcript)  # raw for disfluency detection
    dc = count_disfluencies(transcript)  # raw
    
    return {
        "transcript": clean_text,
        "raw_transcript": transcript,
        "clean_transcript": clean_text,
        "raw_word_count": wc_raw,
        "clean_word_count": wc_clean,
        "filler_count": fc,
        "disfluency_count": dc,
        "filler_rate_per_100w": filler_rate_per_100w(transcript, wc_clean),
        "repeat_rate": repetition_rate(clean_text),
        "readability_proxy": readability_proxy(clean_text),
        "avg_clause_length": avg_clause_length(clean_text),
        "estimated_clause_count": len(clause_lengths(clean_text)),
        "lexical_diversity": lexical_diversity(clean_text),
    }
