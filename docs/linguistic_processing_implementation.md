# Linguistic Processing and Transcription Implementation

## Files

- `app/transcription/crisper_whisper.py`
- `app/transcription/types.py`
- `app/transcription/cache.py`
- `app/text_analysis/metrics.py`
- `app/pipeline/runner.py`

## CrisperWhisper Integration

Runtime transcription uses `faster_whisper.WhisperModel` with:

```python
MODEL_ID = "nyrahealth/faster_CrisperWhisper"
COMPUTE_TYPE = "int8"
CPU_THREADS = 10
```

The model is loaded in `CrisperWhisperTranscriber.__init__`:

```python
self.model = WhisperModel(
    MODEL_ID,
    device=device,
    compute_type=compute_type,
    cpu_threads=cpu_threads,
    num_workers=2,
)
```

The pipeline caches the loaded transcriber object in memory via `app/pipeline/runner.py::get_transcriber`.

## Transcription Call

`_transcribe_once()` calls:

```python
segments, info = self.model.transcribe(
    audio_path,
    language=language,
    beam_size=5,
    word_timestamps=True,
    vad_filter=vad_filter,
    vad_parameters=dict(min_silence_duration_ms=300),
)
```

The output is wrapped in `TranscriptionResult`:

```python
@dataclass
class TranscriptionResult:
    transcript: str
    language: Optional[str]
    segments: List[dict]
    clean_text: str = ""
    words: List[Dict] = field(default_factory=list)
```

## Segment and Word Timestamps

Segments are stored as:

```python
{"start": float(seg.start), "end": float(seg.end), "text": seg_text}
```

Words are stored as:

```python
{"text": clean, "start_s": round(w.start, 3), "end_s": round(w.end, 3)}
```

The cache rejects entries without segment and word timing fields.

## Transcript Cleaning

`clean_transcript(raw)`:

```python
text = raw.replace(",", " ")
text = re.sub(r'\[([A-Z]+)\]', lambda m: m.group(1).lower(), text)
text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Z])", " ", text)
text = re.sub(r'\s+', ' ', text).strip()
```

This preserves disfluencies as words (`[UH] -> uh`, `[UM] -> um`) while removing CrisperWhisper comma separators.

## Incomplete Transcript Fallback

For WAV files longer than 20 seconds, `_looks_incomplete()` returns true if fewer than 20 words are produced. In that case, transcription is retried with `vad_filter=False`.

```python
if duration_sec is None or duration_sec < 20:
    return False
if len(result.words) < 20:
    return True
```

## Text Metrics

Implemented in `app/text_analysis/metrics.py`.

Tokenisation:

```python
_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
```

Filler count:

- Counts `FILLER_WORDS` from `app/contracts/constants.py`.
- Counts CrisperWhisper bracket markers `[UH]`, `[UM]`.

Filler rate:

```python
(filler_count / clean_word_count) * 100.0
```

Repetition rate:

- Counts adjacent repeated non-filler word pairs.
- Formula: `repeated_pairs / valid_pairs`.

ASR-safe clause metrics:

- `clause_lengths()` first splits on punctuation.
- If a chunk is longer than 20 words, it is split into 20-word windows.
- `avg_clause_length` is the mean of these chunk lengths.
- `estimated_clause_count` is the number of ASR-safe chunks.

Readability proxy:

```python
readability_proxy = _score_from_range(
    avg_clause_length,
    ideal_min=6.0,
    ideal_max=16.0,
    hard_min=2.0,
    hard_max=30.0,
)
```

Lexical diversity:

```python
len(set(tokens)) / len(tokens)
```

## Edge Cases

| Edge case | Implemented behaviour |
|---|---|
| Empty transcript | Counts and rates return zero; tests confirm no crash. |
| Missing punctuation | Long ASR chunks are split into 20-word windows for clause/readability metrics. |
| Missing word timestamps in cache | Cache entry is rejected and audio is retranscribed. |
| Implausibly short long-audio cache entry | Cache entry is rejected for WAV files >=20 seconds with fewer than 20 words. |
| Malformed transcript | Metrics use regex tokenisation and default to zero for empty token lists. |

## Recent Transcript Fixes Evidenced in Code

- Comma-separated CrisperWhisper output is normalised with spaces.
- Compact `Word.Word` ASR fragments are split.
- Disfluency markers are converted to words for text metrics.
- Incomplete long-audio transcriptions trigger a VAD-disabled retry.
