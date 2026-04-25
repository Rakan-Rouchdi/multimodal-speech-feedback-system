# Caching Implementation

## What Is Cached

Only CrisperWhisper transcription results are cached. Acoustic metrics, text metrics, scores, feedback, and JSON outputs are recomputed for each pipeline run.

Cached fields are created by `_result_to_dict()` in `app/transcription/cache.py`:

```python
{
    "model_id": MODEL_ID,
    "cleaning_version": CLEANING_VERSION,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "transcript": result.transcript,
    "clean_text": result.clean_text,
    "language": result.language,
    "segments": result.segments,
    "words": result.words,
}
```

## Cache Path

Default path:

```python
DEFAULT_CACHE_PATH = Path("outputs/cache/crisperwhisper_transcriptions.json")
```

`app.main` exposes:

```bash
--use_transcription_cache
--transcription-cache-path outputs/cache/crisperwhisper_transcriptions.json
```

`app.evaluation.run_batch` uses the cache by default and exposes:

```bash
--no_transcription_cache
--transcription-cache-path outputs/cache/crisperwhisper_transcriptions.json
```

## Cache Key

The cache key combines the filename and audio SHA256:

```python
def cache_key(file_path: str) -> str:
    path = Path(file_path)
    return f"{path.name}:{audio_sha256(file_path)}"
```

This avoids reusing a transcript when a file is edited but retains the same filename.

## Validity Checks

`_entry_is_valid(entry)` requires:

- matching `model_id`
- string `transcript`
- string `clean_text`
- non-empty `segments`
- non-empty `words`
- segment fields: `start`, `end`, `text`
- word fields: `text`, `start_s`, `end_s`

`_entry_is_plausible(entry, file_path)` rejects long WAV entries with too few words:

```python
if duration_sec is None or duration_sec < 20:
    return True
return len(entry.get("words") or []) >= 20
```

## Cleaning Version

```python
CLEANING_VERSION = "final_cleaned_output"
```

If a valid cache entry has an older cleaning version, `_refresh_cleaning()` recomputes `clean_text` without retranscribing.

## Performance and Reproducibility

Caching reduces runtime for repeated text-enabled runs because CrisperWhisper does not need to run again. It also improves reproducibility for batch evaluation because repeated runs use the same stored transcript when the audio hash, model ID, and timestamp structure are valid.

However, cached transcriptions must be invalidated when:

- the audio file changes
- the model ID changes
- the output structure lacks word timestamps
- the cached result is implausibly short for long audio
- cleaning rules change and cannot be safely refreshed

## Tests

Relevant tests in `tests/test_transcription_cache.py`:

- `test_transcription_cache_writes_then_hits`
- `test_transcription_cache_rejects_entries_without_word_timestamps`
- `test_transcription_cache_rejects_implausibly_short_long_audio`
- `test_transcription_cache_refreshes_cleaning_without_retranscribing`

## Chapter 3 Paragraph

The transcription cache was implemented as a performance optimisation for repeated local and batch evaluation runs. Since CrisperWhisper transcription is the slowest text-enabled stage, the cache stores validated transcript, segment, and word-timestamp outputs keyed by filename and audio hash. Strict validation prevents stale or incomplete entries from silently entering the evaluation pipeline.

## Limitations

- The cache is a JSON file, not a database.
- Concurrent writes are not protected by file locking.
- Cache correctness depends on the validity checks implemented in `cache.py`.
