from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from app.transcription.whisper_transcribe import TranscriptionResult


CURRENT_MODEL_ID = "nyrahealth/faster_CrisperWhisper"
CLEANING_VERSION = "v1"

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "transcription_cache.json"


def _load_cache(cache_path: Path) -> Dict:
    """Load the cache JSON file, returning an empty dict if it doesn't exist."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: Path, data: Dict) -> None:
    """Persist the cache dict to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _result_to_dict(result: TranscriptionResult) -> Dict:
    """Serialise a TranscriptionResult to a JSON-safe dict."""
    return {
        "model_id": CURRENT_MODEL_ID,
        "cleaning_version": CLEANING_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "transcript": result.transcript,
        "clean_text": result.clean_text,
        "language": result.language,
        "segments": result.segments,
        "words": result.words,
    }


def _dict_to_result(d: Dict) -> TranscriptionResult:
    """Deserialise a dict back into a TranscriptionResult."""
    return TranscriptionResult(
        transcript=d["transcript"],
        clean_text=d.get("clean_text", ""),
        language=d.get("language"),
        segments=d.get("segments", []),
        words=d.get("words", []),
    )


def _convert_crisper_result(d: Dict) -> Dict:
    """
    Convert the format used in app/CrisperWhisper/results/*.json
    (keys: file, language, text, words) into our cache format
    (keys: transcript, language, segments).
    """
    segments = []
    words = d.get("words", d.get("chunks", []))
    if words:
        # Build one segment spanning the whole file
        segments.append({
            "start": words[0].get("start_s", 0.0),
            "end": words[-1].get("end_s", 0.0),
            "text": d.get("text", ""),
        })
    return {
        "transcript": d.get("text", ""),
        "language": d.get("language"),
        "segments": segments,
    }


def get_or_transcribe(
    file_path: str,
    transcriber,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> TranscriptionResult:
    """
    Look up a cached transcription for *file_path*.  If it exists, return it
    immediately.  Otherwise, run the transcriber, cache the result, and return.

    The cache key is the audio file's stem (e.g. ``S01_T1``).
    """
    key = Path(file_path).stem
    cache = _load_cache(cache_path)

    if key in cache:
        entry = cache[key]
        # Staleness check
        if (
            entry.get("model_id") == CURRENT_MODEL_ID
            and entry.get("cleaning_version") == CLEANING_VERSION
            and "created_at" in entry
            and (datetime.now() - datetime.fromisoformat(entry["created_at"])) < timedelta(days=7)
        ):
            return _dict_to_result(entry)

    # Transcribe and cache (stale or missing)
    result = transcriber.transcribe(file_path)
    cache[key] = _result_to_dict(result)
    _save_cache(cache_path, cache)

    return result


def seed_cache_from_crisper_results(
    results_dir: str,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> int:
    """
    Import any ``*.json`` files from ``results_dir`` (the format produced by
    ``app/CrisperWhisper/transcribe.py``) into the transcription cache.

    Returns the number of entries imported.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return 0

    cache = _load_cache(cache_path)
    count = 0

    for json_file in sorted(results_path.glob("*.json")):
        key = json_file.stem  # e.g. S01_T1
        if key in cache:
            continue  # don't overwrite existing

        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cache[key] = _convert_crisper_result(raw)
        count += 1

    if count > 0:
        _save_cache(cache_path, cache)

    return count
