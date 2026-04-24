from __future__ import annotations

import hashlib
import json
import contextlib
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.transcription.crisper_whisper import MODEL_ID
from app.transcription.crisper_whisper import clean_transcript
from app.transcription.types import TranscriptionResult


CLEANING_VERSION = "final_cleaned_output"
DEFAULT_CACHE_PATH = Path("outputs/cache/crisperwhisper_transcriptions.json")


def audio_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(file_path: str) -> str:
    path = Path(file_path)
    return f"{path.name}:{audio_sha256(file_path)}"


def _load_cache(cache_path: Path) -> Dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as cache_file:
        return json.load(cache_file)


def _save_cache(cache_path: Path, data: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(data, cache_file, ensure_ascii=False, indent=2)


def _result_to_dict(result: TranscriptionResult) -> Dict:
    return {
        "model_id": MODEL_ID,
        "cleaning_version": CLEANING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcript": result.transcript,
        "clean_text": result.clean_text,
        "language": result.language,
        "segments": result.segments,
        "words": result.words,
    }


def _entry_is_valid(entry: Dict) -> bool:
    if entry.get("model_id") != MODEL_ID:
        return False
    if not isinstance(entry.get("transcript"), str):
        return False
    if not isinstance(entry.get("clean_text"), str):
        return False
    if not isinstance(entry.get("segments"), list) or not entry["segments"]:
        return False
    if not isinstance(entry.get("words"), list) or not entry["words"]:
        return False

    first_segment = entry["segments"][0]
    first_word = entry["words"][0]
    segment_has_timing = {"start", "end", "text"}.issubset(first_segment)
    word_has_timing = {"text", "start_s", "end_s"}.issubset(first_word)
    return bool(segment_has_timing and word_has_timing)


def _entry_is_plausible(entry: Dict, file_path: str) -> bool:
    try:
        duration_sec = None
        path = Path(file_path)
        if path.suffix.lower() == ".wav":
            with contextlib.closing(wave.open(str(path))) as wav_file:
                duration_sec = wav_file.getnframes() / wav_file.getframerate()
    except Exception:
        duration_sec = None

    if duration_sec is None or duration_sec < 20:
        return True
    return len(entry.get("words") or []) >= 20


def _refresh_cleaning(entry: Dict) -> Dict:
    if entry.get("cleaning_version") == CLEANING_VERSION:
        return entry
    refreshed = dict(entry)
    refreshed["clean_text"] = clean_transcript(refreshed.get("transcript", ""))
    refreshed["cleaning_version"] = CLEANING_VERSION
    refreshed["cleaned_at"] = datetime.now(timezone.utc).isoformat()
    return refreshed


def _dict_to_result(entry: Dict) -> TranscriptionResult:
    return TranscriptionResult(
        transcript=entry["transcript"],
        clean_text=entry["clean_text"],
        language=entry.get("language"),
        segments=entry["segments"],
        words=entry["words"],
    )


def transcribe_with_cache(
    file_path: str,
    transcriber,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> tuple[TranscriptionResult, bool]:
    """
    Return a CrisperWhisper transcription and whether it came from cache.

    Cache entries are intentionally strict: if word-level timestamps are
    missing, the entry is ignored and the audio is transcribed again.
    """
    key = cache_key(file_path)
    cache = _load_cache(cache_path)
    entry: Optional[Dict] = cache.get(key)

    if entry and _entry_is_valid(entry) and _entry_is_plausible(entry, file_path):
        refreshed = _refresh_cleaning(entry)
        if refreshed != entry:
            cache[key] = refreshed
            _save_cache(cache_path, cache)
        return _dict_to_result(refreshed), True

    result = transcriber.transcribe(file_path)
    cache[key] = _result_to_dict(result)
    _save_cache(cache_path, cache)
    return result, False
