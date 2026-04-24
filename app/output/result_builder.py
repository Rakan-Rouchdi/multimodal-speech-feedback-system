from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return str(uuid4())


def build_result(
    *,
    input_file: str,
    variant: str,
    source: str,
    duration_sec: float,
    sample_rate_hz: int,
    scores_block: Dict,
    speech_metrics: Optional[Dict],
    text_metrics: Optional[Dict],
    transcript: str = "",
    raw_transcript: str = "",
    emotion_output: Optional[Dict] = None,
    feedback: Optional[Dict] = None,
    warnings: Optional[list] = None,
    latency_ms: Optional[Dict] = None,
) -> Dict:
    input_path = Path(input_file)
    timestamp = utc_now_iso()
    latency = latency_ms or {
        "preprocess": 0,
        "transcription": 0,
        "speech_analysis": 0,
        "emotion_analysis": 0,
        "text_analysis": 0,
        "fusion": 0,
        "feedback": 0,
        "total": 0,
    }

    return {
        "meta": {
            "session_id": new_session_id(),
            "timestamp_utc": timestamp,
            "timestamp": timestamp,
            "filename": input_path.name,
            "input_file": str(input_path),
            "input_audio": {
                "duration_sec": float(duration_sec),
                "sample_rate_hz": int(sample_rate_hz),
                "source": source,
            },
            "pipeline_variant": variant,
        },
        "filename": input_path.name,
        "variant": variant,
        "timestamp": timestamp,
        "transcript": transcript,
        "raw_transcript": raw_transcript,
        "clean_transcript": transcript,
        "acoustic_features": speech_metrics,
        "text_features": text_metrics,
        "emotion_output": emotion_output,
        "scores": scores_block,
        "confidence_score": scores_block.get("confidence"),
        "clarity_score": scores_block.get("clarity"),
        "engagement_score": scores_block.get("engagement"),
        "bands": scores_block.get("bands", {}),
        "speech_metrics": speech_metrics,
        "text_metrics": text_metrics,
        "feedback": feedback or {"summary": "", "bullets": [], "next_practice": []},
        "latency_timings": latency,
        "debug": {
            "warnings": warnings or [],
            "latency_ms": latency,
        },
    }
