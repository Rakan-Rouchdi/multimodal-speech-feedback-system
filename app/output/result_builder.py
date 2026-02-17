from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from typing import Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return str(uuid4())


def build_result(
    *,
    variant: str,
    source: str,
    duration_sec: float,
    sample_rate_hz: int,
    scores_block: Dict,
    speech_metrics: Optional[Dict],
    text_metrics: Optional[Dict],
    feedback: Optional[Dict] = None,
    warnings: Optional[list] = None,
    latency_ms: Optional[Dict] = None,
) -> Dict:
    return {
        "meta": {
            "session_id": new_session_id(),
            "timestamp_utc": utc_now_iso(),
            "input_audio": {
                "duration_sec": float(duration_sec),
                "sample_rate_hz": int(sample_rate_hz),
                "source": source,
            },
            "pipeline_variant": variant,
        },
        "scores": scores_block,
        "speech_metrics": speech_metrics,
        "text_metrics": text_metrics,
        "feedback": feedback or {"summary": "", "bullets": [], "next_practice": []},
        "debug": {
            "warnings": warnings or [],
            "latency_ms": latency_ms
            or {
                "preprocess": 0,
                "transcription": 0,
                "speech_analysis": 0,
                "text_analysis": 0,
                "fusion": 0,
                "feedback": 0,
                "total": 0,
            },
        },
    }
