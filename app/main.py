import json
from datetime import datetime, timezone
from uuid import uuid4

def make_empty_result(variant: str) -> dict:
    return {
        "meta": {
            "session_id": str(uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_audio": {"duration_sec": 0.0, "sample_rate_hz": 0, "source": "upload"},
            "pipeline_variant": variant
        },
        "scores": {"confidence": 0, "clarity": 0, "engagement": 0, "bands": {"confidence": "Needs improvement", "clarity": "Needs improvement", "engagement": "Needs improvement"}},
        "speech_metrics": None,
        "text_metrics": None,
        "feedback": {"summary": "", "bullets": [], "next_practice": []},
        "debug": {"warnings": [], "latency_ms": {"preprocess": 0, "transcription": 0, "speech_analysis": 0, "text_analysis": 0, "fusion": 0, "feedback": 0, "total": 0}}
    }

if __name__ == "__main__":
    result = make_empty_result("multimodal")
    print(json.dumps(result, indent=2))
