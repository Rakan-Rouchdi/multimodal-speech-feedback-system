import json
from datetime import datetime, timezone
from uuid import uuid4
from app.audio.preprocessing import preprocess_audio
from app.transcription.whisper_transcribe import LocalWhisperTranscriber, word_count
from app.text_analysis.metrics import compute_text_metrics
from app.speech_analysis.metrics import compute_speech_metrics

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
    file_path = "data/harvard.wav"

    waveform, sr, duration = preprocess_audio(file_path)

    print("Sample Rate:", sr)
    print("Duration (sec):", duration)
    print("Waveform shape:", waveform.shape)

    speech = compute_speech_metrics(waveform, sr)
    print("\nSpeech metrics:")
    for k, v in speech.items():
        print(f"  {k}: {v}")

    transcriber = LocalWhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
    result = transcriber.transcribe(file_path)

    print("\nDetected language:", result.language)
    print("Word count:", word_count(result.transcript))
    print("Transcript:\n", result.transcript)

    metrics = compute_text_metrics(result.transcript)
    print("\nText metrics:")
    for k, v in metrics.items():
        if k != "transcript":
            print(f"  {k}: {v}")
