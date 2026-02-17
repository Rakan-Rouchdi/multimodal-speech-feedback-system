import json
from datetime import datetime, timezone
from uuid import uuid4
from app.audio.preprocessing import preprocess_audio
from app.transcription.whisper_transcribe import LocalWhisperTranscriber, word_count
from app.text_analysis.metrics import compute_text_metrics
from app.speech_analysis.metrics import compute_speech_metrics
from app.speech_analysis.speech_rate import speech_rate_wpm
from app.scoring.scoring_v1 import scoring_v1
from app.output.result_builder import build_result
from app.output.save_json import save_result_json
from app.feedback.generator_v1 import generate_feedback_v1

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

    # Speech metrics (audio-only)
    speech = compute_speech_metrics(waveform, sr)
    print("\nSpeech metrics:")
    for k, v in speech.items():
        print(f"  {k}: {v}")

    # Transcription
    transcriber = LocalWhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
    result = transcriber.transcribe(file_path)

    print("\nDetected language:", result.language)
    print("Word count:", word_count(result.transcript))
    print("Transcript:\n", result.transcript)

    # Text metrics (transcript-only)
    metrics = compute_text_metrics(result.transcript)
    print("\nText metrics:")
    for k, v in metrics.items():
        if k != "transcript":
            print(f"  {k}: {v}")

    # Speech rate (WPM) + scoring
    wpm = speech_rate_wpm(word_count(result.transcript), duration)
    speech["speech_rate_wpm"] = wpm
    print(f"\nSpeech rate (WPM): {wpm:.2f}")

    score_out = scoring_v1(speech=speech, text=metrics)

    print("\nHeadline scores:")
    for k, v in score_out["scores"].items():
        if k != "bands":
            print(f"  {k}: {v}")
    print("Bands:", score_out["scores"]["bands"])

    print("\nSubscores:")
    for k, v in score_out["subscores"].items():
        print(f"  {k}: {v}")

    # Feedback (must happen BEFORE saving JSON)
    feedback = generate_feedback_v1(
        scores=score_out["scores"],
        speech=speech,
        text=metrics,
        subscores=score_out["subscores"],
    )

    print("\nFeedback summary:", feedback["summary"])
    print("Feedback bullets:")
    for b in feedback["bullets"]:
        print(" -", b)
    print("Next practice:")
    for p in feedback["next_practice"]:
        print(" -", p)

    # Build + save JSON (includes feedback)
    result_json = build_result(
        variant="multimodal",
        source="upload",
        duration_sec=duration,
        sample_rate_hz=sr,
        scores_block=score_out["scores"],
        speech_metrics=speech,
        text_metrics=metrics,
        feedback=feedback,
    )

    saved_path = save_result_json(result_json, outputs_dir="outputs")
    print("\nSaved result JSON to:", saved_path)
