from __future__ import annotations

from typing import Dict, Optional

from app.audio.preprocessing import preprocess_audio
from app.transcription.whisper_transcribe import LocalWhisperTranscriber, word_count
from app.text_analysis.metrics import compute_text_metrics
from app.speech_analysis.metrics import compute_speech_metrics
from app.speech_analysis.speech_rate import speech_rate_wpm
from app.scoring.scoring_v1 import scoring_v1
from app.feedback.generator_v1 import generate_feedback_v1
from app.output.result_builder import build_result
from app.utils.timing import Timer


def run_pipeline(file_path: str, variant: str = "multimodal") -> Dict:
    """
    Runs the system in one of three modes:
      - speech_only: compute only speech metrics
      - text_only: compute only transcription + text metrics
      - multimodal: compute both

    Returns a schema-compliant result dict.
    """
    if variant not in ("speech_only", "text_only", "multimodal"):
        raise ValueError("variant must be one of: speech_only, text_only, multimodal")

    timer = Timer()

    # --- Preprocess ---
    with timer.track("preprocess"):
        waveform, sr, duration = preprocess_audio(file_path)

    # --- Speech metrics ---
    speech_metrics: Optional[Dict] = None
    if variant in ("speech_only", "multimodal"):
        with timer.track("speech_analysis"):
            speech_metrics = compute_speech_metrics(waveform, sr)

    # --- Transcription + text metrics ---
    text_metrics: Optional[Dict] = None
    transcript = ""
    wc = 0

    if variant in ("text_only", "multimodal"):
        transcriber = LocalWhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
        with timer.track("transcription"):
            result = transcriber.transcribe(file_path)

        transcript = result.transcript
        wc = word_count(transcript)

        with timer.track("text_analysis"):
            text_metrics = compute_text_metrics(transcript)

    # --- Speech rate (WPM) ---
    # Only compute WPM when we have transcript word count.
    if speech_metrics is not None:
        if wc > 0:
            speech_metrics["speech_rate_wpm"] = speech_rate_wpm(wc, duration)
        else:
            speech_metrics["speech_rate_wpm"] = 0.0

    # --- Safe defaults for scoring (so scoring_v1 can run in any variant) ---
    safe_speech = speech_metrics or {
        "energy_mean": 0.0,
        "pause_count": 0,
        "mean_pause_sec": 0.0,
        "total_pause_sec": 0.0,
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "speech_rate_wpm": 0.0,
    }
    safe_text = text_metrics or {
        "transcript": transcript,
        "word_count": wc,
        "filler_count": 0,
        "filler_rate_per_100w": 0.0,
        "repeat_rate": 0.0,
        "readability_proxy": 0.0,
    }

    # --- Scoring / fusion ---
    with timer.track("fusion"):
        score_out = scoring_v1(speech=safe_speech, text=safe_text)

    # --- Feedback ---
    with timer.track("feedback"):
        feedback = generate_feedback_v1(
            scores=score_out["scores"],
            speech=safe_speech,
            text=safe_text,
            subscores=score_out["subscores"],
        )

    # --- Latency ---
    timer.ms["total"] = sum(timer.ms.values())
    latency_ms = {
        "preprocess": timer.ms.get("preprocess", 0.0),
        "transcription": timer.ms.get("transcription", 0.0),
        "speech_analysis": timer.ms.get("speech_analysis", 0.0),
        "text_analysis": timer.ms.get("text_analysis", 0.0),
        "fusion": timer.ms.get("fusion", 0.0),
        "feedback": timer.ms.get("feedback", 0.0),
        "total": timer.ms.get("total", 0.0),
    }

    # --- Build final result (speech_metrics/text_metrics are None when not computed) ---
    return build_result(
        variant=variant,
        source="upload",
        duration_sec=duration,
        sample_rate_hz=sr,
        scores_block=score_out["scores"],
        speech_metrics=speech_metrics,
        text_metrics=text_metrics,
        feedback=feedback,
        latency_ms=latency_ms,
    )
