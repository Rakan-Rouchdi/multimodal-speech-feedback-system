from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from app.audio.preprocessing import preprocess_audio
from app.transcription.whisper_transcribe import word_count
from app.transcription.crisper_whisper import CrisperWhisperTranscriber
from app.transcription.cache import get_or_transcribe
from app.text_analysis.metrics import compute_text_metrics
from app.speech_analysis.metrics import compute_speech_metrics
from app.speech_analysis.speech_rate import speech_rate_wpm
from app.scoring.scoring_v1 import scoring_v1
from app.feedback.generator_v1 import generate_feedback_v1
from app.output.result_builder import build_result
from app.emotion.predictor import predict_emotion
from app.utils.timing import Timer


_TRANSCRIBER: Optional[CrisperWhisperTranscriber] = None


def get_transcriber() -> CrisperWhisperTranscriber:
    """
    Cache the CrisperWhisper transcriber so the model is not reloaded
    for every file.
    """
    global _TRANSCRIBER
    if _TRANSCRIBER is None:
        _TRANSCRIBER = CrisperWhisperTranscriber()
    return _TRANSCRIBER


def run_pipeline(file_path: str, variant: str = "multimodal", use_emotion: bool = False) -> Dict:
    """
    Runs the system in one of three modes:
      - speech_only: compute only speech metrics
      - text_only: compute only transcription + text metrics
      - multimodal: compute both

    Returns a schema-compliant result dict.
    """
    if variant not in ("speech_only", "text_only", "multimodal"):
        raise ValueError("variant must be one of: speech_only, text_only, multimodal")

    session_id = Path(file_path).stem
    timer = Timer()

    # --- Preprocess ---
    with timer.track("preprocess"):
        waveform, sr, duration = preprocess_audio(file_path)

    # --- Speech metrics ---
    speech_metrics: Optional[Dict] = None
    if variant in ("speech_only", "multimodal"):
        with timer.track("speech_analysis"):
            speech_metrics = compute_speech_metrics(waveform, sr)

    # --- Emotion analysis (opt-in, audio-based) ---
    emotion_data: Optional[Dict] = None
    if use_emotion and variant in ("speech_only", "multimodal"):
        with timer.track("emotion_analysis"):
            emotion_data = predict_emotion(file_path)

    # --- Transcription + text metrics ---
    text_metrics: Optional[Dict] = None
    transcript = ""

    if variant in ("text_only", "multimodal"):
        transcriber = get_transcriber()

        with timer.track("transcription"):
            result = get_or_transcribe(file_path, transcriber)

        transcript = result.transcript

        with timer.track("text_analysis"):
            text_metrics = compute_text_metrics(result.transcript, result.clean_text)

    # --- WPM ---
    # Compute whenever transcript exists, even for text_only (use clean_word_count)
    wpm_value: Optional[float] = None
    if text_metrics and text_metrics.get("clean_word_count", 0) > 0:
        wpm_value = speech_rate_wpm(text_metrics["clean_word_count"], duration)

    # Add derived pause features if speech metrics exist
    if speech_metrics is not None:
        speech_metrics["speech_rate_wpm"] = wpm_value if wpm_value is not None else 0.0

        pause_count = float(speech_metrics.get("pause_count", 0))
        total_pause_sec = float(speech_metrics.get("total_pause_sec", 0.0))

        speech_metrics["pause_rate_per_min"] = (pause_count / duration) * 60.0 if duration > 0 else 0.0
        speech_metrics["pause_ratio"] = (total_pause_sec / duration) if duration > 0 else 0.0

    # Attach emotion data to speech metrics
    if speech_metrics is not None and emotion_data is not None:
        speech_metrics["emotion"] = emotion_data

    # --- Scoring inputs ---
    # "available" flags let the scorer know whether a modality truly exists
    score_speech = dict(speech_metrics or {})
    score_speech["available"] = speech_metrics is not None

    # even text_only can provide WPM from transcript + duration
    if wpm_value is not None:
        score_speech["speech_rate_wpm"] = wpm_value

    score_text = dict(text_metrics or {})
    score_text["available"] = text_metrics is not None

    # --- Safe defaults for result / feedback only ---
    safe_speech = dict(speech_metrics or {
        "energy_mean": 0.0,
        "pause_count": 0,
        "mean_pause_sec": 0.0,
        "total_pause_sec": 0.0,
        "pitch_mean_hz": 0.0,
        "pitch_std_hz": 0.0,
        "speech_rate_wpm": 0.0,
        "pause_rate_per_min": 0.0,
        "pause_ratio": 0.0,
    })
    if wpm_value is not None:
        safe_speech["speech_rate_wpm"] = wpm_value

    safe_text = dict(text_metrics or {
        "transcript": transcript,
        "word_count": 0,
        "filler_count": 0,
        "filler_rate_per_100w": 0.0,
        "repeat_rate": 0.0,
        "readability_proxy": 0.0,
    })

    # --- Scoring / fusion ---
    with timer.track("fusion"):
        score_out = scoring_v1(speech=score_speech, text=score_text, use_emotion=use_emotion)

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
        "emotion_analysis": timer.ms.get("emotion_analysis", 0.0),
        "text_analysis": timer.ms.get("text_analysis", 0.0),
        "fusion": timer.ms.get("fusion", 0.0),
        "feedback": timer.ms.get("feedback", 0.0),
        "total": timer.ms.get("total", 0.0),
    }

    # --- Build final result ---
    result = build_result(
        variant=variant,
        source="upload",
        duration_sec=duration,
        sample_rate_hz=sr,
        scores_block=score_out["scores"],
        speech_metrics=safe_speech if variant in ("speech_only", "multimodal") else None,
        text_metrics=safe_text if variant in ("text_only", "multimodal") else None,
        feedback=feedback,
        latency_ms=latency_ms,
    )

    result["meta"]["session_id"] = session_id
    result["meta"]["input_file"] = str(file_path)
    result["meta"]["pipeline_variant"] = variant
    result["meta"]["transcription_source"] = "crisper_whisper"

    return result
