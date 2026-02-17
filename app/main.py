from app.audio.preprocessing import preprocess_audio
from app.transcription.whisper_transcribe import LocalWhisperTranscriber, word_count
from app.text_analysis.metrics import compute_text_metrics
from app.speech_analysis.metrics import compute_speech_metrics
from app.speech_analysis.speech_rate import speech_rate_wpm
from app.scoring.scoring_v1 import scoring_v1
from app.output.result_builder import build_result
from app.output.save_json import save_result_json
from app.feedback.generator_v1 import generate_feedback_v1
from app.utils.timing import Timer


if __name__ == "__main__":
    file_path = "data/harvard.wav"

    timer = Timer()

    # Preprocess
    with timer.track("preprocess"):
        waveform, sr, duration = preprocess_audio(file_path)

    print("Sample Rate:", sr)
    print("Duration (sec):", duration)
    print("Waveform shape:", waveform.shape)

    # Speech metrics
    with timer.track("speech_analysis"):
        speech = compute_speech_metrics(waveform, sr)

    print("\nSpeech metrics:")
    for k, v in speech.items():
        print(f"  {k}: {v}")

    # Transcription
    transcriber = LocalWhisperTranscriber(model_size="base", device="cpu", compute_type="int8")
    with timer.track("transcription"):
        result = transcriber.transcribe(file_path)

    print("\nDetected language:", result.language)
    print("Word count:", word_count(result.transcript))
    print("Transcript:\n", result.transcript)

    # Text metrics
    with timer.track("text_analysis"):
        metrics = compute_text_metrics(result.transcript)

    print("\nText metrics:")
    for k, v in metrics.items():
        if k != "transcript":
            print(f"  {k}: {v}")

    # Speech rate (WPM)
    wpm = speech_rate_wpm(word_count(result.transcript), duration)
    speech["speech_rate_wpm"] = wpm
    print(f"\nSpeech rate (WPM): {wpm:.2f}")

    # Scoring (fusion)
    with timer.track("fusion"):
        score_out = scoring_v1(speech=speech, text=metrics)

    print("\nHeadline scores:")
    for k, v in score_out["scores"].items():
        if k != "bands":
            print(f"  {k}: {v}")
    print("Bands:", score_out["scores"]["bands"])

    print("\nSubscores:")
    for k, v in score_out["subscores"].items():
        print(f"  {k}: {v}")

    # Feedback
    with timer.track("feedback"):
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

    # Total (sum of stage timings)
    timer.ms["total"] = sum(timer.ms.values())

    latency_ms = {
        "preprocess": timer.ms.get("preprocess", 0),
        "transcription": timer.ms.get("transcription", 0),
        "speech_analysis": timer.ms.get("speech_analysis", 0),
        "text_analysis": timer.ms.get("text_analysis", 0),
        "fusion": timer.ms.get("fusion", 0),
        "feedback": timer.ms.get("feedback", 0),
        "total": timer.ms.get("total", 0),
    }

    # Build + save JSON (includes feedback + latency)
    result_json = build_result(
        variant="multimodal",
        source="upload",
        duration_sec=duration,
        sample_rate_hz=sr,
        scores_block=score_out["scores"],
        speech_metrics=speech,
        text_metrics=metrics,
        feedback=feedback,
        latency_ms=latency_ms,
    )

    saved_path = save_result_json(result_json, outputs_dir="outputs")
    print("\nSaved result JSON to:", saved_path)
    print("\nLatency (ms):", latency_ms)
