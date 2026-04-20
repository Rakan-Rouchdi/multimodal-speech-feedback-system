from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("outputs/main_eval/raw")
OUT_CSV = Path("outputs/main_eval/main_eval_results.csv")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_result(data: dict) -> dict:
    meta = data.get("meta", {})
    scores = data.get("scores", {})
    speech = data.get("speech_metrics") or {}
    text = data.get("text_metrics") or {}
    debug = data.get("debug", {})
    latency = debug.get("latency_ms", {})
    input_audio = meta.get("input_audio", {})

    recording_id = meta.get("session_id", "")
    parts = recording_id.split("_")
    speaker_id = parts[0] if len(parts) >= 1 else ""
    task = parts[1] if len(parts) >= 2 else ""

    row = {
        "recording_id": recording_id,
        "speaker_id": speaker_id,
        "task": task,
        "variant": meta.get("pipeline_variant", ""),
        "source_file": meta.get("input_file", ""),
        "duration_sec": input_audio.get("duration_sec", None),
        "sample_rate_hz": input_audio.get("sample_rate_hz", None),

        "confidence": scores.get("confidence", None),
        "clarity": scores.get("clarity", None),
        "engagement": scores.get("engagement", None),

        "energy_mean": speech.get("energy_mean", None),
        "pause_count": speech.get("pause_count", None),
        "mean_pause_sec": speech.get("mean_pause_sec", None),
        "total_pause_sec": speech.get("total_pause_sec", None),
        "pitch_mean_hz": speech.get("pitch_mean_hz", None),
        "pitch_std_hz": speech.get("pitch_std_hz", None),
        "speech_rate_wpm": speech.get("speech_rate_wpm", None),

        "word_count": text.get("word_count", None),
        "filler_count": text.get("filler_count", None),
        "filler_rate_per_100w": text.get("filler_rate_per_100w", None),
        "repeat_rate": text.get("repeat_rate", None),
        "readability_proxy": text.get("readability_proxy", None),
        "transcript": text.get("transcript", None),

        "latency_preprocess_ms": latency.get("preprocess", None),
        "latency_transcription_ms": latency.get("transcription", None),
        "latency_speech_analysis_ms": latency.get("speech_analysis", None),
        "latency_text_analysis_ms": latency.get("text_analysis", None),
        "latency_fusion_ms": latency.get("fusion", None),
        "latency_feedback_ms": latency.get("feedback", None),
        "latency_total_ms": latency.get("total", None),
    }
    return row


def main() -> None:
    rows = []

    for json_path in sorted(RAW_DIR.rglob("*.json")):
        data = load_json(json_path)
        rows.append(flatten_result(data))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["recording_id", "variant"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()