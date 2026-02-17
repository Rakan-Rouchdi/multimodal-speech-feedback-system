from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.pipeline.runner import run_pipeline
from app.output.save_json import save_result_json


VARIANTS = ["speech_only", "text_only", "multimodal"]


def find_audio_files(data_dir: str) -> List[Path]:
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts]
    return sorted(files)


def flatten_result(file_path: Path, result: Dict) -> Dict:
    scores = result.get("scores", {})
    bands = scores.get("bands", {})

    latency = result.get("debug", {}).get("latency_ms", {})

    speech = result.get("speech_metrics") or {}
    text = result.get("text_metrics") or {}

    row = {
        "file": str(file_path),
        "filename": file_path.name,
        "variant": result["meta"]["pipeline_variant"],
        "duration_sec": result["meta"]["input_audio"]["duration_sec"],

        # Headline scores
        "confidence": scores.get("confidence"),
        "clarity": scores.get("clarity"),
        "engagement": scores.get("engagement"),
        "confidence_band": bands.get("confidence"),
        "clarity_band": bands.get("clarity"),
        "engagement_band": bands.get("engagement"),

        # Latency
        "lat_preprocess_ms": latency.get("preprocess"),
        "lat_transcription_ms": latency.get("transcription"),
        "lat_speech_analysis_ms": latency.get("speech_analysis"),
        "lat_text_analysis_ms": latency.get("text_analysis"),
        "lat_fusion_ms": latency.get("fusion"),
        "lat_feedback_ms": latency.get("feedback"),
        "lat_total_ms": latency.get("total"),

        # A few key metrics (optional but useful)
        "speech_rate_wpm": speech.get("speech_rate_wpm"),
        "mean_pause_sec": speech.get("mean_pause_sec"),
        "pitch_std_hz": speech.get("pitch_std_hz"),
        "energy_mean": speech.get("energy_mean"),

        "word_count": text.get("word_count"),
        "filler_rate_per_100w": text.get("filler_rate_per_100w"),
        "repeat_rate": text.get("repeat_rate"),
        "readability_proxy": text.get("readability_proxy"),
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_csv", type=str, default="outputs/eval_summary.csv")
    parser.add_argument("--save_json", action="store_true", help="Also save each full JSON output into outputs/")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of audio files (0 = no limit)")
    args = parser.parse_args()

    audio_files = find_audio_files(args.data_dir)
    if args.limit and args.limit > 0:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        raise RuntimeError(f"No audio files found in {args.data_dir}. Add at least one .wav/.mp3/.m4a file.")

    rows = []
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    for f in audio_files:
        for variant in VARIANTS:
            result = run_pipeline(str(f), variant)

            if args.save_json:
                save_result_json(result, outputs_dir="outputs")

            rows.append(flatten_result(f, result))
            print(f"Processed {f.name} [{variant}] -> scores: {result['scores']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV summary to: {args.out_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
