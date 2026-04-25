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

    latency = result.get("latency_ms", {})

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
        "lat_emotion_analysis_ms": latency.get("emotion_analysis"),
        "lat_text_analysis_ms": latency.get("text_analysis"),
        "lat_fusion_ms": latency.get("fusion"),
        "lat_feedback_ms": latency.get("feedback"),
        "lat_total_ms": latency.get("total"),

        # A few key metrics (optional but useful)
        "speech_rate_wpm": speech.get("speech_rate_wpm"),
        "mean_pause_sec": speech.get("mean_pause_sec"),
        "pause_ratio": speech.get("pause_ratio"),
        "pitch_std_hz": speech.get("pitch_std_hz"),
        "energy_mean": speech.get("energy_mean"),

        "word_count": text.get("clean_word_count"),
        "filler_rate_per_100w": text.get("filler_rate_per_100w"),
        "repeat_rate": text.get("repeat_rate"),
        "readability_proxy": text.get("readability_proxy"),
        "avg_clause_length": text.get("avg_clause_length"),
        "estimated_clause_count": text.get("estimated_clause_count"),
        "lexical_diversity": text.get("lexical_diversity"),
        "filler_count": text.get("filler_count"),
        "disfluency_count": text.get("disfluency_count"),
        "raw_word_count": text.get("raw_word_count"),
        "clean_word_count": text.get("clean_word_count"),
        "transcription_source": result["meta"].get("transcription_source"),
        "transcription_cache_enabled": result["meta"].get("transcription_cache_enabled"),
        "transcription_cache_hit": result["meta"].get("transcription_cache_hit"),

        # Emotion
        "emotion_top": speech.get("emotion", {}).get("top_label") if isinstance(speech.get("emotion"), dict) else None,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/main_eval")
    parser.add_argument("--out_csv", type=str, default="outputs/eval_summary.csv")
    parser.add_argument("--save_json", action="store_true", help="Also save each full JSON output into outputs/")
    parser.add_argument("--use_emotion", action="store_true", help="Enable emotion analysis (default: off)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of audio files (0 = no limit)")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["multimodal"],
        choices=[*VARIANTS, "all"],
        help="Pipeline variants to run. Default: multimodal. Use 'all' for comparison runs.",
    )
    parser.add_argument(
        "--no_transcription_cache",
        action="store_true",
        help="Disable the CrisperWhisper transcription cache during batch runs.",
    )
    parser.add_argument(
        "--transcription-cache-path",
        type=str,
        default="outputs/cache/crisperwhisper_transcriptions.json",
        help="Path to the CrisperWhisper transcription cache used by batch runs.",
    )
    args = parser.parse_args()

    audio_files = find_audio_files(args.data_dir)
    if args.limit and args.limit > 0:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        raise RuntimeError(f"No audio files found in {args.data_dir}. Add at least one .wav/.mp3/.m4a file.")

    rows = []
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    use_emotion = args.use_emotion
    variants = VARIANTS if "all" in args.variants else args.variants
    use_transcription_cache = not args.no_transcription_cache

    if len(set(variants) & {"text_only", "multimodal"}) > 1 and not use_transcription_cache:
        print(
            "Note: text_only and multimodal both run CrisperWhisper. "
            "With transcription caching disabled, each file will be transcribed once per text-enabled variant."
        )

    for f in audio_files:
        for variant in variants:
            result = run_pipeline(
                str(f),
                variant,
                use_emotion=use_emotion,
                use_transcription_cache=use_transcription_cache,
                transcription_cache_path=args.transcription_cache_path,
            )

            if args.save_json:
                save_result_json(result, base_outputs_dir="outputs")

            rows.append(flatten_result(f, result))
            print(f"Processed {f.name} [{variant}] -> scores: {result['scores']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV summary to: {args.out_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
