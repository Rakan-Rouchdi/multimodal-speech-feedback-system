from __future__ import annotations

import argparse
from pathlib import Path

from app.output.save_json import save_result_json
from app.pipeline.runner import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multimodal speech feedback pipeline.",
    )
    parser.add_argument("--file", type=str, default="data/main_eval/S01_T1.wav")
    parser.add_argument(
        "--variant",
        type=str,
        default="multimodal",
        choices=["speech_only", "text_only", "multimodal"],
    )
    parser.add_argument(
        "--use_emotion",
        action="store_true",
        help="Enable optional CNN-BiLSTM emotion analysis for audio-enabled variants.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset label for output path naming. Defaults to the parent folder name of --file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base directory where result JSON files will be saved.",
    )
    parser.add_argument(
        "--use_transcription_cache",
        action="store_true",
        help="Reuse cached CrisperWhisper transcriptions when valid word timestamps are available.",
    )
    parser.add_argument(
        "--transcription-cache-path",
        type=str,
        default="outputs/cache/crisperwhisper_transcriptions.json",
        help="Path to the optional CrisperWhisper transcription cache.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    result_json = run_pipeline(
        str(audio_path),
        args.variant,
        use_emotion=args.use_emotion,
        use_transcription_cache=args.use_transcription_cache,
        transcription_cache_path=args.transcription_cache_path,
    )
    saved_path = save_result_json(
        result_json,
        base_outputs_dir=args.output_dir,
        dataset=args.dataset,
    )

    print("Saved result JSON to:", saved_path)
    print("Variant:", result_json["meta"]["pipeline_variant"])
    print("Scores:", result_json["scores"])
    print("Latency (ms):", result_json["debug"]["latency_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
