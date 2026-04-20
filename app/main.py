import argparse

from app.pipeline.runner import run_pipeline
from app.output.save_json import save_result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/main_eval/S01_T1.wav")
    parser.add_argument(
        "--variant",
        type=str,
        default="multimodal",
        choices=["speech_only", "text_only", "multimodal"],
    )
    parser.add_argument("--use_emotion", action="store_true", help="Enable emotion analysis")
    args = parser.parse_args()

    result_json = run_pipeline(args.file, args.variant, use_emotion=args.use_emotion)

    saved_path = save_result_json(
        result_json,
        base_outputs_dir="outputs",
        dataset="main_eval",   # change to "pilot" when running pilot files
    )

    print("Saved result JSON to:", saved_path)
    print("Variant:", result_json["meta"]["pipeline_variant"])
    print("Scores:", result_json["scores"])
    print("Latency (ms):", result_json["debug"]["latency_ms"])