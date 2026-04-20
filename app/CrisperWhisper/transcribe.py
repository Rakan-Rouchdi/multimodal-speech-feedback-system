"""
CrisperWhisper CLI — transcribe a single audio file.

Usage:
    python transcribe.py recordings/S01_T1.wav
    python transcribe.py recordings/S01_T1.wav --out results/S01_T1.json
    python transcribe.py recordings/S01_T1.wav --language fr
    python transcribe.py recordings/S01_T1.wav --text-only
"""

import argparse
import contextlib
import json
import os
import sys
import time
import wave
from pathlib import Path

# ── HuggingFace auth ──────────────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "token").write_text(hf_token)

from faster_whisper import WhisperModel

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID     = "nyrahealth/faster_CrisperWhisper"
COMPUTE_TYPE = "int8"
CPU_THREADS  = 10
# ─────────────────────────────────────────────────────────────────────────────


def get_duration(filepath: str) -> float | None:
    if filepath.lower().endswith(".wav"):
        with contextlib.closing(wave.open(filepath)) as f:
            return round(f.getnframes() / f.getframerate(), 1)
    return None


def load_model() -> WhisperModel:
    print(f"Loading model … (first run downloads ~1.5 GB, cached after that)")
    model = WhisperModel(
        MODEL_ID,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=2,
    )
    return model


def transcribe(model: WhisperModel, filepath: str, language: str = "en") -> dict:
    segments, info = model.transcribe(
        filepath,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                # faster-whisper uses leading commas/spaces as separators — strip them
                clean = w.word.lstrip(" ,")
                if clean:
                    words.append({
                        "text":    clean,
                        "start_s": round(w.start, 3),
                        "end_s":   round(w.end, 3),
                    })

    # Rebuild clean text from the sanitised word list
    text = " ".join(w["text"] for w in words)

    return {
        "file":     os.path.basename(filepath),
        "language": info.language,
        "text":     text,
        "words":    words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with CrisperWhisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file",            help="Path to audio file (wav, mp3, flac, …)")
    parser.add_argument("--out",           help="Output JSON path (default: <filename>.json beside the audio)")
    parser.add_argument("--language",      default="en", help="Language code, e.g. en, fr, de (default: en)")
    parser.add_argument("--text-only",     action="store_true", help="Print only the transcript, no JSON saved")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    # Determine output path
    if not args.text_only:
        if args.out:
            out_path = args.out
        else:
            stem = Path(args.file).stem
            out_path = str(Path(args.file).parent / f"{stem}.json")

    duration = get_duration(args.file)
    dur_str  = f" ({duration}s)" if duration else ""

    print(f"Transcribing: {os.path.basename(args.file)}{dur_str}")
    model = load_model()

    t0     = time.time()
    result = transcribe(model, args.file, language=args.language)
    elapsed = time.time() - t0
    rtf    = f"RTF={elapsed/duration:.2f}x" if duration else f"{elapsed:.1f}s"

    print(f"Done in {elapsed:.1f}s  [{rtf}]\n")
    print("── Transcript ──────────────────────────────────────────────────────")
    print(result["text"])
    print("────────────────────────────────────────────────────────────────────")

    if not args.text_only:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
