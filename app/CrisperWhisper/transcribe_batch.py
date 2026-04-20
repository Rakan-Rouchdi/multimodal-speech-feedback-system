"""
Batch transcription script for CrisperWhisper — faster-whisper backend.

Uses nyrahealth/faster_CrisperWhisper (CTranslate2 format) which is
4-8x faster than the HuggingFace transformers version on CPU and uses
much less RAM, making it practical for long (>45s) recordings locally.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    python transcribe_batch.py

Output:
    transcriptions.json  — full text + per-word timestamps for every file
"""

import os
import sys
import json
import glob
import time
import wave
import contextlib
from pathlib import Path

# ── HuggingFace authentication ────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "token").write_text(hf_token)
    print("HuggingFace token loaded from environment variable.")
else:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if not token_file.exists():
        print(
            "ERROR: No HuggingFace token found.\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
            "Get token: https://huggingface.co/settings/tokens\n"
            "Accept license: https://huggingface.co/nyrahealth/faster_CrisperWhisper"
        )
        sys.exit(1)

from faster_whisper import WhisperModel

# ── Config ─────────────────────────────────────────────────────────────────────
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), "transcriptions.json")

# faster_CrisperWhisper is the official CTranslate2 port — same weights,
# much faster CPU inference via int8 quantisation.
MODEL_ID       = "nyrahealth/faster_CrisperWhisper"

# int8 cuts memory roughly in half vs float32 with negligible accuracy loss
COMPUTE_TYPE   = "int8"
LANGUAGE       = "en"          # set to None to auto-detect per file
BEAM_SIZE      = 5
CPU_THREADS    = 10            # use most of your 12 cores (leave 2 for the OS)
TEST_ONE_FILE  = True          # ← set to False to run all recordings
# ───────────────────────────────────────────────────────────────────────────────


def get_wav_duration(filepath):
    """Return duration in seconds for WAV files; None for others."""
    if filepath.lower().endswith(".wav"):
        with contextlib.closing(wave.open(filepath)) as f:
            return round(f.getnframes() / f.getframerate(), 1)
    return None


def load_model():
    print(f"Loading model '{MODEL_ID}' (compute_type={COMPUTE_TYPE}, threads={CPU_THREADS}) …")
    model = WhisperModel(
        MODEL_ID,
        device="cpu",
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=2,   # parallel workers for feeding audio chunks
    )
    print("Model loaded.\n")
    return model


def transcribe_file(model, filepath):
    """
    Transcribe a single file and return a dict with:
      text   — full transcript string
      chunks — list of {text, start_s, end_s} word-level entries
    """
    segments, info = model.transcribe(
        filepath,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        word_timestamps=True,
        without_timestamps=False,   # segment-level timestamps kept for stitching
        vad_filter=True,            # skip silent regions → speeds up long files
        vad_parameters=dict(
            min_silence_duration_ms=300,
        ),
    )

    words = []
    full_text_parts = []

    for segment in segments:           # segments is a lazy generator
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words.append({
                    "text":    w.word,
                    "start_s": round(w.start, 3),
                    "end_s":   round(w.end, 3),
                })

    return {
        "text":     " ".join(full_text_parts),
        "language": info.language,
        "chunks":   words,
    }


def find_recordings():
    patterns = ["*.wav", "*.WAV", "*.mp3", "*.ogg", "*.flac", "*.m4a"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(RECORDINGS_DIR, pat)))
    files = sorted(files)
    return files[:1] if TEST_ONE_FILE else files


def main():
    recordings = find_recordings()
    if not recordings:
        print(f"No recordings found in {RECORDINGS_DIR}")
        sys.exit(1)

    print(f"Found {len(recordings)} recording(s):\n")
    for f in recordings:
        dur = get_wav_duration(f)
        dur_str = f"  ({dur}s)" if dur else ""
        print(f"  {os.path.basename(f)}{dur_str}")
    print()

    model = load_model()

    all_results = {}
    failed = []
    total_start = time.time()

    for idx, filepath in enumerate(recordings, 1):
        name = os.path.basename(filepath)
        dur = get_wav_duration(filepath)
        print(f"[{idx}/{len(recordings)}] {name}" + (f" ({dur}s)" if dur else "") + " …")
        t0 = time.time()
        try:
            result = transcribe_file(model, filepath)
            elapsed = time.time() - t0
            rtf = elapsed / dur if dur else 0   # real-time factor (lower = faster)
            all_results[name] = result
            preview = result["text"][:120]
            if len(result["text"]) > 120:
                preview += "…"
            print(f"  ✓  [{elapsed:.1f}s, RTF={rtf:.2f}x] {preview}\n")
        except Exception as e:
            print(f"  ✗  ERROR: {e}\n")
            failed.append(name)

    total_elapsed = time.time() - total_start

    # Save JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"Done in {total_elapsed:.1f}s — {len(all_results)} file(s) transcribed.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"Results → {OUTPUT_FILE}")
    print("=" * 70)

    print("\n── Plain-text transcriptions ──\n")
    for name, data in all_results.items():
        print(f"[{name}]")
        print(data["text"])
        print()


if __name__ == "__main__":
    main()
