from __future__ import annotations

import re
from typing import Dict, List, Optional

from faster_whisper import WhisperModel

from app.transcription.types import TranscriptionResult


# ── CrisperWhisper model config ──────────────────────────────────────────────
MODEL_ID = "nyrahealth/faster_CrisperWhisper"
COMPUTE_TYPE = "int8"
CPU_THREADS = 10


def clean_transcript(raw: str) -> str:
    """Normalise CrisperWhisper raw transcript for NLP metrics."""
    # CrisperWhisper often uses commas as word separators in disfluent speech.
    text = raw.replace(",", " ")
    # Lowercase bracket tokens [UH] -> uh, [UM] -> um
    text = re.sub(r'\[([A-Z]+)\]', lambda m: m.group(1).lower(), text)
    # Remove broken stub tokens: standalone A., S., Oh. etc. (but keep Mr., Dr.)
    text = re.sub(r'\b[A-Z][a-z]*\.(?!\s[A-Z][a-z]*\.)', '', text)
    # Collapse multiple spaces, strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────


class CrisperWhisperTranscriber:
    """
    Local transcription using faster_CrisperWhisper (CTranslate2 int8).

    Key behaviour:
    - Uses only the CrisperWhisper model which transcribes disfluencies as [UH], [UM], etc.
    - Produces word-level timestamps for richer analysis.
    """

    def __init__(
        self,
        device: str = "cpu",
        compute_type: str = COMPUTE_TYPE,
        cpu_threads: int = CPU_THREADS,
    ):
        self.model = WhisperModel(
            MODEL_ID,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=2,
        )

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe an audio file and return a TranscriptionResult.

        CrisperWhisper captures disfluencies ([UH], [UM]) and provides
        word-level timestamps for downstream speech and text analysis.
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        seg_list = []
        transcript_parts = []
        word_list = []

        for seg in segments:
            seg_text = seg.text.strip()
            if seg_text:
                transcript_parts.append(seg_text)

            seg_list.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg_text,
            })

            if seg.words:
                for w in seg.words:
                    clean = w.word.lstrip(" ,")
                    if clean:
                        word_list.append({
                            "text": clean,
                            "start_s": round(w.start, 3),
                            "end_s": round(w.end, 3),
                        })

        transcript = " ".join(t for t in transcript_parts if t)

        language_detected = getattr(info, "language", None)

        clean_text = clean_transcript(transcript)

        return TranscriptionResult(
            transcript=transcript,
            language=language_detected,
            segments=seg_list,
            clean_text=clean_text,
            words=word_list,
        )
