from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from faster_whisper import WhisperModel


@dataclass
class TranscriptionResult:
    transcript: str
    language: Optional[str]
    segments: List[dict]


class LocalWhisperTranscriber:
    """
    Local transcription using faster-whisper.
    Designed to be reusable and testable.
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        # model_size: tiny, base, small, medium, large-v3 (larger = better but slower)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        segments, info = self.model.transcribe(
            audio_path,
            vad_filter=True,          # Voice activity detection to reduce silence
            beam_size=5
        )

        seg_list = []
        transcript_parts = []

        for s in segments:
            seg_list.append({
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip()
            })
            transcript_parts.append(s.text.strip())

        transcript = " ".join([t for t in transcript_parts if t])

        language = getattr(info, "language", None)

        return TranscriptionResult(
            transcript=transcript,
            language=language,
            segments=seg_list
        )


def word_count(text: str) -> int:
    # simple and robust
    return len([w for w in text.strip().split() if w])
